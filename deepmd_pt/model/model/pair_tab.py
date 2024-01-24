

# This Model will be a concrete class of AtomicModel
# 1. to get python code from tensorflow (PairTabModel, PariTab)
#       a) data process
#       b) get cubic spline
# 2. to translate c++ code, use cubic spline para to calculate energy
#   overwrite foward_lower to excldue descriptor and fitting_net


# This model will be used by an interface of HybridEnergyModel
# 1. take a list of AtomicModels and a weightfunc to define hybird energy
# 2. define the derivatives (force and virial)
# 3. define a translation layer (dict map for user outputs)

from .atomic_model import AtomicModel
from deepmd_utils.pair_tab import (
    PairTab,
)
import torch
from torch import nn
import numpy as np
from typing import Dict, List, Optional, Union

from deepmd_utils.model_format import FittingOutputDef, OutputVariableDef
from deepmd_pt.model.task import Fitting

class PairTabModel(nn.Module, AtomicModel):
    """Pairwise tabulation energy model.

    This model can be used to tabulate the pairwise energy between atoms for either
    short-range or long-range interactions, such as D3, LJ, ZBL, etc. It should not
    be used alone, but rather as one submodel of a linear (sum) model, such as
    DP+D3.

    Do not put the model on the first model of a linear model, since the linear
    model fetches the type map from the first model.

    At this moment, the model does not smooth the energy at the cutoff radius, so
    one needs to make sure the energy has been smoothed to zero.

    Parameters
    ----------
    tab_file : str
        The path to the tabulation file.
    rcut : float
        The cutoff radius.
    sel : int or list[int]
        The maxmum number of atoms in the cut-off radius.
    """

    def __init__(
        self, tab_file: str, rcut: float, sel: Union[int, List[int]], **kwargs
    ):
        super().__init__() 
        self.tab_file = tab_file
        self.tab = PairTab(self.tab_file)
        self.ntypes = self.tab.ntypes
        self.rcut = rcut

        tab_info, tab_data = self.tab.get() # this returns -> Tuple[np.array, np.array]
        self.tab_info = torch.from_numpy(tab_info) 
        self.tab_data = torch.from_numpy(tab_data)

        # self.model_type = "ener"
        # self.model_version = MODEL_VERSION ## this shoud be in the parent class


        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")
    
    def get_fitting_net(self)->Fitting:
        # this model has no fitting_net.
        return
    
    def get_fitting_output_def(self)->FittingOutputDef:
        return FittingOutputDef(
            [OutputVariableDef(
                name="energy",
                shape=[1],
                reduciable=True,
                differentiable=True
            )]
        )

    def get_rcut(self)->float:
        return self.rcut
    
    def get_sel(self)->int:
        return self.sel
    
    def distinguish_types(self)->bool:
        # to match DPA1 and DPA2.
        return False
        
    def forward_atomic(
        self,
        extended_coord, 
        extended_atype, 
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
        

        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc] #this is the atype for local atoms, (nframes, nloc)
        pairwise_dr = self._get_pairwise_dist(extended_coord) # (nframes, nall, nall, 3)
        pairwise_rr = pairwise_dr.pow(2).sum(-1).sqrt() # (nframes, nall, nall), this is the pairwise scalar distance for all atoms in all frames.

        self.tab_data = self.tab_data.reshape(self.tab.ntypes,self.tab.ntypes,self.tab.nspline,4)

        atomic_energy = torch.zeros(nframes, nloc)

        for atom_i in range(nloc):
            i_type = atype[:,atom_i] # (nframes, 1)
            for atom_j in range(nnei):
                j_idx = nnei[atom_j]
                j_type = extended_atype[:,j_idx] # (nframes, 1)
                
                rr = pairwise_rr[:, atom_i, j_idx] # (nframes, 1)

                # the input shape is (nframes, 1), (nframes, 1), (nframes, 1),
                # the expected output shape then becomes (nframes,1)
                pairwise_ene = self._pair_tabulated_inter(i_type, j_type, rr)
                atomic_energy[:, atom_i] += pairwise_ene
        
        return {"atomic_energy": atomic_energy}

    def _pair_tabulated_inter(self, i_type: torch.Tensor, j_type: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """Pairwise tabulated energy.
        
        Parameters
        ----------
        i_type : torch.Tensor
            The integer representation of atom type for atom i for all frames.

        j_type : torch.Tensor
            The integer representation of atom type for atom j for all frames.

        rr : torch.Tensor
            The salar distance vector between two atoms  for all frames.
        
        Returns
        -------
        torch.Tensor
            The energy between two atoms for all frames.
        
        Raises
        ------
        Exception
            If the distance is beyond the table.
        
        Notes
        -----
        This function is used to calculate the pairwise energy between two atoms.
        It uses a table containing cubic spline coefficients calculated in PairTab.
        """

        rmin = self.tab_info[0] 
        hh = self.tab_info[1]
        hi = 1. / hh

        nspline = int(self.tab_info[2] + 0.1)

        uu = (rr - rmin) * hi # this is broadcasted to (nframes,1)

        if any(uu < 0):
            raise Exception("coord go beyond table lower boundary")

        idx = uu.to(torch.int)

        uu -= idx
        cur_tab = self.tab_data[i_type.squeeze(),j_type.squeeze()] # this should have shape (nframes, nspline, 4)
        

        # we need to check in the elements in the index tensor (nframes, 1) to see if they are beyond the table.
        # when the spline idx is beyond the table, all spline coefficients are set to `0`, and the resulting ener corresponding to the idx is also `0`.
        final_coef = self._extract_spline_coefficient(cur_tab, idx) # this should have shape (nframes, 4)

        a3, a2, a1, a0 = final_coef[:,0], final_coef[:,1], final_coef[:,2], final_coef[:,3] # the four coefficients should all be (nframes, 1)

        etmp = (a3 * uu + a2) * uu + a1 # this should be elementwise operations.
        ener = etmp * uu + a0
        return ener

    @staticmethod
    def _get_pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : torch.Tensor
            The coordinate of the atoms shape of (nframes * nall * 3).

        Returns
        -------
        torch.Tensor
            The pairwise distance between the atoms (nframes * nall * nall * 3).
        
        Examples
        --------
        coords = torch.tensor([[
                [0,0,0],
                [1,3,5],
                [2,4,6]
            ]])
        
        dist = tensor([[
            [[ 0,  0,  0],
            [-1, -3, -5],
            [-2, -4, -6]],

            [[ 1,  3,  5],
            [ 0,  0,  0],
            [-1, -1, -1]],

            [[ 2,  4,  6],
            [ 1,  1,  1],
            [ 0,  0,  0]]
            ]])
        """
        return coords.unsqueeze(2) - coords.unsqueeze(1)

    @staticmethod
    def _extract_spline_coefficient(cur_tab: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Extract the spline coefficient from the table.

        Parameters
        ----------
        cur_tab : torch.Tensor
            The table containing the spline coefficients. (nframes, nspline, 4)

        idx : torch.Tensor
            The index of the spline coefficient. (nframes, 1)

        Returns
        -------
        torch.Tensor
            The spline coefficient. (nframes, 4)

        Example
        -------
        cur_tab = tensor([[[0, 3, 1, 3],
                         [1, 1, 2, 1],
                         [3, 1, 3, 3]],

                        [[3, 1, 3, 1],
                         [2, 3, 1, 0],
                         [2, 1, 3, 1]],

                        [[3, 0, 2, 2],
                         [2, 3, 1, 1],
                         [1, 2, 3, 1]],

                        [[0, 3, 3, 0],
                         [1, 3, 0, 3],
                         [3, 1, 2, 3]],

                        [[3, 0, 3, 3],
                         [1, 1, 1, 0],
                         [3, 1, 2, 3]]])

        idx = tensor([[1],
                        [0],
                        [2],
                        [5],
                        [1]])
        

        final_coef = tensor([[[1, 1, 2, 1]],

                            [[3, 1, 3, 1]],

                            [[1, 2, 3, 1]],

                            [[0, 0, 0, 0]],

                            [[1, 1, 1, 0]]])
        """
        clipped_indices = torch.clamp(idx, 0, cur_tab.shape[1] - 1)
        final_coef = torch.gather(cur_tab, 1, clipped_indices.unsqueeze(-1).expand(-1, -1, cur_tab.shape[-1]))
        final_coef[idx.squeeze() >= cur_tab.shape[1]] = 0

        return final_coef