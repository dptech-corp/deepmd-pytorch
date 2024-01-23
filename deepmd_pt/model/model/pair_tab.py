

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
from deepmd.utils.pair_tab import (
    PairTab,
)
import torch
import numpy as np
from typing import Dict, List, Optional, Union

from deepmd_utils.model_format import FittingOutputDef, OutputVariableDef
from deepmd_pt.model.task import Fitting

class PairTabModel(AtomicModel):
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
        atype = extended_atype[:, :nloc] #this is the atype for local atoms, nframes * nloc
        pairwise_dr = self._get_pairwise_dist(extended_coord)

        self.tab_data = self.tab_data.reshape(self.tab.ntypes,self.tab.ntypes,self.tab.nspline,4)

        atomic_energy = torch.zeros(nloc)

        for atom_i in range(nloc):
            i_type = atype[:,i_type] # not quite sure about this on frame dimension
            for atom_j in range(nnei):
                j_idx = nnei[atom_j]
                j_type = extended_atype[:,j_idx] #same here
                rr = pairwise_dr[atom_i][atom_j].pow(2).sum().sqrt()

                # need to handle i_type and j_type frame dimension
                pairwise_ene = self._pair_tabulated_inter(i_type,j_type,rr)
                atomic_energy[atom_i] += pairwise_ene
        
        return {"atomic_energy": atomic_energy}

    def _pair_tabulated_inter(self, i_type: int, j_type: int, rr: torch.Tensor) -> torch.Tensor:
        """Pairwise tabulated energy.
        
        Parameters
        ----------
        i_type : int
            The integer representation of atom type for atom i.

        j_type : int
            The integer representation of atom type for atom j.

        rr : torch.Tensor
            The salar distance vector between two atoms.
        
        Returns
        -------
        torch.Tensor
            The energy between two atoms.
        
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

        uu = (rr - rmin) * hi

        if uu < 0:
            raise Exception("coord go beyond table lower boundary")

        idx = int(uu)

        if idx >= nspline:
            ener = 0
            # fscale = 0
            return

        uu -= idx

        a3, a2, a1, a0 = self.tab_data[i_type][j_type][idx]

        etmp = (a3 * uu + a2) * uu + a1
        ener = etmp * uu + a0
        return ener

    @staticmethod
    def _get_pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : torch.Tensor
            The coordinate of the atoms.

        Returns
        -------
        torch.Tensor
            The pairwise distance between the atoms.
        
        Examples
        --------
        coords = torch.tensor([
                [0,0,0],
                [1,3,5],
                [2,4,6]
            ])
        
        dist = tensor([
            [[ 0,  0,  0],
            [-1, -3, -5],
            [-2, -4, -6]],

            [[ 1,  3,  5],
            [ 0,  0,  0],
            [-1, -1, -1]],

            [[ 2,  4,  6],
            [ 1,  1,  1],
            [ 0,  0,  0]]
            ])
        """
        return coords.unsqueeze(1) - coords.unsqueeze(0)

