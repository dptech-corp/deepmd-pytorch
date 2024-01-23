

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

from deepmd_utils.model_format import FittingOutputDef
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
        pass #TODO

    def get_rcut(self)->float:
        return self.rcut
    
    def get_sel(self)->int:
        return self.sel
    
    def distinguish_types(self)->bool:
        # this model has no descriptor, thus no type_split.
        return
        
    # Since ZBL model is different from other AtomicModels, overwritting it here
    def forward_atomic(
        self,
        extended_coord, 
        extended_atype, 
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:

        #this should get atomic energy for all local atoms?
        

        nframes, nloc, nnei = nlist.shape
        # atype = extended_atype[:, :nloc]
        
        """
        below is the sudo code, need to figure out how the index works.

        atomic_energy = torch.zeros(nloc)
        for a_loc in range(nloc):
            
            for a_nei in range(nnei):
                # there will be duplicated calculation (pairwise), maybe cache it somewhere.
                # removing _pair_tab_jloop method, just unwrap here.

                cur_table_data --> subtable based on atype.
                dr = extended_coord[:, a_loc] - extended_coord[:, a_nei]
                pairwise_ene = self._pair_tabulated_inter(cur_table_data, dr)
                atomic_energy[a_loc] += pairwise_ene
                
        return {"atomic_energy": atomic_energy} --> convert to FittingOutputDef

        """

    def _pair_tabulated_inter(self, cur_table_data: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        """Pairwise tabulated energy.
        
        Parameters
        ----------
        cur_table_data : torch.Tensor
            The tabulated cubic spline coefficients for the current atom types.

        dr : torch.Tensor
            The distance vector between two atoms.
        
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
        ndata = nspline * 4

        r2 = dr[0]**2 + dr[1]**2 + dr[2]**2
        rr = torch.sqrt(r2) # scaler distance

        uu = (rr - rmin) * hi

        if uu < 0:
            raise Exception("coord go beyond table lower boundary")

        idx = int(uu)

        if idx >= nspline:
            ener = 0
            fscale = 0
            return

        uu -= idx

        a3 = cur_table_data[4 * idx + 0]
        a2 = cur_table_data[4 * idx + 1] 
        a1 = cur_table_data[4 * idx + 2]
        a0 = cur_table_data[4 * idx + 3]

        etmp = (a3 * uu + a2) * uu + a1
        ener = etmp * uu + a0
        return ener



