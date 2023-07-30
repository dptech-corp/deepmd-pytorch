"""ASE calculator interface module."""

from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Union,
)

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

import torch
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.env import DEVICE, JIT
from deepmd_pt.utils.preprocess import Region3D, make_env_mat

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )

__all__ = ["DP"]


class DP(Calculator):
    """Implementation of ASE deepmd_pytorch calculator.

    Implemented propertie are `energy`, `forces` and `stress`

    Parameters
    ----------
    model : Union[str, Path]
        path to the model
    label : str, optional
        calculator label, by default "DP"
    type_dict : Dict[str, int], optional
        mapping of element types and their numbers, best left None and the calculator
        will infer this information from model, by default None

    Examples
    --------
    Compute potential energy

    >>> from ase import Atoms
    >>> from deepmd_pt.calculator import DP
    >>> water = Atoms('H2O',
    >>>             positions=[(0.7601, 1.9270, 1),
    >>>                        (1.9575, 1, 1),
    >>>                        (1., 1., 1.)],
    >>>             cell=[100, 100, 100],
    >>>             calculator=DP(model="frozen_model.pb"))
    >>> print(water.get_potential_energy())
    >>> print(water.get_forces())

    Run BFGS structure optimization

    >>> from ase.optimize import BFGS
    >>> dyn = BFGS(water)
    >>> dyn.run(fmax=1e-6)
    >>> print(water.get_positions())
    """

    name = "DP"
    implemented_properties = ["energy", "free_energy", "forces", "virial", "stress"]

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP",
        type_dict: Dict[str, int] = None,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)

        state_dict = torch.load(model, map_location=DEVICE)
        model_params = state_dict.pop("_extra_state")["model_params"]
        model_params["resuming"] = True

        state_dict_unwrap = {str(k)[14:]: v for k, v in state_dict.items()}  # Remove "model.Default." from the key
        self.dp = get_model(model_params).to(DEVICE)
        self.dp.load_state_dict(state_dict_unwrap)

        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = {element: i for i, element in enumerate(model_params["type_map"])}

    def calculate_impl(self, coords, cells, atom_types, type_split=True):
        rcut = self.dp.descriptor.rcut
        sec = self.dp.descriptor.sec
        # sec = torch.cumsum(torch.tensor(sel, dtype=torch.int32), dim=0)
        # still problematic
        if cells is not None:
            region = Region3D(cells)
        else:
            region = None
        if type(atom_types[0]) == str:
            atom_types = [self.type_dict[k] for k in atom_types]
        # inputs: coord, atype, regin; rcut, sec

        # add batch dim for atom types
        batch_coord, batch_atype = torch.tensor(coords), torch.unsqueeze(torch.tensor(atom_types), 0)
        # build batch env_mat
        batch_selected, batch_selected_loc, batch_selected_type, batch_shift, batch_mapping = \
            [torch.stack(tensors) for tensors in
             zip(*[make_env_mat(coord, batch_atype[0], region, rcut, sec, type_split=type_split, pbc=region is not None)
                   for coord in batch_coord])]
        # inference, assumes pbc
        ret = self.dp(
            batch_coord, batch_atype, None,
            batch_mapping, batch_shift,
            batch_selected, batch_selected_type, batch_selected_loc,
            box=cells,
        )

        return ret

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces", "virial"],
        system_changes: List[str] = all_changes,
    ):
        """Run calculation with deepmd model.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            atoms object to run the calculation on, by default None
        properties : List[str], optional
            unused, only for function signature compatibility,
            by default ["energy", "forces", "stress"]
        system_changes : List[str], optional
            unused, only for function signature compatibility, by default all_changes
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        coord = self.atoms.get_positions().reshape([-1, 3])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        batch_results = self.calculate_impl(coords=coord[None, :, :], cells=cell, atom_types=atype)
        e, f, v = batch_results["energy"], batch_results["force"], batch_results["virial"]

        self.results = {
            "energy": e[0][0],
            "free_energy": e[0][0],
            "forces": f[0],
            "virial": v[0].reshape(3, 3),
        }

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            if sum(atoms.get_pbc()) > 0:
                # the usual convention (tensile stress is positive)
                # stress = -virial / volume
                stress = -0.5 * (v[0].copy() + v[0].copy().T) / atoms.get_volume()
                # Voigt notation
                self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError