"""ASE calculator interface module."""
import traceback
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

from deepmd_pt.infer.inference import load_unwrapped_model, inference_multiconf, inference_singleconf

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
except:
    traceback.print_exc()
    Calculator = object
    PropertyNotImplementedError = Exception
    all_changes = []

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
    >>>             calculator=DP(model="frozen_model.pt"))
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

        self.dp, type_dict_loaded = load_unwrapped_model(model)
        self.type_dict = type_dict if type_dict is not None else type_dict_loaded

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
            cells = cell[None, :, :]
        else:
            cell = cells = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        batch_results = inference_multiconf(self.dp, coords=coord[None, :, :], cells=cells, atom_types=atype)
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
