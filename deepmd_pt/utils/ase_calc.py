from ase import Atoms
from ase.calculators.calculator import Calculator
from deepmd_pt.infer.deep_eval import DeepPot
import numpy as np
import dpdata


class DPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
            self,
            model
    ):
        Calculator.__init__(self)
        self.dp = DeepPot(model)
        self.type_map = self.dp.type_map

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        system = dpdata.System(atoms, fmt="ase/structure")
        type_trans = np.array([self.type_map.index(i) for i in system.data['atom_names']])
        input_coords = system.data['coords']
        input_cells = system.data['cells']
        input_types = list(type_trans[system.data['atom_types']])
        model_predict = self.dp.eval(input_coords, input_cells, input_types)
        self.results["energy"] = model_predict[0].item()
        self.results["forces"] = model_predict[1].reshape(-1, 3)
