import json
import os
import torch
import unittest
from copy import deepcopy

from deepmd_pt.entrypoints.main import get_trainer
from deepmd_pt.infer import inference
from deepmd_pt.utils.ase_calc import DPCalculator

dtype = torch.float64


class TestCalculator(unittest.TestCase):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["validation_data"]["systems"] = ["tests/water/data/single"]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        input_dict, label_dict, _ = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        self.calculator = DPCalculator("model.pt")

    def test_calculator(self):
        from ase import Atoms

        natoms = 5
        cell = torch.eye(3, dtype=dtype) * 10
        coord = torch.rand([natoms, 3], dtype=dtype)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        atomic_numbers = [1, 1, 1, 8, 8]
        idx_perm = [1, 0, 4, 3, 2]

        prec = 1e-10
        low_prec = 1e-4

        ase_atoms0 = Atoms(
            numbers=atomic_numbers,
            positions=coord,
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
            calculator=self.calculator,
        )
        e0, f0 = ase_atoms0.get_potential_energy(), ase_atoms0.get_forces()
        s0, v0 = ase_atoms0.get_stress(voigt=True), -ase_atoms0.get_stress(voigt=False) * ase_atoms0.get_volume()

        ase_atoms1 = Atoms(
            numbers=[atomic_numbers[i] for i in idx_perm],
            positions=coord[idx_perm, :],
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
            calculator=self.calculator,
        )
        e1, f1 = ase_atoms1.get_potential_energy(), ase_atoms1.get_forces()
        s1, v1 = ase_atoms1.get_stress(voigt=True), -ase_atoms1.get_stress(voigt=False) * ase_atoms1.get_volume()

        assert type(e0) == float
        assert f0.shape == (natoms, 3)
        assert v0.shape == (3, 3)
        torch.testing.assert_close(e0, e1, rtol=low_prec, atol=prec)
        torch.testing.assert_close(f0[idx_perm, :], f1, rtol=low_prec, atol=prec)
        torch.testing.assert_close(s0, s1, rtol=low_prec, atol=prec)
        torch.testing.assert_close(v0, v1, rtol=low_prec, atol=prec)