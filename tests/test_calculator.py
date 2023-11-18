import json
import os
import torch
import unittest
from copy import deepcopy

from deepmd_pt.entrypoints.main import get_trainer
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

        input_dict, label_dict = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        self.calculator = DPCalculator("model.pt")

    def test_calculator(self):
        from ase import Atoms

        natoms = 5
        cell = torch.eye(3, dtype=dtype) * 10
        coord = torch.rand([natoms, 3], dtype=dtype)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        atomic_numbers = [1, 1, 1, 6, 6]
        idx_perm = [1, 0, 4, 3, 2]

        ret0 = self.calculator.calculate(coord[None, :, :], cell, atype)
        ret1 = self.calculator.calculate(coord[None, idx_perm, :], cell, atype[idx_perm])
        prec = 1e-10
        low_prec = 1e-4
        assert ret0['energy'].shape == (1,)
        assert ret0['force'].shape == (natoms, 3)
        assert ret0['virial'].shape == (3, 3)
        torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
        torch.testing.assert_close(ret0['force'][idx_perm, :], ret1['force'], rtol=prec, atol=prec)
        torch.testing.assert_close(ret0['virial'], ret1['virial'], rtol=prec, atol=prec)

        ase_atoms = Atoms(
            numbers=atomic_numbers,
            positions=coord,
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
            calculator=self.calculator,
        )
        e = ase_atoms.get_potential_energy()
        torch.testing.assert_close(ret0['energy'].item(), e, rtol=low_prec, atol=0.2)
