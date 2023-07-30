import os
import sys
import torch
import unittest

from deepmd_pt.calculator import DP, inference_multiconf, inference_singleconf


if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtype = torch.float64


class TestCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = DP(model="druglike/model_0620.pt")

    def test_calculator(self):
        from ase import Atoms

        natoms = 5
        cell = torch.eye(3, dtype=dtype) * 10
        coord = torch.rand([natoms, 3], dtype=dtype)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        atomic_numbers = [1, 1, 1, 6, 6]
        idx_perm = [1, 0, 4, 3, 2]

        ret0 = inference_multiconf(self.calculator.dp, coord[None, :, :], cell[None, :, :], atype, type_split=False)
        ret1 = inference_multiconf(self.calculator.dp, coord[None, idx_perm, :], cell[None, :, :], atype[idx_perm], type_split=False)
        prec = 1e-10
        low_prec = 1e-4
        assert ret0['energy'].shape == (1, 1)
        assert ret0['force'].shape == (1, natoms, 3)
        assert ret0['virial'].shape == (1, 3, 3)
        torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
        torch.testing.assert_close(ret0['force'][:, idx_perm, :], ret1['force'], rtol=prec, atol=prec)
        torch.testing.assert_close(ret0['virial'], ret1['virial'], rtol=prec, atol=prec)

        ase_atoms = Atoms(
            numbers=atomic_numbers,
            positions=coord,
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
            calculator=self.calculator,
        )
        e = ase_atoms.get_potential_energy()
        torch.testing.assert_close(ret0['energy'][0][0], e, rtol=low_prec, atol=0.2)


if __name__ == '__main__':
    unittest.main()
