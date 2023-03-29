import numpy as np
import os
import torch
import unittest

from deepmd.descriptor.se_a import DescrptSeA
from deepmd.fit.ener import EnerFitting
from deepmd.model.model_stat import make_stat_input as dp_make, merge_sys_stat as dp_merge
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils import random as dp_random
from deepmd.common import expand_sys_str

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.embedding_net import EmbeddingNet
from deepmd_pt.stat import make_stat_input as my_make, compute_output_stats
from deepmd_pt import env


CUR_DIR = os.path.dirname(__file__)


def compare(ut, base, given):
    if isinstance(base, list):
        ut.assertEqual(len(base), len(given))
        for idx in range(len(base)):
            compare(ut, base[idx], given[idx])
    elif isinstance(base, np.ndarray):
        ut.assertTrue(np.allclose(base.reshape(-1), given.cpu().reshape(-1)))
    else:
        ut.assertEqual(base, given)

class TestDataset(unittest.TestCase):

    def setUp(self):
        if env.TEST_DATASET == 'water':
            self.systems = [
                os.path.join(CUR_DIR, 'water/data/data_0'),
                os.path.join(CUR_DIR, 'water/data/data_1'),
                os.path.join(CUR_DIR, 'water/data/data_2')
            ]
            self.sel = [46, 92]
        elif env.TEST_DATASET == 'Cu':
            self.systems = "/data/cu_test.hdf5"
            self.sel = [128]
        self.batch_size = 3
        self.rcut = 6.
        self.data_stat_nbatch = 2
        self.rcut_smth = 0.5
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16
        self.n_neuron = [240, 240, 240]

        dp_random.seed(10)
        if env.TEST_DATASET == 'Cu':
            self.systems = expand_sys_str(self.systems)
        dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)
        dp_dataset.add('energy', 1, atomic=False, must=False, high_prec=True)
        dp_dataset.add('force',  3, atomic=True,  must=False, high_prec=False)
        self.dp_sampled = dp_make(dp_dataset, self.data_stat_nbatch, False)
        self.dp_merged = dp_merge(self.dp_sampled)
        self.dp_mesh = self.dp_merged.pop('default_mesh')
        self.dp_d = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron
        )
        def my_merge(energy, natoms):
            energy_lst = []
            natoms_lst = []
            for i in range(len(energy)):
                for j in range(len(energy[i])):
                    energy_lst.append(torch.tensor(energy[i][j]))
                    natoms_lst.append(torch.tensor(natoms[i][j]).unsqueeze(0).expand(energy[i][j].shape[0], -1))
            return energy_lst, natoms_lst
        energy = self.dp_sampled['energy']
        natoms = self.dp_sampled['natoms_vec']
        self.energy, self.natoms = my_merge(energy, natoms)

    def test_stat_output(self):
        dp_fn = EnerFitting(self.dp_d, self.n_neuron)
        dp_fn.compute_output_stats(self.dp_sampled)
        bias_atom_e = compute_output_stats(self.energy, self.natoms)
        self.assertTrue(np.allclose(dp_fn.bias_atom_e, bias_atom_e[:,0]))

    def test_stat_input(self):
        my_random.seed(10)
        if env.TEST_DATASET == 'Cu':
            my_dataset = DeepmdDataSet(self.systems, self.batch_size, ['Cu'], 1.0, [1, 1])
        else:
            my_dataset = DeepmdDataSet(self.systems, self.batch_size, ['O', 'H'], 1.0, [1, 1])
        my_sampled = my_make(my_dataset, self.data_stat_nbatch)
        dp_keys = set(self.dp_merged.keys())
        self.dp_merged['natoms'] = self.dp_merged['natoms_vec']
        for key in dp_keys:
            if not key in my_sampled[0]:
                continue
            lst = []
            for item in my_sampled:
                for j in range(self.data_stat_nbatch):
                    lst.append(item[key][j*self.batch_size:(j+1)*self.batch_size])
            compare(self, self.dp_merged[key], lst)

    def test_descriptor(self):
        coord = self.dp_merged['coord']
        atype = self.dp_merged['type']
        natoms = self.dp_merged['natoms_vec']
        box = self.dp_merged['box']
        self.dp_d.compute_input_stats(coord, box, atype, natoms, self.dp_mesh, {})

        my_random.seed(10)
        if env.TEST_DATASET == 'Cu':
            my_dataset = DeepmdDataSet(self.systems, self.batch_size, ['Cu'], self.rcut, self.sel)
        else:
            my_dataset = DeepmdDataSet(self.systems, self.batch_size, ['O', 'H'], self.rcut, self.sel)
        my_en = EmbeddingNet(self.rcut, self.rcut_smth, self.sel, self.filter_neuron, self.axis_neuron)
        sampled = my_make(my_dataset, self.data_stat_nbatch)
        for sys in sampled:
            for key in ['coord', 'force', 'energy', 'atype', 'natoms', 'extended_coord', 'selected', 'shift', 'mapping']:
                if key in sys.keys():
                    sys[key] = sys[key].to(env.DEVICE)
        my_en.compute_input_stats(sampled)
        my_en.mean = my_en.mean
        my_en.stddev = my_en.stddev
        self.assertTrue(np.allclose(self.dp_d.davg.reshape([-1]), my_en.mean.cpu().reshape([-1])))
        self.assertTrue(np.allclose(self.dp_d.dstd.reshape([-1]), my_en.stddev.cpu().reshape([-1])))

if __name__ == '__main__':
    unittest.main()
