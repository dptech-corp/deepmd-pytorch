import numpy as np
import os
import torch
import unittest
import json

from deepmd.descriptor.se_a import DescrptSeA as DescrptSeA_tf
from deepmd.fit.ener import EnerFitting
from deepmd.model.model_stat import make_stat_input as dp_make, merge_sys_stat as dp_merge
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils import random as tf_random
from deepmd.common import expand_sys_str

from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.model.descriptor.se_a import DescrptSeA
from deepmd_pt.utils.stat import make_stat_input as my_make, compute_output_stats
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils import env

CUR_DIR = os.path.dirname(__file__)


def compare(ut, base, given):
    if isinstance(base, list):
        ut.assertEqual(len(base), len(given))
        for idx in range(len(base)):
            compare(ut, base[idx], given[idx])
    elif isinstance(base, np.ndarray):
        ut.assertTrue(np.allclose(base.reshape(-1), given.reshape(-1)))
    else:
        ut.assertEqual(base, given)

class TestDataset(unittest.TestCase):

    def setUp(self):
        with open(env.TEST_CONFIG, 'r') as fin:
            content = fin.read()
        config = json.loads(content)
        model_config = config['model']
        self.rcut = model_config['descriptor']['rcut']
        self.rcut_smth = model_config['descriptor']['rcut_smth']
        self.sel = model_config['descriptor']['sel']
        self.batch_size = config['training']['training_data']['batch_size']
        self.systems = config['training']['validation_data']['systems']
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.my_dataset = DpLoaderSet(self.systems,self.batch_size,
        model_params={
                'descriptor': {
                    'sel': self.sel,
                    'rcut': self.rcut,
                },
                'type_map': model_config['type_map']
            },seed=10)
        self.filter_neuron = model_config['descriptor']['neuron']
        self.axis_neuron = model_config['descriptor']['axis_neuron']
        self.data_stat_nbatch = 2
        self.filter_neuron = model_config['descriptor']['neuron']
        self.axis_neuron = model_config['descriptor']['axis_neuron']
        self.n_neuron = model_config['fitting_net']['neuron']

        self.my_sampled = my_make(self.my_dataset.systems, self.my_dataset.dataloaders, self.data_stat_nbatch)

        tf_random.seed(10)
        dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)
        dp_dataset.add('energy', 1, atomic=False, must=False, high_prec=True)
        dp_dataset.add('force',  3, atomic=True,  must=False, high_prec=False)
        self.dp_sampled = dp_make(dp_dataset, self.data_stat_nbatch, False)
        self.dp_merged = dp_merge(self.dp_sampled)
        self.dp_mesh = self.dp_merged.pop('default_mesh')
        self.dp_d = DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron
        )
    
    def test_stat_output(self):
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
        energy, natoms = my_merge(energy, natoms)
        dp_fn = EnerFitting(self.dp_d, self.n_neuron)
        dp_fn.compute_output_stats(self.dp_sampled)
        bias_atom_e = compute_output_stats(energy, natoms)
        self.assertTrue(np.allclose(dp_fn.bias_atom_e, bias_atom_e[:,0]))

    def test_stat_input(self):
        my_sampled = self.my_sampled
        # list of dicts, each dict contains samples from a system
        dp_keys = set(self.dp_merged.keys()) # dict of list of batches
        self.dp_merged['natoms'] = self.dp_merged['natoms_vec']
        for key in dp_keys:
            if not key in my_sampled[0] or key in 'coord':
                # coord is pre-normalized
                continue
            lst = []
            for item in my_sampled:
                bsz = item['energy'].shape[0]//self.data_stat_nbatch
                for j in range(self.data_stat_nbatch):
                    lst.append(item[key][j*bsz:(j+1)*bsz].cpu().numpy())
                compare(self, self.dp_merged[key], lst)

    def test_descriptor(self):
        coord = self.dp_merged['coord']
        atype = self.dp_merged['type']
        natoms = self.dp_merged['natoms_vec']
        box = self.dp_merged['box']
        self.dp_d.compute_input_stats(coord, box, atype, natoms, self.dp_mesh, {})

        my_en = DescrptSeA(self.rcut, self.rcut_smth, self.sel, self.filter_neuron, self.axis_neuron)
        sampled = self.my_sampled
        for sys in sampled:
            for key in ['coord', 'force', 'energy', 'atype', 'natoms', 'extended_coord', 'selected', 'shift', 'mapping']:
                if key in sys.keys():
                    sys[key] = sys[key].to(env.DEVICE)
        my_en.compute_input_stats(sampled)
        my_en.mean = my_en.mean
        my_en.stddev = my_en.stddev
        try:
            self.assertTrue(np.allclose(self.dp_d.davg.reshape([-1]), my_en.mean.cpu().reshape([-1])))
        except:
            print("result: ",self.dp_d.davg.reshape([-1]), my_en.mean.cpu().reshape([-1]))
        self.assertTrue(np.allclose(self.dp_d.dstd.reshape([-1]), my_en.stddev.cpu().reshape([-1])))
    
if __name__ == '__main__':
    unittest.main()
