from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.loss.ener import EnergyStdLoss
from deepmd_pt.model.ener import EnergyModel
from deepmd_pt.utils import env
from deepmd_pt.utils import dp_random
import unittest
import torch
import os
import json
from deepmd.common import expand_sys_str

class TestEnergy(unittest.TestCase):

    def setUp(self):
        with open(env.TEST_CONFIG, 'r') as fin:
            content = fin.read()
        config = json.loads(content)
        model_config = config['model']
        self.rcut = model_config['descriptor']['rcut']
        self.rcut_smth = model_config['descriptor']['rcut_smth']
        self.sel = model_config['descriptor']['sel']
        self.systems = config['training']['validation_data']['systems']
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.batch_size = config['training']['training_data']['batch_size']
        self.type_map = model_config['type_map']
        self.filter_neuron = model_config['descriptor']['neuron']
        self.axis_neuron = model_config['descriptor']['axis_neuron']
        self.n_neuron = model_config['fitting_net']['neuron']

        self.data_stat_nbatch = 3
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600
        self.start_pref_e = 1.
        self.limit_pref_e = 2.
        self.start_pref_f = 2.
        self.limit_pref_f = 1.

        self.ntypes = len(self.type_map)
        self.dataset = DeepmdDataSet(
            systems=self.systems,
            batch_size=self.batch_size,
            type_map=self.type_map,
            rcut=self.rcut,
            sel=self.sel
        )   
        model_params={
            'descriptor': {
                'type': 'se_e2_a',
                'sel': self.sel,
                'rcut_smth': self.rcut_smth,
                'rcut': self.rcut,
                'neuron': self.filter_neuron,
                'axis_neuron': self.axis_neuron,
            },
            'fitting_net': {
                'neuron': self.n_neuron
            },
            'data_stat_nbatch': self.data_stat_nbatch
        }
        model_params2 = model_params.copy()
        self.model = EnergyModel(model_params, self.dataset).to(env.DEVICE)
        self.model2 = EnergyModel(model_params2, self.dataset).to(env.DEVICE)

    def test_saveload(self):
        batch = self.dataset.__getitem__()
        keys = ["coord", "atype", "natoms", "mapping", "shift", "selected", "box"]
        batch = {key:batch[key] for key in batch if key in keys}
        batch['coord'].requires_grad = True
        result1 = self.model(**batch)
        torch.save(self.model.state_dict(), 'tmp.ckpt')
        state_dict = torch.load('tmp.ckpt')
        self.model2.load_state_dict(state_dict)
        result2 = self.model2(**batch)
        for i in range(2):
            assert torch.allclose(result1[i], result2[i])

if __name__ == '__main__':
    unittest.main()