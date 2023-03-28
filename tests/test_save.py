from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel
from deepmd_pt import env
from deepmd_pt import my_random
import unittest
import torch

class TestEnergy(unittest.TestCase):

    def setUp(self):
        self.batch_size = 3
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16
        self.n_neuron = [32, 32, 32]
        self.data_stat_nbatch = 3
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600
        self.start_pref_e = 1.
        self.limit_pref_e = 2.
        self.start_pref_f = 2.
        self.limit_pref_f = 1.
        if env.TEST_DATASET == 'water':
            self.type_map = ['O', 'H']
            self.sel = [46, 92],
            kDataSystems = [
                os.path.join(CUR_DIR, 'water/data/data_0'),
                os.path.join(CUR_DIR, 'water/data/data_1'),
                os.path.join(CUR_DIR, 'water/data/data_2')
            ]
        elif env.TEST_DATASET == 'Cu':
            self.type_map = ['Cu']
            self.sel = [138]
            self.n_neuron = [240, 240, 240]
            kDataSystems = ["/data/cu_test.hdf5#/Cu16"]
        self.ntypes = len(self.type_map)
        self.dataset = DeepmdDataSet(
            systems=kDataSystems,
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
        keys = ["coord", "atype", "natoms", "mapping", "shift", "selected"]
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