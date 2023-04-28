from typing import List
import unittest
import torch
import json
import numpy as np
import copy

from deepmd_pt.utils import env
from deepmd_pt.utils.dataset import DeepmdDataSystem
from deepmd_pt.model.model import EnergyModelSeA
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input

# exchange i and j
def exchange(frames, name, nframes, i, j):
    tmp = frames[name].reshape(nframes, -1, 3)
    result = copy.deepcopy(tmp)
    result[:,i,:] = tmp[:,j,:]
    result[:,j,:] = tmp[:,i,:]
    return result.reshape(nframes, -1)


class CheckSymmetry(DeepmdDataSystem):
    def __init__(self, sys_path: str, rcut, sec, type_map: List[str] = None, type_split=True):
        super().__init__(sys_path, rcut, sec, type_map, type_split)
    def get_permutation(self, index):
        for i in range(0,len(self._dirs) + 1):#note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self._dirs[i-1])
        ei, ej = self.get_exchange_index()
        frames['coord'] = exchange(frames, 'coord', self.nframes, ei, ej)
        frames['force'] = exchange(frames, 'force', self.nframes, ei, ej)
        frame = self.single_preprocess(frames,index-self.prefix_sum[i-1])
        return frame, ei, ej
    # get random ei and ej
    def get_exchange_index(self, random_seed=20):
        np.random.seed(random_seed)
        atom_index = np.random.randint(len(self._type_map))
        atom_locations = np.where(self._atom_type == atom_index)[0]
        ei, ej = np.random.choice(atom_locations, size=2, replace=False)
        return ei, ej
    
def get_data(batch):
    inputs = {}
    for key in ['coord', 'atype', 'mapping', 'shift', 'selected', 'selected_type', 'box']:
        inputs[key] = batch[key].unsqueeze(0).to(env.DEVICE)
    inputs['natoms'] = None
    return inputs

class TestPermutation(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        with open(env.TEST_CONFIG, 'r') as fin:
            self.config = json.load(fin)
        self.get_dataset(0)
        self.get_model()

    def get_model(self):
        training_systems = self.config['training']['training_data']['systems']
        model_params = self.config['model']
        data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
        train_data = DpLoaderSet(training_systems,self.config['training']['training_data']['batch_size'],model_params)
        sampled = make_stat_input(train_data.systems, train_data.dataloaders, data_stat_nbatch)
        self.model = EnergyModelSeA(self.config['model'], sampled).to(env.DEVICE)

    def get_dataset(self, system_index=0, batch_index=0):
        systems = self.config['training']['training_data']['systems']
        rcut = self.config['model']['descriptor']['rcut']
        sel = self.config['model']['descriptor']['sel']
        sec = torch.cumsum(torch.tensor(sel), dim=0)
        type_map = self.config['model']['type_map']
        dpdatasystem = CheckSymmetry(sys_path=systems[system_index], rcut=rcut, sec=sec, type_map=type_map)
        self.origin_batch = dpdatasystem._get_item(batch_index)
        self.permutate_batch, self.ei, self.ej = dpdatasystem.get_permutation(batch_index)

    def test_permutation(self):
        result1 = self.model(**get_data(self.origin_batch))
        result2 = self.model(**get_data(self.permutate_batch))
        self.assertTrue(result1['energy']==result2['energy'])
        if 'force' in result1:
            permutate_force = copy.deepcopy(result1['force'][0].detach().cpu().numpy())
            permutate_force[self.ei] = result1['force'][0, self.ej].detach().cpu().numpy()
            permutate_force[self.ej] = result1['force'][0, self.ei].detach().cpu().numpy()
            self.assertTrue(torch.allclose(result2['force'][0], torch.tensor(permutate_force).to(env.DEVICE)))
        if 'virial' in result1:
            self.assertTrue(torch.allclose(result2['virial'][0], result1['virial'][0]))

if __name__ == '__main__':
    unittest.main()
    