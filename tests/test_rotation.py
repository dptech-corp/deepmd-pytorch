from typing import List
import unittest
import torch
import json
from scipy.stats import special_ortho_group
import numpy as np

from deepmd_pt.utils import env
from deepmd_pt.utils.dataset import DeepmdDataSystem
from deepmd_pt.model.model import EnergyModelSeA
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input

class CheckSymmetry(DeepmdDataSystem):
    def __init__(self, sys_path: str, rcut, sec, type_map: List[str] = None, type_split=True):
        super().__init__(sys_path, rcut, sec, type_map, type_split)
    def get_rotation(self, index, rotation_matrix):
        for i in range(0,len(self._dirs) + 1):#note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self._dirs[i-1])
        frames['coord'] = np.dot(rotation_matrix, frames['coord'].reshape(-1, 3).T).T.reshape(self.nframes, -1)
        frames['box'] = np.dot(rotation_matrix, frames['box'].reshape(-1, 3).T).T.reshape(self.nframes, -1)
        frames['force'] = np.dot(rotation_matrix, frames['force'].reshape(-1, 3).T).T.reshape(self.nframes, -1)
        frames['virial'] = np.dot(rotation_matrix, frames['virial'].reshape(-1, 3).T).T.reshape(self.nframes, -1)
        frame = self.single_preprocess(frames,index-self.prefix_sum[i-1])
        return frame

def get_data(batch):
    inputs = {}
    for key in ['coord', 'atype', 'mapping', 'shift', 'selected', 'selected_type', 'box']:
        inputs[key] = batch[key].unsqueeze(0).to(env.DEVICE)
    inputs['natoms'] = None
    return inputs

class TestRotation(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        with open(env.TEST_CONFIG, 'r') as fin:
            self.config = json.load(fin)
        self.rotation = special_ortho_group.rvs(3)
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
        self.rotated_batch = dpdatasystem.get_rotation(batch_index, self.rotation)

    def test_rotation(self):
        result1 = self.model(**get_data(self.origin_batch))
        result2 = self.model(**get_data(self.rotated_batch))
        rotation = torch.from_numpy(self.rotation).to(env.DEVICE)
        self.assertTrue(result1['energy']==result2['energy'])
        if 'force' in result1:
            self.assertTrue(torch.allclose(result2['force'][0], torch.matmul(rotation, result1['force'][0].T).T))
        if 'virial' in result1:
            self.assertTrue(torch.allclose(result2['virial'][0], torch.matmul(torch.matmul(rotation, result1['virial'][0].T), rotation.T)))

if __name__ == '__main__':
    unittest.main()
    