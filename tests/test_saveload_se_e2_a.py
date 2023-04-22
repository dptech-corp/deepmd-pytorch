import unittest
import torch
import json
import copy
import os

from deepmd.common import expand_sys_str
from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.model.model import EnergyModelSeA
from deepmd_pt.utils import env
from deepmd_pt.utils.stat import make_stat_input
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.loss.ener import EnergyStdLoss

def get_dataset(config):
    model_config = config['model']
    rcut = model_config['descriptor']['rcut']
    sel = model_config['descriptor']['sel']
    systems = config['training']['validation_data']['systems']
    if isinstance(systems, str):
        systems = expand_sys_str(systems)
    batch_size = config['training']['training_data']['batch_size']
    type_map = model_config['type_map']

    dataset = DeepmdDataSet(
        systems=systems,
        batch_size=batch_size,
        type_map=type_map,
        rcut=rcut,
        sel=sel
    )
    data_stat_nbatch = model_config.get('data_stat_nbatch', 10)
    sampled = make_stat_input(dataset, data_stat_nbatch)
    return dataset, sampled

class TestSaveLoadDPA1(unittest.TestCase):
    def setUp(self):
        input_json = env.TEST_CONFIG
        with open(input_json, 'r') as fin:
            self.config = json.load(fin)
        self.config['loss']['starter_learning_rate'] = self.config['learning_rate']['start_lr']
        self.dataset, self.sampled = get_dataset(self.config)
        self.loss = EnergyStdLoss(**self.config['loss'])
        self.cur_lr = 1
        self.task_key = "Default"
        self.input_dict, self.label_dict = self.get_data()
        self.start_lr = self.config['learning_rate']['start_lr']
        
    def get_model_result(self, read=False, model_file='tmp_model.pt'):
        wrapper = self.create_wrapper()
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=self.start_lr)
        optimizer.zero_grad()
        if read:
            wrapper.load_state_dict(torch.load(model_file))
            os.remove(model_file)
        else:
            torch.save(wrapper.state_dict(), model_file)
        result = wrapper(**self.input_dict,cur_lr=self.cur_lr, label=self.label_dict, task_key=self.task_key)[0]
        return result
    def create_wrapper(self):
        model_config = copy.deepcopy(self.config['model'])
        sampled = copy.deepcopy(self.sampled)
        model = EnergyModelSeA(model_config, sampled).to(env.DEVICE)
        return ModelWrapper(model, self.loss)
        
    def get_data(self):
        batch_data = self.dataset.get_training_batch()
        input_dict = {}
        for item in ['coord', 'atype', 'natoms', 'mapping', 'shift', 'selected', 'selected_type', 'box']:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ['energy', 'force', 'virial']:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        return input_dict, label_dict
    
    def test_saveload(self):
        result1 = self.get_model_result()
        result2 = self.get_model_result(read=True)
        final_result = all([torch.allclose(result1[item], result2[item]) for item in result1])
        self.assertTrue(final_result)
        
if __name__ == '__main__':
    unittest.main()