import unittest
import torch
import json
import os
import copy

from deepmd.common import expand_sys_str
from deepmd_pt.utils.dataset import DeepmdDataSet
import deepmd_pt.model.model as models
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
    return dataset

class ModelSave():
    def __init__(self, model_name, input_json):
        with open(input_json, 'r') as fin:
            self.config = json.load(fin)
        self.model_name = model_name
        self.config['loss']['starter_learning_rate'] = self.config['learning_rate']['start_lr']
        self.dataset = get_dataset(self.config)
        self.loss = EnergyStdLoss(**self.config['loss'])
        self.wrapper_save = self.create_wrapper()
        self.wrapper_load = self.create_wrapper()
        
    def create_wrapper(self):
        model_config = copy.deepcopy(self.config['model'])
        model = models.__dict__[self.model_name](model_config, self.dataset).to(env.DEVICE)
        return ModelWrapper(model, self.loss)
    
    def is_consistency(self):
        tmp_model_file = self.model_name + '_tmp.pt'
        batch = self.dataset.__getitem__()
        labels = {
            'energy':batch['energy'],
            'force':batch['force'],
            'virial':batch['virial'],
        }
        keys = ["coord", "atype", "natoms", "mapping", "shift", "selected", "box", "selected_type"]
        batch = {key:batch[key] for key in batch if key in keys}
        batch['coord'].requires_grad = True
        cur_lr = 1
        task_key="Default"
        result1 = self.wrapper_save(**batch, cur_lr=cur_lr, task_key=task_key, label=labels)[0]
        torch.save(self.wrapper_save.state_dict(), tmp_model_file)
        state_dict = torch.load(tmp_model_file)
        self.wrapper_load.load_state_dict(state_dict)
        result2 = self.wrapper_load(**batch, cur_lr=cur_lr, task_key=task_key, label=labels)[0]
        final_result = all([torch.allclose(result1[item], result2[item]) for item in result1])
        return final_result
class TestEnergyModelSeA(unittest.TestCase):

    def setUp(self):
        self.infos = {
            'EnergyModelSeA':env.TEST_CONFIG,
            'EnergyModelDPA1':'tests/water/se_atten.json',
        }
        self.wrappers = []
        for model_name, input_json in self.infos.items():
            self.wrappers.append(ModelSave(model_name, input_json))
            
    def test_saveload(self):
        for wrapper in self.wrappers:
            self.assertTrue(wrapper.is_consistency())
            
if __name__ == '__main__':
    unittest.main()