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

descriptors = {
    'EnergyModelSeA': 'se_e2_a',
    'EnergyModelDPA1': 'se_atten',
}

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

class TestEnergy(unittest.TestCase):

    def setUp(self):
        with open(env.TEST_CONFIG, 'r') as fin:
            self.config = json.load(fin)
        self.dataset = get_dataset(self.config)
        self.models_save, self.models_load = [], []
        for test_model, descriptor in descriptors.items():
            self.models_save.append(self.create_model(test_model, descriptor))
            self.models_load.append(self.create_model(test_model, descriptor))
            
    def create_model(self, model, descriptor):
        model_config = copy.deepcopy(self.config['model'])
        model_config['descriptor']['type'] = descriptor
        return models.__dict__[model](model_config, self.dataset).to(env.DEVICE)

    def test_saveload(self):
        model_file_name = 'tmp.ckpt'
        batch = self.dataset.__getitem__()
        keys = ["coord", "atype", "natoms", "mapping", "shift", "selected", "box", "selected_type"]
        batch = {key:batch[key] for key in batch if key in keys}
        batch['coord'].requires_grad = True
        for i in range(len(descriptors)):
            result1 = self.models_save[i](**batch)
            torch.save(self.models_save[i].state_dict(), model_file_name)
            state_dict = torch.load(model_file_name)
            self.models_load[i].load_state_dict(state_dict)
            result2 = self.models_load[i](**batch)
            for item in result1:
                assert torch.allclose(result1[item], result2[item])
            if os.path.exists(model_file_name):
                os.remove(model_file_name)

if __name__ == '__main__':
    unittest.main()