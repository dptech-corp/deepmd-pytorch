import numpy as np
import os
import unittest
import json

from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils import random as tf_random
from deepmd.common import expand_sys_str

from deepmd_pt.utils.dataloader import DpLoaderSet, get_weighted_sampler
from deepmd_pt.utils import env

CUR_DIR = os.path.dirname(__file__)



class TestSampler(unittest.TestCase):

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
        self.my_dataset = DpLoaderSet(self.systems, self.batch_size,
                                      model_params={
                                          'descriptor': {
                                              'sel': self.sel,
                                              'rcut': self.rcut,
                                          },
                                          'type_map': model_config['type_map']
                                      }, seed=10)

        tf_random.seed(10)
        self.dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)

    def test_auto_prob_uniform(self):
        auto_prob_style= 'prob_uniform'
        sampler = get_weighted_sampler(self.my_dataset,prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs,dp_probs))


    def test_auto_prob_sys_size(self):
        auto_prob_style= 'prob_sys_size'
        sampler = get_weighted_sampler(self.my_dataset,prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs,dp_probs))

    def test_auto_prob_sys_size_ext(self):
        auto_prob_style= 'prob_sys_size;0:1:0.2;1:3:0.8'
        sampler = get_weighted_sampler(self.my_dataset,prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs,dp_probs))

    def test_sys_probs(self):
        sys_probs= [0.1,0.4,0.5]
        sampler = get_weighted_sampler(self.my_dataset,prob_style=sys_probs,sys_prob=True)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(sys_probs=sys_probs)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs,dp_probs))

if __name__ == '__main__':
    unittest.main()