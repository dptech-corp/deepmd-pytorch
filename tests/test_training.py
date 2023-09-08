import json
import os
import shutil
import unittest
from copy import deepcopy

import numpy as np
from deepmd_pt.entrypoints.main import get_trainer
from .test_permutation import (
  model_se_e2_a,
  model_dpa1,
  model_dpau,
  model_hybrid,
)


class TestDPTrain:
    def test_dp_train(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        self.tearDown()

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


# class TestEnergyModelSeA(unittest.TestCase, TestDPTrain):
#     def setUp(self):
#         input_json = "tests/water/se_atten.json"
#         with open(input_json, "r") as f:
#             self.config = json.load(f)
#         self.config["model"] = deepcopy(model_se_e2_a)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1


# class TestEnergyModelDPA1(unittest.TestCase, TestDPTrain):
#     def setUp(self):
#         input_json = "tests/water/se_atten.json"
#         with open(input_json, "r") as f:
#             self.config = json.load(f)
#         self.config["model"] = deepcopy(model_dpa1)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1
#
#
# class TestEnergyModelDPAU(unittest.TestCase, TestDPTrain):
#     def setUp(self):
#         input_json = "tests/water/se_atten.json"
#         with open(input_json, "r") as f:
#             self.config = json.load(f)
#         self.config["model"] = deepcopy(model_dpau)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1
#
#
# class TestEnergyModelHybrid(unittest.TestCase, TestDPTrain):
#     def setUp(self):
#         input_json = "tests/water/se_atten.json"
#         with open(input_json, "r") as f:
#             self.config = json.load(f)
#         self.config["model"] = deepcopy(model_hybrid)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1


if __name__ == '__main__':
    unittest.main()
