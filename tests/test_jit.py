import json
import os
import shutil
import unittest
from copy import deepcopy
import torch

import numpy as np
from deepmd_pt.entrypoints.main import get_trainer
from deepmd_pt.infer import inference
from .test_permutation import (
  model_se_e2_a,
  model_dpa1,
  model_dpau,
  model_hybrid,
)


class TestJIT:
    def test_jit(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        model = torch.jit.script(inference.Tester(deepcopy(self.config), './model.pt', 1).model)
        torch.jit.save(model, './frozen_model.pth', {})
        self.tearDown()

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestEnergyModelSeA(unittest.TestCase, TestJIT):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10


class TestEnergyModelDPA1(unittest.TestCase, TestJIT):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["model"] = deepcopy(model_dpa1)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10


class TestEnergyModelDPAU(unittest.TestCase, TestJIT):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["model"] = deepcopy(model_dpau)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10


class TestEnergyModelHybrid(unittest.TestCase, TestJIT):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["model"] = deepcopy(model_hybrid)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

class TestEnergyModelHybrid2(unittest.TestCase, TestJIT):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["model"] = deepcopy(model_hybrid)
        self.config["model"]["descriptor"]["hybrid_mode"] = "sequential"
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10


if __name__ == '__main__':
    unittest.main()
