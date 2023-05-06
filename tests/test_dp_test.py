import json
import os
import unittest
from copy import deepcopy

import numpy as np
from deepmd_pt.entrypoints.main import get_trainer
from deepmd_pt.infer import inference


class TestDPTest(unittest.TestCase):
    def setUp(self):
        input_json = "tests/water/se_atten.json"
        with open(input_json, "r") as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["validation_data"]["systems"] = ["tests/water/data/single"]

    def test_dp_test(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        input_dict, label_dict = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        tester = inference.Tester(deepcopy(self.config), "model.pt")
        res = tester.run()
        for k, v in res.items():
            if k == "rmse" or "mae" in k:
                continue
            self.assertTrue(np.allclose(v, more_loss[k].cpu().detach().numpy()),
                            "Test result of %s is incorrect" % k)

    def tearDown(self):
        os.remove("model.pt")
        os.remove("lcurve.out")


if __name__ == '__main__':
    unittest.main()