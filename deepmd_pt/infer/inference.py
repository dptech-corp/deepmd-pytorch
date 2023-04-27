import logging
import math
from copy import deepcopy
from typing import Any, Dict

import torch
from deepmd_pt.loss.ener import EnergyStdLoss
from deepmd_pt.model.model import EnergyModelDPA1, EnergyModelSeA
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils import dp_random
from deepmd_pt.utils.dataloader import BufferedIterator, DpLoaderSet
from deepmd_pt.utils.env import DEVICE, JIT
from deepmd_pt.utils.stat import make_stat_input
from torch.utils.data import DataLoader

if torch.__version__.startswith("2"):
    import torch._dynamo


class Trainer(object):

    def __init__(self, config: Dict[str, Any], ckpt, numb_test=100):
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        self.numb_test = numb_test
        model_params = config['model']
        training_params = config['training']

        # Data + Model
        dp_random.seed(training_params['seed'])
        self.dataset_params = training_params.pop('validation_data')
        self.type_split = True
        if model_params['descriptor']['type'] in ['se_atten']:
            self.type_split = False
        self.model_params = deepcopy(model_params)

        if model_params["descriptor"]["type"] == "se_e2_a":
            self.model = EnergyModelSeA(model_params).to(DEVICE)
        elif model_params["descriptor"]["type"] == "se_atten":
            self.model = EnergyModelDPA1(model_params).to(DEVICE)
        else:
            raise NotImplementedError

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model) # inference only
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)

        state_dict = torch.load(ckpt)
        self.wrapper.load_state_dict(state_dict)

        # Loss
        loss_params = config.pop("loss")
        assert loss_params.pop("type", "ener"), "Only loss `ener` is supported!"
        loss_params["starter_learning_rate"] = 1.0 # TODO: lr here is useless
        self.loss = EnergyStdLoss(**loss_params)

    def get_data(self, data):
        batch_data = next(iter(data))
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(DEVICE)
        input_dict = {}
        for item in [
            "coord",
            "atype",
            "natoms",
            "mapping",
            "shift",
            "selected",
            "selected_type",
            "box",
        ]:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ["energy", "force", "virial"]:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        return input_dict, label_dict

    def run(self):
        systems = self.dataset_params["systems"]
        system_results = {}
        global_sum_natoms = 0
        for system in systems:
            logging.info("# ---------------output of dp test--------------- ")
            logging.info(f"# testing system : {system}")
            dataset = DpLoaderSet([system], self.dataset_params['batch_size'], self.model_params, type_split=self.type_split)
            dataloader = DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset),
                batch_size=None,
                num_workers=8,  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
            )
            data = BufferedIterator(iter(dataloader))

            single_results = {}
            sum_natoms = 0
            for _ in range(self.numb_test):
                try:
                    input_dict, label_dict = self.get_data(data)
                except StopIteration:
                    break
                model_pred, _, _ = self.wrapper(**input_dict)
                _, more_loss = self.loss(model_pred, label_dict, input_dict["natoms"], 1.0) # TODO: lr here is useless
                natoms = input_dict["natoms"][0, 0]
                sum_natoms += natoms
                for k, v in more_loss.items():
                    if "rmse" in k:
                        single_results[k] = single_results.get(k, 0.0) + v**2 * natoms
            results = {
                k: math.sqrt(v / sum_natoms) for k, v in single_results.items()
            }
            for item in sorted(list(results.keys())):
                logging.info(f"{item}: {results[item]:.4f}")
            logging.info("# ----------------------------------------------- ")
            for k, v in single_results.items():
                system_results[k] = system_results.get(k, 0.0) + v
            global_sum_natoms += sum_natoms

        global_results = {
            k: math.sqrt(v / global_sum_natoms) for k, v in system_results.items()
        }
        logging.info("# ----------weighted average of errors----------- ")
        logging.info(f"# number of systems : {len(systems)}")
        for item in sorted(list(global_results.keys())):
            logging.info(f"{item}: {global_results[item]:.4f}")
        logging.info("# ----------------------------------------------- ")
