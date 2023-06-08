import logging
import math
from copy import deepcopy
from typing import Any, Dict

import torch
from deepmd_pt.loss import EnergyStdLoss, DenoiseLoss
from deepmd_pt.model.model import get_model
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils import dp_random
from deepmd_pt.utils.dataloader import BufferedIterator, DpLoaderSet
from deepmd_pt.utils.env import DEVICE, JIT
from deepmd_pt.utils.stat import make_stat_input
from torch.utils.data import DataLoader

if torch.__version__.startswith("2"):
    import torch._dynamo


class Tester(object):

    def __init__(self, config: Dict[str, Any], ckpt, numb_test=100):
        """Construct a DeePMD tester.

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
        if model_params['descriptor']['type'] in ['se_atten', 'se_uni']:
            self.type_split = False
        self.model_params = deepcopy(model_params)

        model_params["resuming"] = (ckpt is not None)  # should always be True for inferencing
        self.model = get_model(model_params).to(DEVICE)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model)  # inference only
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)

        state_dict = torch.load(ckpt)
        self.wrapper.load_state_dict(state_dict)

        # Loss
        self.noise_settings = None
        loss_params = config.pop("loss")
        loss_type = loss_params.pop("type", "ener")
        if loss_type == 'ener':
            loss_params["starter_learning_rate"] = 1.0  # TODO: lr here is useless
            self.loss = EnergyStdLoss(**loss_params)
        elif loss_type == 'denoise':
            if loss_type == 'denoise':
                self.noise_settings = {"noise_type": loss_params.pop("noise_type", "uniform"),
                                       "noise": loss_params.pop("noise", 1.0),
                                       "noise_mode": loss_params.pop("noise_mode", "fix_num"),
                                       "mask_num": loss_params.pop("mask_num", 8),
                                       "same_mask": loss_params.pop("same_mask", False),
                                       "mask_coord": loss_params.pop("mask_coord", False),
                                       "mask_type": loss_params.pop("mask_type", False),
                                       "mask_type_idx": len(model_params["type_map"]) - 1}
            loss_params['ntypes'] = len(model_params['type_map'])
            self.loss = DenoiseLoss(**loss_params)
        else:
            raise NotImplementedError

    @staticmethod
    def get_data(data):
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
            "selected_loc",
            "selected_type",
            "box",
        ]:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ["energy", "force", "virial", "clean_coord", "clean_type", "coord_mask", "type_mask"]:
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
            dataset = DpLoaderSet([system], self.dataset_params['batch_size'], self.model_params,
                                  type_split=self.type_split, noise_settings=self.noise_settings)
            dataloader = DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset),
                batch_size=None,
                num_workers=8,  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
            )
            data = iter(dataloader)

            single_results = {}
            sum_natoms = 0
            for _ in range(self.numb_test):
                try:
                    input_dict, label_dict = self.get_data(data)
                except StopIteration:
                    break
                model_pred, _, _ = self.wrapper(**input_dict)
                _, more_loss = self.loss(model_pred, label_dict, input_dict["natoms"], 1.0, mae=True)  # TODO: lr here is useless
                natoms = input_dict["natoms"][0, 0]
                sum_natoms += natoms
                for k, v in more_loss.items():
                    if "mae" in k:
                        single_results[k] = single_results.get(k, 0.0) + v * natoms
                    else:
                        single_results[k] = single_results.get(k, 0.0) + v ** 2 * natoms
            results = {
                k: v / sum_natoms if "mae" in k else math.sqrt(v / sum_natoms) for k, v in single_results.items()
            }
            for item in sorted(list(results.keys())):
                logging.info(f"{item}: {results[item]:.4f}")
            logging.info("# ----------------------------------------------- ")
            for k, v in single_results.items():
                system_results[k] = system_results.get(k, 0.0) + v
            global_sum_natoms += sum_natoms

        global_results = {
            k: v / global_sum_natoms if "mae" in k else math.sqrt(v / global_sum_natoms) for k, v in system_results.items()
        }
        logging.info("# ----------weighted average of errors----------- ")
        logging.info(f"# number of systems : {len(systems)}")
        for item in sorted(list(global_results.keys())):
            logging.info(f"{item}: {global_results[item]:.4f}")
        logging.info("# ----------------------------------------------- ")
        return global_results
