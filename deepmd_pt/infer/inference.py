import logging
import math
from copy import deepcopy
from typing import Any, Dict
from pathlib import Path
import numpy as np

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

    def __init__(self, config: Dict[str, Any], ckpt, numb_test=100, detail_file=None, shuffle_test=False):
        """Construct a DeePMD tester.

        Args:
        - config: The Dict-like configuration with training options.
        """
        self.numb_test = numb_test
        self.detail_file = detail_file
        self.shuffle_test = shuffle_test
        model_params = config['model']
        training_params = config['training']

        # Data + Model
        dp_random.seed(training_params['seed'])
        self.dataset_params = training_params.pop('validation_data')
        self.type_split = False
        if model_params['descriptor']['type'] in ['se_e2_a']:
            self.type_split = True
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
            if not isinstance(batch_data[key], list):
                batch_data[key] = batch_data[key].to(DEVICE)
            else:
                batch_data[key] = [item.to(DEVICE) for item in batch_data[key]]
        input_dict = {}
        for item in [
            "coord",
            "atype",
            "natoms",
            "mapping",
            "shift",
            "nlist",
            "nlist_loc",
            "nlist_type",
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
        for cc, system in enumerate(systems):
            logging.info("# ---------------output of dp test--------------- ")
            logging.info(f"# testing system : {system}")
            system_pred = []
            system_label = []
            dataset = DpLoaderSet([system], self.dataset_params['batch_size'], self.model_params,
                                  type_split=self.type_split, noise_settings=self.noise_settings, shuffle=self.shuffle_test)
            dataloader = DataLoader(
                dataset,
                sampler=None,
                batch_size=None,
                num_workers=1,  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
            )
            data = iter(dataloader)

            single_results = {}
            sum_natoms = 0
            sys_natoms = None
            for _ in range(self.numb_test):
                try:
                    input_dict, label_dict = self.get_data(data)
                except StopIteration:
                    break
                model_pred, _, _ = self.wrapper(**input_dict)
                system_pred.append({item: model_pred[item].detach().cpu().numpy() for item in model_pred})
                system_label.append({item: label_dict[item].detach().cpu().numpy() for item in label_dict})
                _, more_loss = self.loss(model_pred, label_dict, input_dict["natoms"], 1.0, mae=True)  # TODO: lr here is useless
                natoms = int(input_dict["natoms"][0, 0])
                if sys_natoms is None:
                    sys_natoms = natoms
                else:
                    assert sys_natoms == natoms, "Frames in one system must be the same!"
                sum_natoms += natoms
                for k, v in more_loss.items():
                    if "mae" in k:
                        single_results[k] = single_results.get(k, 0.0) + v * natoms
                    else:
                        single_results[k] = single_results.get(k, 0.0) + v ** 2 * natoms
            if self.detail_file is not None:
                save_detail_file(Path(self.detail_file), system_pred, system_label, sys_natoms, system_name=system, append=(cc != 0))
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


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
):
    """Save numpy array to test file.

    Parameters
    ----------
    fname : str
        filename
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended insted of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)


def save_detail_file(detail_path, system_pred, system_label, natoms, system_name, append=False):
    ntest = len(system_pred)
    data_e = np.concatenate([item['energy'] for item in system_label]).reshape([-1, 1])
    pred_e = np.concatenate([item['energy'] for item in system_pred]).reshape([-1, 1])
    pe = np.concatenate(
        (
            data_e,
            pred_e,
        ),
        axis=1,
    )
    save_txt_file(
        detail_path.with_suffix(".e.out"),
        pe,
        header="%s: data_e pred_e" % system_name,
        append=append,
    )
    pe_atom = pe / natoms
    save_txt_file(
        detail_path.with_suffix(".e_peratom.out"),
        pe_atom,
        header="%s: data_e pred_e" % system_name,
        append=append,
    )
    if "force" in system_pred[0]:
        data_f = np.concatenate([item['force'] for item in system_label]).reshape([-1, 3])
        pred_f = np.concatenate([item['force'] for item in system_pred]).reshape([-1, 3])
        pf = np.concatenate(
            (
                data_f,
                pred_f,
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".f.out"),
            pf,
            header="%s: data_fx data_fy data_fz pred_fx pred_fy pred_fz" % system_name,
            append=append,
        )
