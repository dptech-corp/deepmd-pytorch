import math
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
        dataset_params = training_params.pop('validation_data')
        systems = dataset_params['systems']
        data = DpLoaderSet(systems, dataset_params['batch_size'], model_params)
        data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
        sampled = make_stat_input(data.systems, data.dataloaders, data_stat_nbatch)

        self.dataloader = DataLoader(
            data,
            sampler=torch.utils.data.RandomSampler(data),
            batch_size=None,
            num_workers=8,  # setting to 0 diverges the behavior of its iterator; should be >=1
            drop_last=False,
        )
        self.data = BufferedIterator(iter(self.dataloader))

        if model_params["descriptor"]["type"] == "se_e2_a":
            self.model = EnergyModelSeA(model_params, sampled).to(DEVICE)
        elif model_params["descriptor"]["type"] == "se_atten":
            self.model = EnergyModelDPA1(model_params, sampled).to(DEVICE)
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

    def get_data(self):
        batch_data = next(iter(self.data))
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
        single_results = {}
        sum_natoms = 0
        for _ in range(self.numb_test):
            try:
                input_dict, label_dict = self.get_data()
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
            print(f"{item}: {results[item]:.4f}")
