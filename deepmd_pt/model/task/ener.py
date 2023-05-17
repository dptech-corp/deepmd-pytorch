import logging
import numpy as np
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import ResidualDeep
from deepmd_pt.model.task import TaskBaseMethod
from deepmd_pt.utils.env import DEVICE
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input

class EnergyFittingNet(TaskBaseMethod):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, use_tebd=False, **kwargs):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(EnergyFittingNet, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        self.use_tebd = use_tebd
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), 'Element count mismatches!'
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer('bias_atom_e', bias_atom_e)

        filter_layers = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(type_i, embedding_width, neuron, bias_atom_e[type_i], resnet_dt=resnet_dt)
            filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self, inputs, atype):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = 0
        for type_i, filter_layer in enumerate(self.filter_layers):
            mask = atype == type_i
            atom_energy = filter_layer(inputs)
            if not env.ENERGY_BIAS_TRAINABLE:
                atom_energy = atom_energy + self.bias_atom_e[type_i]
            atom_energy = atom_energy * mask.unsqueeze(-1)
            outs = outs + atom_energy # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)


class EnergyFittingNetType(TaskBaseMethod):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, **kwargs):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(EnergyFittingNetType, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer('bias_atom_e', bias_atom_e)

        filter_layers = []
        one = ResidualDeep(0, embedding_width, neuron, 0.0, resnet_dt=resnet_dt)
        filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self, inputs, atype, atype_tebd):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = 0
        inputs = torch.concat([inputs, atype_tebd], dim=-1)
        atom_energy = self.filter_layers[0](inputs) + self.bias_atom_e[atype].unsqueeze(-1)
        outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)

    def change_energy_bias(self, config, model, old_type_map, new_type_map, bias_shift='delta', ntest=10):
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        config : Dict
            The configuration.
        model : EnergyModel
            Energy model loaded pre-trained model.
        new_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        old_type_map : str
            The full type_map in pretrained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        ntest : int
            The number of test samples in a system to change the energy bias.
        """
        logging.info(
            "Changing energy bias in pretrained model for types {}... "
            "(this step may take long time)".format(str(new_type_map))
        )
        # data
        systems = config['training']['training_data']['systems']
        finetune_data = DpLoaderSet(systems, ntest, config["model"], type_split=False, noise_settings=None)
        sampled = make_stat_input(finetune_data.systems, finetune_data.dataloaders, 1)
        # map
        sorter = np.argsort(old_type_map)
        idx_type_map = sorter[
            np.searchsorted(old_type_map, new_type_map, sorter=sorter)
        ]
        mixed_type = np.all([i.mixed_type for i in finetune_data.systems])
        numb_type = len(old_type_map)
        type_numbs, energy_ground_truth, energy_predict = [], [], []
        for test_data in sampled:
            if mixed_type:
                atype = test_data["atype"]
            else:
                atype = test_data["atype"][0]
            assert np.array(
                [i.item() in idx_type_map for i in list(set(atype.reshape(-1)))]
            ).all(), "Some types are not in 'type_map'!"
            energy_ground_truth.append(
                test_data["energy"].cpu().numpy()
            )
            if mixed_type:
                type_numbs.append(
                    np.array(
                        [(atype == i).sum(axis=-1) for i in idx_type_map],
                        dtype=np.int32,
                    ).T
                )
            else:
                type_numbs.append(
                    np.tile(
                        np.bincount(atype.detach().cpu().numpy(), minlength=numb_type)[idx_type_map],
                        (ntest, 1),
                    )
                )
            if bias_shift == "delta":
                coord = test_data["coord"].to(DEVICE)
                atype = test_data['atype'].to(DEVICE)
                natoms = test_data["natoms"].to(DEVICE)
                mapping = test_data['mapping'].to(DEVICE)
                shift = test_data['shift'].to(DEVICE)
                selected = test_data['selected'].to(DEVICE)
                selected_type = test_data['selected_type'].to(DEVICE)
                selected_loc = test_data['selected_loc'].to(DEVICE)
                ret = model(coord, atype, natoms, mapping, shift, selected, selected_type, selected_loc)
                energy_predict.append(ret['energy'].reshape([ntest, 1]).detach().cpu().numpy())
        type_numbs = np.concatenate(type_numbs)
        energy_ground_truth = np.concatenate(energy_ground_truth)
        old_bias = self.bias_atom_e[idx_type_map]
        if bias_shift == "delta":
            energy_predict = np.concatenate(energy_predict)
            bias_diff = energy_ground_truth - energy_predict
            delta_bias = np.linalg.lstsq(type_numbs, bias_diff, rcond=None)[0]
            unbias_e = energy_predict + type_numbs @ delta_bias
            atom_numbs = type_numbs.sum(-1)
            rmse_ae = (
                np.sqrt(np.square(unbias_e - energy_ground_truth)) / atom_numbs
            ).mean()
            self.bias_atom_e[idx_type_map] += torch.from_numpy(delta_bias.reshape(-1)).to(DEVICE)
            logging.info(
                "RMSE of atomic energy after linear regression is: {:10.5e} eV/atom.".format(
                    rmse_ae
                )
            )
        elif bias_shift == "statistic":
            statistic_bias = np.linalg.lstsq(
                type_numbs, energy_ground_truth, rcond=None
            )[0]
            self.bias_atom_e[idx_type_map] = torch.from_numpy(statistic_bias.reshape(-1)).to(DEVICE)
        else:
            raise RuntimeError("Unknown bias_shift mode: " + bias_shift)
        logging.info(
            "Change energy bias of {} from {} to {}.".format(
                str(new_type_map), str(old_bias.detach().cpu().numpy()), str(self.bias_atom_e[idx_type_map].detach().cpu().numpy())
            )
        )
        return None