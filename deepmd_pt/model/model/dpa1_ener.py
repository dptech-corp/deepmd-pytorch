import logging
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten
from deepmd_pt.model.task import EnergyFittingNetType
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class EnergyModelDPA1(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(EnergyModelDPA1, self).__init__()
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        descriptor_param['ntypes'] = ntypes
        if descriptor_param.get('type_embedding', None) is None:
            self.type_embedding = TypeEmbedNet(ntypes, 8)
            descriptor_param['tebd_dim'] = 8
            descriptor_param['tebd_input_mode'] = 'concat'
            self.tebd_dim = 8
        else:
            tebd_dim = descriptor_param['type_embedding']['neuron'][-1]
            tebd_input_mode = descriptor_param['type_embedding'].get('tebd_input_mode', 'concat')
            self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
            descriptor_param['tebd_dim'] = tebd_dim
            descriptor_param['tebd_input_mode'] = tebd_input_mode
            self.tebd_dim = tebd_dim

        self.descriptor_type = descriptor_param['type']

        assert self.descriptor_type == 'se_atten', 'Only descriptor `se_atten` is supported for DPA-1!'
        self.descriptor = DescrptSeAtten(**descriptor_param)

        # Statistics
        if not model_params["resuming"]:
            if sampled is not None:
                for sys in sampled:
                    for key in sys:
                        sys[key] = sys[key].to(env.DEVICE)
                sumr, suma, sumn, sumr2, suma2 = self.descriptor.compute_input_stats(sampled)
            else:
                logging.info(f"Loading stat file from {model_params.get('stat_file')}")
                stats = np.load(model_params.get("stat_file"))
                sumr, suma, sumn, sumr2, suma2=stats["sumr"], stats["suma"], stats["sumn"], stats["sumr2"], stats["suma2"]
            self.descriptor.init_desc_stat(sumr, suma, sumn, sumr2, suma2)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener'), 'Only fitting net `ener` is supported!'
        fitting_param['ntypes'] = 1
        fitting_param['embedding_width'] = self.descriptor.dim_out + self.tebd_dim
        if sampled is not None:
            energy = [item['energy'] for item in sampled]
            mixed_type = 'real_natoms_vec' in sampled[0]
            if mixed_type:
                input_natoms = [item['real_natoms_vec'] for item in sampled]
            else:
                input_natoms = [item['natoms'] for item in sampled]
            tmp = compute_output_stats(energy, input_natoms)
            fitting_param['bias_atom_e'] = tmp[:, 0]
        elif not model_params["resuming"]:
            fitting_param['bias_atom_e'] = stats["bias_atom_e"]
        else:
            fitting_param['bias_atom_e'] = [0.0] * ntypes
        fitting_param['use_tebd'] = True
        self.fitting_net = EnergyFittingNetType(**fitting_param)

        if sampled is not None:
            logging.info(f"Saving stat file to {model_params.get('stat_file')}")
            np.savez_compressed(model_params.get("stat_file"), sumr=sumr, suma=suma, sumn=sumn, sumr2=sumr2, suma2=suma2, bias_atom_e=fitting_param['bias_atom_e'])

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type, selected_loc=None, box=None):
        """Return total energy of the system.
        Args:
        - coord: Atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Atom types with shape [nframes, natoms[1]].
        - natoms: Atom statisics with shape [self.ntypes+2].
        - box: Simulation box with shape [nframes, 9].
        Returns:
        - energy: Energy per atom.
        - force: XYZ force per atom.
        """
        index = mapping.unsqueeze(-1).expand(-1, -1, 3)
        # index nframes x nall x 3
        # coord nframes x nloc x 3
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        extended_coord.requires_grad_(True)
        atype_tebd = self.type_embedding(atype)
        selected_type[selected_type == -1] = self.ntypes
        nlist_tebd = self.type_embedding(selected_type)

        descriptor, env_mat, _ = self.descriptor(extended_coord, selected, atype, selected_type, atype_tebd, nlist_tebd)
        atom_energy = self.fitting_net(descriptor, atype, atype_tebd)
        energy = atom_energy.sum(dim=1)
        faked_grad = torch.ones_like(energy)
        lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
        extended_force = torch.autograd.grad([energy], [extended_coord], grad_outputs=lst, create_graph=True)[0]
        assert extended_force is not None
        virial = -torch.transpose(extended_coord, 1, 2)@extended_force
        mapping = mapping.unsqueeze(-1).expand(-1, -1, 3)
        force = torch.zeros_like(coord)
        force = torch.scatter_reduce(force, 1, index=mapping, src=extended_force, reduce='sum')
        force = -force
        model_predict = {'energy': energy,
                         'force': force,
                         'virial': virial,
                         }
        return model_predict
