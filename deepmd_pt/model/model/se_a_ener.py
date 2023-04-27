import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeA, DescrptSeAtten
from deepmd_pt.model.task.ener import EnergyFittingNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class EnergyModelSeA(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - training_data: The training dataset.
        """
        super(EnergyModelSeA, self).__init__()
        ntypes = len(model_params['type_map'])
        # Descriptor + Embedding Net
        descriptor_param = model_params.pop('descriptor')
        self.descriptor_type = descriptor_param['type']

        if self.descriptor_type == 'se_e2_a':
            self.descriptor = DescrptSeA(**descriptor_param)
        else:
            NotImplementedError('Only descriptor `se_e2_a` is supported for se_a model!')

        # Statistics
        if sampled is not None:
            for sys in sampled:
                for key in ['coord', 'force', 'energy', 'atype', 'natoms', 'extended_coord', 'selected', 'shift', 'mapping']:
                    if key in sys.keys():
                        sys[key] = sys[key].to(env.DEVICE)
            self.descriptor.compute_input_stats(sampled)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener'), 'Only fitting net `ener` is supported!'
        fitting_param['ntypes'] = self.descriptor.ntypes
        fitting_param['embedding_width'] = self.descriptor.dim_out
        if sampled is not None:
            energy = [item['energy'] for item in sampled]
            natoms = [item['natoms'] for item in sampled]
            tmp = compute_output_stats(energy, natoms)
            fitting_param['bias_atom_e'] = tmp[:, 0]
        else:
            fitting_param['bias_atom_e'] = [0.0] * ntypes
        self.fitting_net = EnergyFittingNet(**fitting_param)

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type=None, box=None):
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
        descriptor = self.descriptor(extended_coord, selected, atype)
        atom_energy = self.fitting_net(descriptor, atype)
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
