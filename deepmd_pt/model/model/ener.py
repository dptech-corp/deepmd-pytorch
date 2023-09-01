import torch
from typing import Optional, List, Union
from deepmd_pt.model.descriptor import Descriptor
from deepmd_pt.model.task import Fitting
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.model.model import BaseModel
from deepmd_pt.utils.nlist import extend_coord_with_ghosts, build_neighbor_list
from deepmd_pt.utils.region import phys2inter, inter2phys


class EnergyModel(BaseModel):
    """Energy model.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    type_embedding
            Type embedding net
    resuming
            Whether to resume/fine-tune from checkpoint or not.
    stat_file_dir
            The directory to the state files.
    stat_file_path
            The path to the state files.
    sampled
            Sampled frames to compute the statistics.
    """

    model_type = "ener"

    def __init__(
            self,
            descriptor: dict,
            fitting_net: dict,
            type_map: Optional[List[str]],
            type_embedding: Optional[dict] = None,
            resuming: bool = False,
            stat_file_dir=None,
            stat_file_path=None,
            sampled=None,
            **kwargs,
    ):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(EnergyModel, self).__init__()
        # Descriptor + Type Embedding Net (Optional)
        ntypes = len(type_map)
        self.ntypes = ntypes
        descriptor['ntypes'] = ntypes
        self.descriptor_type = descriptor['type']

        self.has_type_embedding = False
        self.type_split = True
        if self.descriptor_type not in ['se_e2_a']:
            self.has_type_embedding = True
            self.type_split = False
            if type_embedding is None:
                self.type_embedding = TypeEmbedNet(ntypes, 8)
                descriptor['tebd_dim'] = 8
                descriptor['tebd_input_mode'] = 'concat'
                self.tebd_dim = 8
            else:
                tebd_dim = type_embedding.get('neuron', [8])[-1]
                tebd_input_mode = type_embedding.get('tebd_input_mode', 'concat')
                self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
                descriptor['tebd_dim'] = tebd_dim
                descriptor['tebd_input_mode'] = tebd_input_mode
                self.tebd_dim = tebd_dim

        self.descriptor = Descriptor(**descriptor)
        self.rcut = self.descriptor.rcut
        self.sel = self.descriptor.sel

        # Statistics
        self.compute_or_load_stat(fitting_net, ntypes,
                                  resuming=resuming,
                                  type_map=type_map,
                                  stat_file_dir=stat_file_dir,
                                  stat_file_path=stat_file_path,
                                  sampled=sampled)

        # Fitting
        fitting_net['type'] = fitting_net.get('type', 'ener')
        if self.descriptor_type not in ['se_e2_a']:
            fitting_net['ntypes'] = 1
            fitting_net['embedding_width'] = self.descriptor.dim_out + self.tebd_dim
        else:
            fitting_net['ntypes'] = self.descriptor.ntypes
            fitting_net['use_tebd'] = False
            fitting_net['embedding_width'] = self.descriptor.dim_out

        self.grad_force = 'direct' not in fitting_net['type']
        if not self.grad_force:
            fitting_net['out_dim'] = self.descriptor.dim_emb
            if 'ener' in fitting_net['type']:
                fitting_net['return_energy'] = True
        self.fitting_net = Fitting(**fitting_net)

    def forward(self, coord, atype, box: Optional[torch.Tensor] = None, **kwargs):
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
        nframes, nloc = atype.shape[:2]
        if box is not None:
            box = box.reshape(-1, 3, 3)
            inter_cood = torch.remainder(phys2inter(coord, box), 1.0)
            coord_normalized = inter2phys(inter_cood, box)
        else:
            coord_normalized = coord.clone()
        if not isinstance(self.rcut, list):
            rcut_max = self.rcut
        else:
            rcut_max = max(self.rcut)
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, box, rcut_max)
        if not isinstance(self.rcut, list):
            nlist = build_neighbor_list(
                extended_coord, extended_atype, nloc,
                self.rcut, self.sel, distinguish_types=self.type_split)
        else:
            nlist = []
            for rcut, sel in zip(self.rcut, self.sel):
                nlist.append(
                    build_neighbor_list(
                        extended_coord, extended_atype, nloc,
                        rcut, sel, distinguish_types=self.type_split))
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        model_predict_lower = self.forward_lower(extended_coord, extended_atype, nlist, mapping)
        if self.grad_force:
            mapping = mapping.unsqueeze(-1).expand(-1, -1, 3)
            force = torch.zeros_like(coord)
            model_predict_lower['force'] = torch.scatter_reduce(force, 1, index=mapping, src=model_predict_lower['force'], reduce='sum')
        return model_predict_lower

    def forward_lower(self, extended_coord, extended_atype, nlist, mapping: Optional[torch.Tensor] = None):
        nlist_loc, nlist_type, nframes, nloc = self.process_nlist(nlist, extended_atype, mapping=mapping)
        atype = extended_atype[:, :nloc]
        if self.grad_force:
            extended_coord.requires_grad_(True)
        atype_tebd = None
        nlist_tebd = None
        if self.has_type_embedding:
            atype_tebd = self.type_embedding(atype)
            if not isinstance(nlist, list):
                assert nlist_type is not None
                nlist_type[nlist_type == -1] = self.ntypes
                nlist_tebd = self.type_embedding(nlist_type)
            else:
                nlist_tebd = []
                for nlist_type_item in nlist_type:
                    nlist_type_item[nlist_type_item == -1] = self.ntypes
                    nlist_tebd.append(self.type_embedding(nlist_type_item))

        descriptor, env_mat, diff, rot_mat = self.descriptor(extended_coord, nlist, atype, nlist_type=nlist_type,
                                                             nlist_loc=nlist_loc, atype_tebd=atype_tebd,
                                                             nlist_tebd=nlist_tebd)
        atom_energy, force = self.fitting_net(descriptor, atype, atype_tebd=atype_tebd, rot_mat=rot_mat)
        energy = atom_energy.sum(dim=1)
        model_predict = {'energy': energy,
                         'atom_energy': atom_energy,
                         }
        if self.grad_force:
            faked_grad = torch.ones_like(energy)
            lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
            force = -torch.autograd.grad([energy], [extended_coord], grad_outputs=lst, create_graph=True)[0]
            assert force is not None
            virial = torch.transpose(force, 1, 2) @ extended_coord
            model_predict['virial'] = virial
        model_predict['force'] = force
        return model_predict

    @staticmethod
    def process_nlist(nlist, extended_atype, mapping: Optional[torch.Tensor] = None):
        # process the nlist_type and nlist_loc
        if not isinstance(nlist, list):
            nframes, nloc = nlist.shape[:2]
            nmask = nlist == -1
            nlist[nmask] = 0
            nlist_loc = None
            if mapping is not None:
                nlist_loc = torch.gather(mapping, dim=1, index=nlist.reshape(nframes, -1)).reshape(nframes, nloc, -1)
                nlist_loc[nmask] = -1
            nlist_type = torch.gather(extended_atype, dim=1, index=nlist.reshape(nframes, -1)).reshape(nframes, nloc,
                                                                                                       -1)
            nlist_type[nmask] = -1
            nlist[nmask] = -1
        else:
            nframes, nloc = nlist[0].shape[:2]
            nlist_type = []
            nlist_loc = []
            for nlist_item in nlist:
                nmask = nlist_item == -1
                nlist_item[nmask] = 0
                if mapping is not None:
                    nlist_loc_item = torch.gather(mapping, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
                        nframes, nloc,
                        -1)
                    nlist_loc_item[nmask] = -1
                    nlist_loc.append(nlist_loc_item)
                else:
                    nlist_loc = None
                nlist_type_item = torch.gather(extended_atype, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
                    nframes,
                    nloc,
                    -1)
                nlist_type_item[nmask] = -1
                nlist_type.append(nlist_type_item)
                nlist_item[nmask] = -1
        return nlist_loc, nlist_type, nframes, nloc
