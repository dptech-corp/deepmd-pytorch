import torch
from typing import Optional, List, Union, Dict
from deepmd_pt.model.descriptor import Descriptor
from deepmd_pt.model.task import Fitting, DenoiseNet
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.model.model import BaseModel
from deepmd_pt.utils.nlist import extend_coord_with_ghosts, build_neighbor_list
from deepmd_pt.utils.region import normalize_coord
from deepmd_pt.model.descriptor import make_default_type_embedding

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
        self.type_map = type_map
        self.ntypes = ntypes
        descriptor['ntypes'] = ntypes
        self.combination = descriptor.get('combination',False)
        if(self.combination):
            self.prefactor=descriptor.get('prefactor', [0.5,0.5])
        self.descriptor_type = descriptor['type']

        self.type_split = True
        if self.descriptor_type not in ['se_e2_a']:
            self.type_split = False

        self.descriptor = Descriptor(**descriptor)
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.split_nlist = False

        # Statistics
        self.compute_or_load_stat(fitting_net, ntypes,
                                  resuming=resuming,
                                  type_map=type_map,
                                  stat_file_dir=stat_file_dir,
                                  stat_file_path=stat_file_path,
                                  sampled=sampled)

        # Fitting
        if fitting_net:
            fitting_net['type'] = fitting_net.get('type', 'ener')
            if self.descriptor_type not in ['se_e2_a']:
                fitting_net['ntypes'] = 1
            else:
                fitting_net['ntypes'] = self.descriptor.get_ntype()
                fitting_net['use_tebd'] = False
            fitting_net['embedding_width'] = self.descriptor.dim_out

            self.grad_force = 'direct' not in fitting_net['type']
            if not self.grad_force:
                fitting_net['out_dim'] = self.descriptor.dim_emb
                if 'ener' in fitting_net['type']:
                    fitting_net['return_energy'] = True
            self.fitting_net = Fitting(**fitting_net)
        else:
            self.fitting_net = None
            self.grad_force = False
            if not self.split_nlist:
                self.coord_denoise_net = DenoiseNet(self.descriptor.dim_out, self.ntypes - 1, self.descriptor.dim_emb)
            elif self.combination:
                self.coord_denoise_net = DenoiseNet(self.descriptor.dim_out, self.ntypes - 1, self.descriptor.dim_emb_list, self.prefactor)
            else:
                self.coord_denoise_net = DenoiseNet(self.descriptor.dim_out, self.ntypes - 1, self.descriptor.dim_emb)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None, 
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return total energy of the system.
        Args:
        - coord: Atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Atom types with shape [nframes, natoms[1]].
        - natoms: Atom statisics with shape [self.ntypes+2].
        - box: Simulation box with shape [nframes, 9].
        - atomic_virial: Whether or not compoute the atomic virial.
        Returns:
        - energy: Energy per atom.
        - force: XYZ force per atom.
        """
        nframes, nloc = atype.shape[:2]
        if box is not None:
            coord_normalized = normalize_coord(coord, box.reshape(-1, 3, 3))
        else:
            coord_normalized = coord.clone()
        rcut_max = self.rcut
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, box, rcut_max)
        nlist = build_neighbor_list(
            extended_coord, extended_atype, nloc,
            self.rcut, self.sel, distinguish_types=self.type_split)
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        model_predict_lower = self.forward_lower(extended_coord, extended_atype, nlist, mapping, do_atomic_virial=do_atomic_virial)
        if self.fitting_net is not None:
            if self.grad_force:
                mapping = mapping.unsqueeze(-1).expand(-1, -1, 3)
                force = torch.zeros_like(coord)
                model_predict_lower['force'] = torch.scatter_reduce(force, 1, index=mapping,
                                                                src=model_predict_lower['extended_force'], reduce='sum')
                atomic_virial = torch.zeros_like(coord).unsqueeze(-1).expand(-1, -1, -1, 3)
                mapping = mapping.unsqueeze(-1).expand(-1, -1, -1, 3)
                reduced_virial = torch.scatter_reduce(atomic_virial, 1, index=mapping,
                                                  src=model_predict_lower['extended_virial'],
                                                  reduce='sum')
                model_predict_lower['virial'] = torch.sum(reduced_virial, dim=1)
                if do_atomic_virial:
                    model_predict_lower['atomic_virial'] = reduced_virial
            else:
                model_predict_lower['force'] = model_predict_lower['dforce']
        else:
            model_predict_lower['updated_coord'] += coord
        return model_predict_lower

    def forward_lower(
        self, 
        extended_coord, 
        extended_atype, 
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.grad_force:
            extended_coord.requires_grad_(True)
        descriptor, env_mat, diff, rot_mat, sw = \
          self.descriptor(
            nlist,
            extended_coord,
            extended_atype,
            mapping=mapping,
          )

        assert descriptor is not None
        # energy, force
        if self.fitting_net is not None:
            atom_energy, dforce = self.fitting_net(descriptor, atype, atype_tebd=None, rot_mat=rot_mat)
            energy = atom_energy.sum(dim=1)
            model_predict = {'energy': energy,
                            'atom_energy': atom_energy,
                            }
            if self.grad_force:
                faked_grad = torch.ones_like(energy)
                lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
                extended_force = torch.autograd.grad([energy], [extended_coord], grad_outputs=lst, create_graph=True)[0]
                assert extended_force is not None
                extended_force = -extended_force
                extended_virial = extended_force.unsqueeze(-1) @ extended_coord.unsqueeze(-2)
                model_predict['extended_force'] = extended_force
                if do_atomic_virial:
                    # the correction sums to zero, which does not contribute to global virial
                    extended_virial_corr = self.atomic_virial_corr(extended_coord, atom_energy)
                    model_predict['extended_virial'] = extended_virial + extended_virial_corr
                else:
                    model_predict['extended_virial'] = extended_virial
            else:
                assert dforce is not None
                model_predict['dforce'] = dforce
        # denoise
        else:
            nlist_list = [nlist]
            if not self.split_nlist:
                nnei_mask = nlist != -1
            elif self.combination:
                nnei_mask = []
                for item in nlist_list:
                    nnei_mask_item = item != -1
                    nnei_mask.append(nnei_mask_item)
            else:
                env_mat = env_mat[-1]
                diff = diff[-1]
                nnei_mask = nlist_list[-1] != -1
            updated_coord, logits = self.coord_denoise_net(env_mat, diff, nnei_mask, descriptor, sw)
            model_predict = {'updated_coord': updated_coord,
                             'logits': logits,
                            }

        return model_predict


    def atomic_virial_corr(self, extended_coord, atom_energy):
      nall = extended_coord.shape[1]
      nloc = atom_energy.shape[1]
      coord, _ = torch.split(extended_coord, [nloc, nall-nloc], dim=1)
      # no derivative with respect to the loc coord.
      coord = coord.detach()
      ce = coord * atom_energy
      sumce0, sumce1, sumce2 = torch.split(torch.sum(ce, dim=1), [1,1,1], dim=-1)
      faked_grad = torch.ones_like(sumce0)
      lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
      extended_virial_corr0 = torch.autograd.grad([sumce0], [extended_coord], grad_outputs=lst, create_graph=True)[0]
      assert extended_virial_corr0 is not None
      extended_virial_corr1 = torch.autograd.grad([sumce1], [extended_coord], grad_outputs=lst, create_graph=True)[0]
      assert extended_virial_corr1 is not None
      extended_virial_corr2 = torch.autograd.grad([sumce2], [extended_coord], grad_outputs=lst, create_graph=True)[0]
      assert extended_virial_corr2 is not None
      extended_virial_corr = torch.concat([extended_virial_corr0.unsqueeze(-1),
                                           extended_virial_corr1.unsqueeze(-1),
                                           extended_virial_corr2.unsqueeze(-1)], dim=-1)
      return extended_virial_corr
      



# should be a stand-alone function!!!!
def process_nlist(
    nlist, 
    extended_atype, 
    mapping: Optional[torch.Tensor] = None,
):
    # process the nlist_type and nlist_loc
    nframes, nloc = nlist.shape[:2]
    nmask = nlist == -1
    nlist[nmask] = 0
    if mapping is not None:
        nlist_loc = torch.gather(
          mapping, 
          dim=1, 
          index=nlist.reshape(nframes, -1),
        ).reshape(nframes, nloc, -1)
        nlist_loc[nmask] = -1
    else:
        nlist_loc = None
    nlist_type = torch.gather(
      extended_atype, 
      dim=1, 
      index=nlist.reshape(nframes, -1),
    ).reshape(nframes, nloc,-1)
    nlist_type[nmask] = -1
    nlist[nmask] = -1
    return nlist_loc, nlist_type, nframes, nloc

def process_nlist_gathered(
    nlist, 
    extended_atype, 
    split_sel: List[int],
    mapping: Optional[torch.Tensor] = None,
):
    nlist_list = list(torch.split(nlist, split_sel, -1))
    nframes, nloc = nlist_list[0].shape[:2]
    nlist_type_list = []
    nlist_loc_list = []
    for nlist_item in nlist_list:
        nmask = nlist_item == -1
        nlist_item[nmask] = 0
        if mapping is not None:
            nlist_loc_item = torch.gather(mapping, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
                nframes, nloc,
                -1)
            nlist_loc_item[nmask] = -1
            nlist_loc_list.append(nlist_loc_item)
        nlist_type_item = torch.gather(extended_atype, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
            nframes,
            nloc,
            -1)
        nlist_type_item[nmask] = -1
        nlist_type_list.append(nlist_type_item)
        nlist_item[nmask] = -1

    if mapping is not None:
        nlist_loc = torch.cat(nlist_loc_list, -1)
    else:
        nlist_loc = None
    nlist_type = torch.cat(nlist_type_list, -1)
    return nlist_loc, nlist_type, nframes, nloc
