from pathlib import Path
import torch
import numpy as np
from deepmd_pt.utils.env import DEVICE, PREPROCESS_DEVICE, GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.model.model import get_model
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils.preprocess import Region3D, normalize_coord, make_env_mat
from deepmd_pt.utils.dataloader import collate_batch


class DeepEval:
    def __init__(
            self,
            model_file: "Path"
    ):
        self.model_path = model_file
        state_dict = torch.load(model_file)
        self.input_param = state_dict['_extra_state']['model_params']
        self.input_param['resuming'] = True
        self.multi_task = "model_dict" in self.input_param
        assert not self.multi_task, "multitask mode currently not supported!"
        self.type_split = self.input_param['descriptor']['type'] in ['se_e2_a']
        self.dp = ModelWrapper(get_model(self.input_param, None).to(DEVICE))
        self.dp.load_state_dict(state_dict)
        self.rcut = self.dp.model['Default'].descriptor.rcut
        self.sec = self.dp.model['Default'].descriptor.sec

    def eval(
            self,
            coords: np.ndarray,
            atom_types: np.ndarray,
            cells: np.ndarray = None,
            atomic: bool = False,
            mixed_type: bool = False,
            infer_frames: int = 10,
    ):
        raise NotImplementedError


class DeepEner(DeepEval):
    def __init__(
            self,
            model_file: "Path"
    ):
        super(DeepEner, self).__init__(model_file)

    def eval(
            self,
            coords: np.ndarray,
            atom_types: np.ndarray,
            cells: np.ndarray = None,
            atomic: bool = False,
            mixed_type: bool = False,
            infer_frames: int = 3,
    ):
        energy_out = []
        atomic_energy_out = []
        force_out = []
        virial_out = []
        atom_types = atom_types.astype(np.int32)
        nframes = coords.shape[0]
        if not mixed_type:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])
        coords = np.reshape(np.array(coords), [-1, natoms, 3])
        coord_input = torch.tensor(coords, dtype=GLOBAL_PT_FLOAT_PRECISION, device=PREPROCESS_DEVICE)
        type_input = torch.tensor(atom_types, dtype=torch.long, device=PREPROCESS_DEVICE)
        box_input = None
        if cells is None:
            pbc = False
        else:
            pbc = True
            box_input = torch.tensor(np.reshape(np.array(cells), [-1, 3, 3]),
                                     dtype=GLOBAL_PT_FLOAT_PRECISION, device=PREPROCESS_DEVICE)
        num_iter = int((nframes + infer_frames - 1) / infer_frames)
        for ii in range(num_iter):
            coord_tmp = coord_input[ii * infer_frames:(ii + 1) * infer_frames]
            type_tmp = type_input[ii * infer_frames:(ii + 1) * infer_frames]
            box_tmp = None
            if pbc:
                box_tmp = box_input[ii * infer_frames:(ii + 1) * infer_frames]
            batches = []
            batch_frames = coord_tmp.size()[0]
            for ind in range(batch_frames):
                batch = {}
                _coord = coord_tmp[ind]
                atype = type_tmp[ind]
                batch['atype'] = atype
                if pbc:
                    single_box = box_tmp[ind]
                    region = Region3D(single_box)
                    _coord = normalize_coord(_coord, region, natoms)
                else:
                    region = None
                batch['coord'] = _coord
                nlist, nlist_loc, nlist_type, shift, mapping = make_env_mat(_coord, atype, region, self.rcut, self.sec,
                                                                            pbc=pbc,
                                                                            type_split=self.type_split)
                batch['nlist'] = nlist
                batch['nlist_loc'] = nlist_loc
                batch['nlist_type'] = nlist_type
                batch['shift'] = shift
                batch['mapping'] = mapping
                batches.append(batch)
            batch_input = collate_batch(batches)
            [batch_coord, batch_atype, batch_mapping, batch_shift, batch_nlist, batch_nlist_type, batch_nlist_loc] \
                = [batch_input[item].to(DEVICE) if not isinstance(batch_input[item], list) else
                   [kk.to(DEVICE) for kk in batch_input[item]] for item in
                   ['coord', 'atype', 'mapping', 'shift', 'nlist', 'nlist_type', 'nlist_loc']]
            batch_output, _, _ = self.dp(batch_coord, batch_atype, None, batch_mapping, batch_shift, batch_nlist,
                                         batch_nlist_type, batch_nlist_loc)
            energy_out.append(batch_output['energy'].detach().cpu().numpy())
            atomic_energy_out.append(batch_output['atom_energy'].detach().cpu().numpy())
            force_out.append(batch_output['force'].detach().cpu().numpy())
            virial_out.append(batch_output['virial'].detach().cpu().numpy())
        energy_out = np.concatenate(energy_out)
        atomic_energy_out = np.concatenate(atomic_energy_out)
        force_out = np.concatenate(force_out)
        virial_out = np.concatenate(virial_out)
        if not atomic:
            return energy_out, force_out, virial_out
        else:
            return energy_out, force_out, virial_out, atomic_energy_out
