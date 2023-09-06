from pathlib import Path
import torch
import numpy as np
from deepmd_pt.utils.env import DEVICE, PREPROCESS_DEVICE, GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.model.model import get_model
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils.preprocess import Region3D, normalize_coord, make_env_mat
from deepmd_pt.utils.dataloader import collate_batch
from typing import Optional, Union, List
from deepmd_pt.utils import env


class DeepEval:
    def __init__(
            self,
            model_file: "Path"
    ):
        self.model_path = model_file
        state_dict = torch.load(model_file, map_location=env.DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
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
            coords: Union[np.ndarray, torch.Tensor],
            cells: Optional[Union[np.ndarray, torch.Tensor]],
            atom_types: Union[np.ndarray, torch.Tensor, List[int]],
            atomic: bool = False,
            infer_batch_size: int = 2,
    ):
        raise NotImplementedError


class DeepPot(DeepEval):
    def __init__(
            self,
            model_file: "Path"
    ):
        super(DeepPot, self).__init__(model_file)

    def eval(
            self,
            coords: Union[np.ndarray, torch.Tensor],
            cells: Optional[Union[np.ndarray, torch.Tensor]],
            atom_types: Union[np.ndarray, torch.Tensor, List[int]],
            atomic: bool = False,
            infer_batch_size: int = 2,
    ):
        return eval_model(self.dp, coords, cells, atom_types, atomic, infer_batch_size)


def eval_model(
    model,
    coords: Union[np.ndarray, torch.Tensor],
    cells: Optional[Union[np.ndarray, torch.Tensor]],
    atom_types: Union[np.ndarray, torch.Tensor, List[int]],
    atomic: bool = False,
    infer_batch_size: int = 2,
):
    model = model.to(DEVICE)
    energy_out = []
    atomic_energy_out = []
    force_out = []
    virial_out = []
    atomic_virial_out = []
    err_msg = f"All inputs should be the same format, " \
              f"but found {type(coords)}, {type(cells)}, {type(atom_types)} instead! "
    return_tensor = True
    if isinstance(coords, torch.Tensor):
        if cells is not None:
            assert isinstance(cells, torch.Tensor), err_msg
        assert isinstance(atom_types, torch.Tensor) or isinstance(atom_types, list)
        atom_types = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
    elif isinstance(coords, np.ndarray):
        if cells is not None:
            assert isinstance(cells, np.ndarray), err_msg
        assert isinstance(atom_types, np.ndarray) or isinstance(atom_types, list)
        atom_types = np.array(atom_types, dtype=np.int32)
        return_tensor = False

    nframes = coords.shape[0]
    if len(atom_types.shape) == 1:
        natoms = len(atom_types)
        if isinstance(atom_types, torch.Tensor):
            atom_types = torch.tile(atom_types.unsqueeze(0), [nframes, 1]).reshape(nframes, -1)
        else:
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
    else:
        natoms = len(atom_types[0])

    coord_input = torch.tensor(coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
    type_input = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
    box_input = None
    if cells is None:
        pbc = False
    else:
        pbc = True
        box_input = torch.tensor(cells.reshape([-1, 3, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
    num_iter = int((nframes + infer_batch_size - 1) / infer_batch_size)

    for ii in range(num_iter):
        batch_coord = coord_input[ii * infer_batch_size:(ii + 1) * infer_batch_size]
        batch_atype = type_input[ii * infer_batch_size:(ii + 1) * infer_batch_size]
        batch_box = None
        if pbc:
            batch_box = box_input[ii * infer_batch_size:(ii + 1) * infer_batch_size]
        batch_output = model(batch_coord, batch_atype, box=batch_box)
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        if not return_tensor:
            if 'energy' in batch_output:
                energy_out.append(batch_output['energy'].detach().cpu().numpy())
            if 'atom_energy' in batch_output:
                atomic_energy_out.append(batch_output['atom_energy'].detach().cpu().numpy())
            if 'force' in batch_output:
                force_out.append(batch_output['force'].detach().cpu().numpy())
            if 'virial' in batch_output:
                virial_out.append(batch_output['virial'].detach().cpu().numpy())
            if 'atomic_virial' in batch_output:
                atomic_virial_out.append(batch_output['atomic_virial'].detach().cpu().numpy())
        else:
            if 'energy' in batch_output:
                energy_out.append(batch_output['energy'])
            if 'atom_energy' in batch_output:
                atomic_energy_out.append(batch_output['atom_energy'])
            if 'force' in batch_output:
                force_out.append(batch_output['force'])
            if 'virial' in batch_output:
                virial_out.append(batch_output['virial'])
            if 'atomic_virial' in batch_output:
                atomic_virial_out.append(batch_output['atomic_virial'])
    if not return_tensor:
        energy_out = np.concatenate(energy_out) if energy_out else np.zeros([nframes, 1])
        atomic_energy_out = np.concatenate(atomic_energy_out) if atomic_energy_out else np.zeros([nframes, natoms, 1])
        force_out = np.concatenate(force_out) if force_out else np.zeros([nframes, natoms, 3])
        virial_out = np.concatenate(virial_out) if virial_out else np.zeros([nframes, 3, 3])
        atomic_virial_out = np.concatenate(atomic_virial_out) if atomic_virial_out else np.zeros([nframes, natoms, 3, 3])
    else:
        energy_out = torch.cat(energy_out) if energy_out else torch.zeros([nframes, 1], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
        atomic_energy_out = torch.cat(atomic_energy_out) if atomic_energy_out else torch.zeros([nframes, natoms, 1], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
        force_out = torch.cat(force_out) if force_out else torch.zeros([nframes, natoms, 3], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
        virial_out = torch.cat(virial_out) if virial_out else torch.zeros([nframes, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
        atomic_virial_out = torch.cat(atomic_virial_out) if atomic_virial_out else torch.zeros([nframes, natoms, 3, 3], dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
    if not atomic:
        return energy_out, force_out, virial_out
    else:
        return energy_out, force_out, virial_out, atomic_energy_out, atomic_virial_out
