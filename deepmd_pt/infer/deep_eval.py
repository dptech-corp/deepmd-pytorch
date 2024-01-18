from pathlib import Path
import torch
import numpy as np
from deepmd_pt.utils.env import DEVICE, PREPROCESS_DEVICE, GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.model.model import get_model
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils.preprocess import Region3D, normalize_coord, make_env_mat
from deepmd_pt.utils.dataloader import collate_batch
from typing import Callable, Optional, Tuple, Union, List
from deepmd_pt.utils import env
from deepmd_pt.utils.auto_batch_size import AutoBatchSize


class DeepEval:
    def __init__(
            self,
            model_file: "Path",
            auto_batch_size: Union[bool, int, AutoBatchSize] = True,
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
        self.type_map = self.input_param['type_map']
        self.dp = ModelWrapper(get_model(self.input_param, None).to(DEVICE))
        self.dp.load_state_dict(state_dict)
        self.rcut = self.dp.model['Default'].descriptor.get_rcut()
        self.sec = np.cumsum(self.dp.model['Default'].descriptor.get_sel())
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

    def eval(
            self,
            coords: Union[np.ndarray, torch.Tensor],
            cells: Optional[Union[np.ndarray, torch.Tensor]],
            atom_types: Union[np.ndarray, torch.Tensor, List[int]],
            atomic: bool = False,
    ):
        raise NotImplementedError


class DeepPot(DeepEval):
    def __init__(
        self,
        model_file: "Path",
        auto_batch_size: Union[bool, int, AutoBatchSize] = True,
        neighbor_list=None,
    ):
        if neighbor_list is not None:
            raise NotImplementedError
        super(DeepPot, self).__init__(
            model_file,
            auto_batch_size=auto_batch_size,
        )

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        atom_types: List[int],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        efield: Optional[np.ndarray] = None,
        mixed_type: bool = False,
    ):
        if fparam is not None or aparam is not None or efield is not None:
            raise NotImplementedError
        # convert all of the input to numpy array
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        natoms, numb_test = self._get_natoms_and_nframes(coords, atom_types, len(atom_types.shape) > 1)
        return self._eval_func(self._eval_model, numb_test, natoms)(coords, cells, atom_types, atomic)

    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
        """Wrapper method with auto batch size.

        Parameters
        ----------
        inner_func : Callable
            the method to be wrapped
        numb_test : int
            number of tests
        natoms : int
            number of atoms

        Returns
        -------
        Callable
            the wrapper
        """
        if self.auto_batch_size is not None:

            def eval_func(*args, **kwargs):
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: Union[List[int], np.ndarray],
        mixed_type: bool = False,
    ) -> Tuple[int, int]:
        if mixed_type:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
    ):
        model = self.dp.to(DEVICE)
        return eval(model, coords, cells, atom_types, atomic=atomic)

def eval(
        model,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: np.ndarray,
        atomic: bool = False,
        denoise: bool = False,
    ):
    # cast input to numpy array
    coords = np.array(coords)
    if cells is not None:
        cells = np.array(cells)
    atom_types = np.array(atom_types, dtype=np.int32)

    energy_out = None
    atomic_energy_out = None
    force_out = None
    virial_out = None
    atomic_virial_out = None
    updated_coord_out = None
    logits_out = None

    nframes = coords.shape[0]
    if len(atom_types.shape) == 1:
        natoms = len(atom_types)
        atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
    else:
        natoms = len(atom_types[0])

    coord_input = torch.tensor(coords.reshape([-1, natoms, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
    type_input = torch.tensor(atom_types, dtype=torch.long).to(DEVICE)
    if cells is not None:
        box_input = torch.tensor(cells.reshape([-1, 3, 3]), dtype=GLOBAL_PT_FLOAT_PRECISION).to(DEVICE)
    else:
        box_input = None

    batch_output = model(coord_input, type_input, box=box_input, do_atomic_virial=atomic)
    if isinstance(batch_output, tuple):
        batch_output = batch_output[0]
    if 'energy' in batch_output:
        energy_out = batch_output['energy'].detach().cpu().numpy()
    if 'atom_energy' in batch_output:
        atomic_energy_out = batch_output['atom_energy'].detach().cpu().numpy()
    if 'force' in batch_output:
        force_out = batch_output['force'].detach().cpu().numpy()
    if 'virial' in batch_output:
        virial_out = batch_output['virial'].detach().cpu().numpy()
    if 'atomic_virial' in batch_output:
        atomic_virial_out = batch_output['atomic_virial'].detach().cpu().numpy()
    if 'updated_coord' in batch_output:
        updated_coord_out = batch_output['updated_coord'].detach().cpu().numpy()
    if 'logits' in batch_output:
        logits_out = batch_output['logits'].detach().cpu().numpy()

    if denoise:
        return updated_coord_out, logits_out
    else:    
        if not atomic:
            return energy_out, force_out, virial_out
        else:
            return energy_out, force_out, virial_out, atomic_energy_out, atomic_virial_out
