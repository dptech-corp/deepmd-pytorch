import numpy as np
import torch
import logging
import os
from deepmd_pt.utils import env
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from IPython import embed


class BaseModel(torch.nn.Module):

    def __init__(self):
        """Construct a basic model for different tasks.
        """
        super(BaseModel, self).__init__()

    def forward(self, coord, atype, natoms, mapping, shift, selected, box, **kwargs):
        """Model output.
        """
        raise NotImplementedError

    def compute_or_load_stat(self, model_params, fitting_param, ntypes, sampled=None, set_zero_energy_bias=False):
        resuming = model_params.get("resuming", False)
        if not resuming:
            if sampled is not None:  # compute stat
                for sys in sampled:
                    for key in sys:
                        sys[key] = sys[key].to(env.DEVICE)
                sumr, suma, sumn, sumr2, suma2 = self.descriptor.compute_input_stats(sampled)

                energy = [item['energy'] for item in sampled]
                mixed_type = 'real_natoms_vec' in sampled[0]
                if mixed_type:
                    input_natoms = [item['real_natoms_vec'] for item in sampled]
                else:
                    input_natoms = [item['natoms'] for item in sampled]
                if not set_zero_energy_bias:
                    tmp = compute_output_stats(energy, input_natoms)
                    fitting_param['bias_atom_e'] = tmp[:, 0]
                else:
                    fitting_param['bias_atom_e'] = [0.0] * ntypes
                if model_params.get("stat_file_path", None) is not None:
                    logging.info(f'Saving stat file to {model_params["stat_file_path"]}')
                    if not os.path.exists(model_params["stat_file_dir"]):
                        os.mkdir(model_params["stat_file_dir"])
                    if sumr is not None:
                        np.savez_compressed(model_params["stat_file_path"],
                                            sumr=sumr, suma=suma, sumn=sumn, sumr2=sumr2, suma2=suma2,
                                            bias_atom_e=fitting_param['bias_atom_e'], type_map=model_params['type_map'])
                    else:
                        np.savez_compressed(model_params["stat_file_path"],
                                            bias_atom_e=fitting_param['bias_atom_e'], type_map=model_params['type_map'])
            else:  # load stat
                try:
                    stats = np.load(model_params["stat_file_path"])
                    logging.info(f'Loading stat file from {model_params["stat_file_path"]}')
                except FileNotFoundError as e:
                    logging.warning(f'Stat data not exist! '
                                    f'The statistics weights in descriptor and fitting will be set to default!')
                    fitting_param['bias_atom_e'] = [0.0] * ntypes
                    return
                stat_type_map = list(stats["type_map"])
                target_type_map = model_params['type_map']
                missing_type = [i for i in target_type_map if i not in stat_type_map]
                assert not missing_type, \
                    f"These type are not in stat file: {missing_type}! Please change the stat file path!"
                idx_map = [stat_type_map.index(i) for i in target_type_map]
                if "sumr" in stats:
                    sumr, suma, sumn, sumr2, suma2 = stats["sumr"][idx_map], stats["suma"][idx_map], \
                                                     stats["sumn"][idx_map], stats["sumr2"][idx_map], \
                                                     stats["suma2"][idx_map]
                else:
                    sumr, suma, sumn, sumr2, suma2 = None, None, None, None, None
                if not set_zero_energy_bias:
                    fitting_param['bias_atom_e'] = stats["bias_atom_e"][idx_map]
                else:
                    fitting_param['bias_atom_e'] = [0.0] * ntypes
            self.descriptor.init_desc_stat(sumr, suma, sumn, sumr2, suma2)
        else:  # resuming for checkpoint; init model params from scratch
            fitting_param['bias_atom_e'] = [0.0] * ntypes
