import logging
import torch

def get_model_params(ckpt, finetune, config):
    """Update model_params.

    Args:
    - origin_model: ckpt.
    - config: Read from json file.
    """
    model_params = config['model']
    model_params["resuming"] = False
    if ((ckpt is not None) or (finetune is not None)):
        origin_model = finetune if finetune is not None else ckpt
        state_dict = torch.load(origin_model)
        if 'other_info' in state_dict:
            origin_config = state_dict.pop('other_info', {})
            last_model_params = origin_config['model_params']
            old_type_map, new_type_map = last_model_params['type_map'], model_params['type_map']
            assert set(new_type_map).issubset(old_type_map), "Only support for smaller type map when finetuning or resuming."
            model_params = last_model_params
            model_params["new_type_map"] = new_type_map
        model_params["resuming"] = True
    config['model'] = model_params
    return config