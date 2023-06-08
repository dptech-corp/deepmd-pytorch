import logging
import torch

def load_model_params(ckpt, finetune_model, config):
    """Load model_params according to the pretrained one.

    Args:
    - ckpt & finetune_model: origin model.
    - config: Read from json file.
    """
    model_params = config['model']
    if finetune_model is not None:
        state_dict = torch.load(finetune_model)
        last_model_params = state_dict['_extra_state']['model_params']
        old_type_map, new_type_map = last_model_params['type_map'], model_params['type_map']
        assert set(new_type_map).issubset(old_type_map), "Only support for smaller type map when finetuning or resuming."
        last_model_params['data_bias_nsample'] = model_params.get("data_bias_nsample", 10)
        model_params = last_model_params
        logging.info("Change the model configurations according to the pretrained one...")
        model_params["new_type_map"] = new_type_map
    model_params["resuming"] = (finetune_model is not None) or (ckpt is not None)
    config['model'] = model_params
    return config