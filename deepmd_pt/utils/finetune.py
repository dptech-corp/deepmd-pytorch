import logging
import torch

def change_finetune_model_params(ckpt, finetune_model, model_config, multi_task=False):
    """Load model_params according to the pretrained one.

    Args:
    - ckpt & finetune_model: origin model.
    - config: Read from json file.
    """
    if multi_task:
        #TODO
        print('finetune mode need modification for multitask mode!')
    if finetune_model is not None:
        state_dict = torch.load(finetune_model)
        last_model_params = state_dict['_extra_state']['model_params']
        old_type_map, new_type_map = last_model_params['type_map'], model_config['type_map']
        assert set(new_type_map).issubset(old_type_map), "Only support for smaller type map when finetuning or resuming."
        model_config = last_model_params
        logging.info("Change the model configurations according to the pretrained one...")
        model_config["new_type_map"] = new_type_map
    return model_config
