from .model import BaseModel
from .ener import EnergyModel

def get_model(model_params, sampled=None):
    return EnergyModel(
      descriptor=model_params['descriptor'],
      fitting_net=model_params.get('fitting_net', None),
      type_map=model_params['type_map'],
      type_embedding=model_params.get('type_embedding', None),
      resuming=model_params.get("resuming", False),
      stat_file_dir=model_params.get("stat_file_dir", None),
      stat_file_path=model_params.get("stat_file_path", None),
      sampled=sampled,
    )
