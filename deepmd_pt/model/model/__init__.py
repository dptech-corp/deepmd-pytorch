from .model import BaseModel
from .se_a_ener import EnergyModelSeA
from .dpa1_ener import EnergyModelDPA1
from .dpa2_denoise import DenoiseModelDPA2
from .dpa2_ener import EnergyModelDPA2
from .dpa1_denoise import DenoiseModelDPA1
from .dpa1_force import ForceModelDPA1
from .dpa2_force import ForceModelDPA2
from .dpau_ener import EnergyModelDPAUni
from .dpau_force import ForceModelDPAUni
from .dpa2_lcc_force import ForceModelDPA2Lcc
from .hybrid_ener import EnergyModelHybrid
from .hybrid_force import ForceModelHybrid
from .ener import EnergyModel


def get_model(model_params, sampled=None):
    return EnergyModel(descriptor=model_params['descriptor'],
                       fitting_net=model_params.get('fitting_net', None),
                       type_map=model_params['type_map'],
                       type_embedding=model_params.get('type_embedding', None),
                       resuming=model_params.get("resuming", False),
                       stat_file_dir=model_params.get("stat_file_dir", None),
                       stat_file_path=model_params.get("stat_file_path", None),
                       sampled=sampled,
                       )

# backup
# def get_model(model_params, sampled=None):
#     if model_params.get("fitting_net", None) is not None:
#         if model_params.get("backbone", None) is None:
#             if model_params["descriptor"]["type"] == "se_e2_a":
#                 return EnergyModelSeA(model_params, sampled)
#             elif model_params["descriptor"]["type"] == "se_atten":
#                 if model_params["fitting_net"].get("type", "ener") == "ener":
#                     return EnergyModelDPA1(model_params, sampled)
#                 elif "direct" in model_params["fitting_net"].get("type", "ener"):
#                     return ForceModelDPA1(model_params, sampled)
#             elif model_params["descriptor"]["type"] == "se_uni":
#                 if model_params["fitting_net"].get("type", "ener") == "ener":
#                     return EnergyModelDPAUni(model_params, sampled)
#                 elif "direct" in model_params["fitting_net"].get("type", "ener"):
#                     return ForceModelDPAUni(model_params, sampled)
#             elif model_params["descriptor"]["type"] == "hybrid":
#                 if model_params["fitting_net"].get("type", "ener") == "ener":
#                     return EnergyModelHybrid(model_params, sampled)
#                 elif model_params["fitting_net"].get("type", "ener") in ["atten_vec_lcc", "direct_force",
#                                                                          "direct_force_ener"]:
#                     return ForceModelHybrid(model_params, sampled)
#             elif model_params["descriptor"]["type"] == "gaussian_lcc":
#                 if model_params["fitting_net"].get("type", "ener") == "atten_vec_lcc":
#                     return ForceModelDPA2Lcc(model_params, sampled)
#             else:
#                 raise NotImplementedError
#         else:
#             if model_params["fitting_net"].get("type", "ener") == "ener":
#                 return EnergyModelDPA2(model_params, sampled)
#             elif "direct" in model_params["fitting_net"].get("type", "ener"):
#                 return ForceModelDPA2(model_params, sampled)
#     else:
#         if model_params.get("backbone", None) is None:
#             return DenoiseModelDPA1(model_params, sampled)
#         else:
#             return DenoiseModelDPA2(model_params, sampled)
