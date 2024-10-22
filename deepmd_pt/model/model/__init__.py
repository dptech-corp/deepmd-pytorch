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
from .hybrid_ener import EnergyModelHybrid
from .hybrid_force import ForceModelHybrid


def get_model(model_params, sampled=None):
    if model_params.get("fitting_net", None) is not None:
        if model_params.get("backbone", None) is None:
            if model_params["descriptor"]["type"] == "se_e2_a":
                return EnergyModelSeA(model_params, sampled)
            elif model_params["descriptor"]["type"] == "se_atten":
                if model_params["fitting_net"].get("type", "ener") == "ener":
                    return EnergyModelDPA1(model_params, sampled)
                elif "direct" in model_params["fitting_net"].get("type", "ener"):
                    return ForceModelDPA1(model_params, sampled)
            elif model_params["descriptor"]["type"] == "se_uni":
                if model_params["fitting_net"].get("type", "ener") == "ener":
                    return EnergyModelDPAUni(model_params, sampled)
                elif "direct" in model_params["fitting_net"].get("type", "ener"):
                    return ForceModelDPAUni(model_params, sampled)
            elif model_params["descriptor"]["type"] == "hybrid":
                if model_params["fitting_net"].get("type", "ener") == "ener":
                    return EnergyModelHybrid(model_params, sampled)
                elif "direct" in model_params["fitting_net"].get("type", "ener"):
                    return ForceModelHybrid(model_params, sampled)
            else:
                raise NotImplementedError
        else:
            if model_params["fitting_net"].get("type", "ener") == "ener":
                return EnergyModelDPA2(model_params, sampled)
            elif "direct" in model_params["fitting_net"].get("type", "ener"):
                return ForceModelDPA2(model_params, sampled)
    else:
        if model_params.get("backbone", None) is None:
            return DenoiseModelDPA1(model_params, sampled)
        else:
            return DenoiseModelDPA2(model_params, sampled)
