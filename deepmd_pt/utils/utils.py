import torch
import torch.nn.functional as F
from typing import List, Callable, Any, Dict, Optional


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class ActivationFn(torch.nn.Module):
    def __init__(self, activation: Optional[str]):
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the tensor after applying activation function corresponding to `activation` """
        # See jit supported types: https://pytorch.org/docs/stable/jit_language_reference.html#supported-type

        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "linear":
            return x
        else:
            raise RuntimeError("activation-fn not supported")
