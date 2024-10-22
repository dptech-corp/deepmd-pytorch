import logging
from typing import Optional
import numpy as np
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import MaskLMHead
from deepmd_pt.model.task import TaskBaseMethod


class TypePredictNet(TaskBaseMethod):

    def __init__(self,
                 feature_dim,
                 ntypes,
                 activation_function="gelu",
                 **kwargs):
        """Construct a type predict net.

        Args:
        - feature_dim: Input dm.
        - ntypes: Numer of types to predict.
        - activation_function: Activate function.
        """
        super(TypePredictNet, self).__init__()
        self.feature_dim = feature_dim
        self.ntypes = ntypes
        self.lm_head = MaskLMHead(
                embed_dim=self.feature_dim,
                output_dim=ntypes,
                activation_fn=activation_function,
                weight=None,
            )

    def forward(self, features, masked_tokens: Optional[torch.Tensor]=None):
        """Calculate the predicted logits.
        Args:
        - features: Input features with shape [nframes, nloc, feature_dim].
        - masked_tokens: Input masked tokens with shape [nframes, nloc].

        Returns:
        - logits: Predicted probs with shape [nframes, nloc, ntypes].
        """
        # [nframes, nloc, ntypes]
        logits = self.lm_head(features, masked_tokens=masked_tokens)
        return logits
