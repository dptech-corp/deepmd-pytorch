import logging
import os
import torch
from typing import Dict, Optional, Union

if torch.__version__.startswith("2"):
    import torch._dynamo


class ModelWrapper(torch.nn.Module):

    def __init__(self,
                 model: Union[torch.nn.Module, Dict],
                 loss: Union[torch.nn.Module, Dict] = None):
        """Construct a DeePMD model wrapper.

        Args:
        - config: The Dict-like configuration with training options.
        """
        super(ModelWrapper, self).__init__()
        self.multi_task = False
        self.model = torch.nn.ModuleDict()
        # Model
        if isinstance(model, torch.nn.Module):
            self.model["Default"] = model
        elif isinstance(model, dict):
            self.multi_task = True
            for task_key in model:
                assert isinstance(model[task_key], torch.nn.Module), \
                    f"{task_key} in model_dict is not a torch.nn.Module!"
                self.model[task_key] = model[task_key]
        # Loss
        self.loss = None
        if loss is not None:
            self.loss = torch.nn.ModuleDict()
            if isinstance(loss, torch.nn.Module):
                self.loss["Default"] = loss
            elif isinstance(loss, dict):
                for task_key in loss:
                    assert isinstance(loss[task_key], torch.nn.Module), \
                        f"{task_key} in loss_dict is not a torch.nn.Module!"
                    self.loss[task_key] = loss[task_key]
        self.inference_only = self.loss is None

    def shared_params(self):  # TODO ZD:multitask share params
        pass

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type, selected_loc: Optional[torch.Tensor]=None, box: Optional[torch.Tensor]=None,
                cur_lr: Optional[torch.Tensor]=None, label: Optional[torch.Tensor]=None, task_key: Optional[torch.Tensor]=None, inference_only=False):
        if not self.multi_task:
            task_key = "Default"
        else:
            assert task_key is not None, \
                f"Multitask model must specify the inference task! Supported tasks are {list(self.model.keys())}."
        model_pred = self.model[task_key](coord, atype, natoms, mapping, shift, selected, selected_type,
                                          selected_loc=selected_loc, box=box)
        if not self.inference_only and not inference_only:
            loss, more_loss = self.loss[task_key](model_pred, label, natoms=natoms, learning_rate=cur_lr)
            return model_pred, loss, more_loss
        else:
            return model_pred, None, None
