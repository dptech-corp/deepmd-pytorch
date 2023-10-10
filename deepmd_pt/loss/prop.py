import torch
from deepmd_pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.loss import TaskLoss
from deepmd_pt.utils import env
import torch.nn.functional as F

class PropertyLoss(TaskLoss):

    def __init__(self,
                 use_l1_all: bool = False,
                 prop_type: str='extensive',
                 l2_loss: bool = False,
                 **kwargs):
        """Construct a layer to compute loss on energy, force and virial."""
        super(PropertyLoss, self).__init__()
        self.use_l1_all = use_l1_all
        self.prop_type = prop_type
        self.l2_loss = l2_loss

    def forward(self, model_pred, label, natoms, learning_rate, mae=False):
        """Return loss on loss and force.

        Args:
        - natoms: Tell atom count.
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.

        Returns:
        - loss: Loss to minimize.
        """
        loss = torch.tensor(0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more_loss = {}
        atom_norm = 1. / natoms

        import logging
        #logging.info(f"loss label after:{label['property']}")
        if self.prop_type == 'intensive':
            label['property'] = label['property']/natoms
        if not self.use_l1_all:
            l2_prop_loss = torch.mean(torch.square(model_pred['property'] - label['property']))
            more_loss['l2_prop_loss'] = l2_prop_loss.detach()
            loss += atom_norm * l2_prop_loss
            rmse_p = l2_prop_loss.sqrt() * atom_norm
            more_loss['rmse_p'] = rmse_p.detach()
        else: # use l1 and for all atoms
            #logging.info(f"loss model pred:{model_pred['property']}")
            #logging.info(f"loss label:{label['property']}")
            if self.l2_loss:
                l2_prop_loss = torch.mean(torch.square(model_pred['property'] - label['property']))
                if self.prop_type == 'intensive':
                    loss += l2_prop_loss
                else:
                    loss += atom_norm * l2_prop_loss
            else:
                l1_prop_loss = F.l1_loss(model_pred['property'].reshape(-1), label['property'].reshape(-1), reduction="sum")
                loss += l1_prop_loss 
            more_loss['mae_p'] = F.l1_loss(model_pred['property'].reshape(-1), label['property'].reshape(-1), reduction="mean").detach()
            # more_loss['log_keys'].append('rmse_e')
        if mae:
            mae_p = torch.mean(torch.abs(model_pred['property'] - label['property'])) * atom_norm
            more_loss['mae_p'] = mae_p.detach()
            mae_p_all = torch.mean(torch.abs(model_pred['property'] - label['property']))
            more_loss['mae_p_all'] = mae_p_all.detach()
        more_loss['rmse'] = torch.sqrt(loss.detach())
        return loss, more_loss