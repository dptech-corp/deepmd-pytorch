import torch

from deepmd_pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.loss.loss import TaskLoss


class EnergyStdLoss(TaskLoss):

    def __init__(self,
                 starter_learning_rate,
                 start_pref_e=0.02,
                 limit_pref_e=1.,
                 start_pref_f=1000.,
                 limit_pref_f=1.,
                 start_pref_v=0.0,
                 limit_pref_v=0.0,
                 **kwargs):
        """Construct a layer to compute loss on energy, force and virial."""
        super(EnergyStdLoss, self).__init__()
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v

    def forward(self, model_pred, label, natoms, learning_rate):
        """Return loss on loss and force.

        Args:
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.

        Returns:
        - loss: Loss to minimize.
        """
        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        loss = 0.
        more_loss = {}
        atom_norm = 1. / natoms[0, 0]
        if 'energy' in model_pred and 'energy' in label:
            l2_ener_loss = torch.mean(torch.square(model_pred['energy'] - label['energy']))
            more_loss['l2_ener_loss'] = l2_ener_loss.detach()
            loss += atom_norm * (pref_e * l2_ener_loss)
            rmse_e = l2_ener_loss.sqrt() * atom_norm
            more_loss['rmse_e'] = rmse_e.detach()

        if 'force' in model_pred and 'force' in label:
            diff_f = label['force'] - model_pred['force']
            l2_force_loss = torch.mean(torch.square(diff_f))
            more_loss['l2_force_loss'] = l2_force_loss.detach()
            loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            rmse_f = l2_force_loss.sqrt()
            more_loss['rmse_f'] = rmse_f.detach()

        if 'virial' in model_pred and 'virial' in label:
            diff_v = label['virial'] - model_pred['virial'].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            more_loss['l2_virial_loss'] = l2_virial_loss.detach()
            loss += atom_norm * (pref_v * l2_virial_loss)
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss['rmse_v'] = rmse_v.detach()

        return loss, more_loss
