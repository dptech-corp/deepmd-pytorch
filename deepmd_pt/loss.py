import torch

from deepmd_pt.env import GLOBAL_PT_FLOAT_PRECISION


class EnergyStdLoss(torch.nn.Module):

    def __init__(self, starter_learning_rate,
        start_pref_e=0.02, limit_pref_e=1., start_pref_f=1000., limit_pref_f=1., start_pref_v=0, limit_pref_v=0, **kwargs):
        '''Construct a layer to compute loss on energy and force.'''
        super(EnergyStdLoss, self).__init__()
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v

    def forward(self, learning_rate, natoms, p_energy, p_force, p_virial, l_energy, l_force, l_virial):
        '''Return loss on loss and force.

        Args:
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - p_virial: Predicted virial of all atoms.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.
        - l_virial: Actual virial of all atoms.

        Returns:
        - loss: Loss to minimize.
        '''
        coef = learning_rate / self.starter_learning_rate
        # energy
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        l2_ener_loss = torch.mean(torch.square(p_energy - l_energy))
        atom_norm_ener = 1./ natoms[0, 0]
        energy_loss = atom_norm_ener * (pref_e * l2_ener_loss)
        rmse_e = l2_ener_loss.sqrt() * atom_norm_ener
        # force
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        diff_f = l_force- p_force
        l2_force_loss = torch.mean(torch.square(diff_f))
        force_loss = (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
        rmse_f = l2_force_loss.sqrt()
        # virial
        if l_virial.abs().sum() > 1e-8:
            pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
            diff_v = l_virial - p_virial
            l2_virial_loss = torch.mean(torch.square(diff_v))* atom_norm_ener
            virial_loss = (pref_v * l2_virial_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            rmse_v = l2_virial_loss.sqrt()
        else:
            virial_loss, rmse_v = torch.tensor(0), torch.tensor(0)
        return energy_loss + force_loss + virial_loss, rmse_e.detach(), rmse_f.detach(), rmse_v.detach()
