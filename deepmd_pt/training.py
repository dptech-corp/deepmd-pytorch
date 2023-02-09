import logging
import torch

from typing import Any, Dict

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel
from env import DEVICE


class Trainer(object):

    def __init__(self, config: Dict[str, Any]):
        '''Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        '''
        model_params = config['model']
        training_params = config['training']

        # Iteration config
        self.num_steps = training_params['numb_steps']
        self.disp_file = training_params.get('disp_file', 'lcurve.out')
        self.disp_freq = training_params.get('disp_freq', 1000)
        self.save_ckpt = training_params.get('save_ckpt', 'model.ckpt')
        self.save_freq = training_params.get('save_freq', 1000)
        self.device = 'cpu'

        # Data + Model
        my_random.seed(training_params['seed'])
        dataset_params = training_params.pop('training_data')
        self.training_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map']
        )
        self.model = EnergyModel(model_params, self.training_data).to(DEVICE)

        # Learning rate
        lr_params = config.pop('learning_rate')
        assert lr_params.pop('type', 'exp'), 'Only learning rate `exp` is supported!'
        lr_params['stop_steps'] = self.num_steps
        self.lr_exp = LearningRateExp(**lr_params)

        # Loss
        loss_params = config.pop('loss')
        assert loss_params.pop('type', 'ener'), 'Only loss `ener` is supported!'
        loss_params['starter_learning_rate'] = lr_params['start_lr']
        self.loss = EnergyStdLoss(**loss_params)

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_exp.start_lr)
        fout = open(self.disp_file, 'w')
        logging.info('Start to train %d steps.', self.num_steps)
        
        def step(step_id):
            bdata = self.training_data.get_batch()
            optimizer.zero_grad()
            cur_lr = self.lr_exp.value(step_id)

            # Prepare inputs
            coord = torch.from_numpy(bdata['coord'])
            atype = torch.from_numpy(bdata['type'])
            natoms = bdata['natoms_vec']
            box = bdata['box']
            l_energy = torch.from_numpy(bdata['energy']).to(DEVICE)
            l_force = torch.from_numpy(bdata['force']).to(DEVICE)

            # Compute prediction error
            coord.requires_grad_(True)
            p_energy, p_force = self.model(coord.to(DEVICE), atype.to(DEVICE), natoms, box)
            loss, rmse_e, rmse_f = self.loss(cur_lr, natoms, p_energy, p_force, l_energy, l_force)
            loss_val = loss.cpu().detach().numpy().tolist()
            logging.info('step=%d, lr=%f, loss=%f', step_id, cur_lr, loss_val)

            # Backpropagation
            loss.backward()
            for g in optimizer.param_groups:
                g['lr'] = cur_lr
            optimizer.step()

            # Log and persist
            if step_id % self.disp_freq == 0:
                rmse_e_val = rmse_e.cpu().detach().numpy().tolist()
                rmse_f_val = rmse_f.cpu().detach().numpy().tolist()
                record = 'step=%d, rmse_e=%f, rmse_f=%f\n' % (step_id, rmse_e_val, rmse_f_val)
                fout.write(record)
                fout.flush()
            if step_id > 0:
                if step_id % self.save_freq == 0:
                    torch.save(self.model.state_dict(), self.save_ckpt)

        for step_id in range(self.num_steps):
            step(step_id)

        fout.close()
        logging.info('Saving model after all steps...')
        torch.save(self.model.state_dict(), self.save_ckpt)
