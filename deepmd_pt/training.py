import logging
import os
import torch
import time

from typing import Any, Dict
from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from env import DEVICE, JIT
if torch.__version__.startswith("2"):
    import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Trainer(object):

    def __init__(self, config: Dict[str, Any], dataloader ,sampled ,resume_from = None):
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




        # Data + Model
        my_random.seed(training_params['seed'])
        self.training_data = dataloader
        self.model = EnergyModel(model_params, sampled).to(DEVICE)
        if JIT:
            self.model = torch.jit.script(self.model)

        # Initialize DDP
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is not None:
            local_rank=int(local_rank)

            assert dist.is_nccl_available()
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             output_device=local_rank)
            self.rank = dist.get_rank()
        # TODO: load state dict
        # Learning rate
        lr_params = config.pop('learning_rate')
        assert lr_params.pop('type', 'exp'), 'Only learning rate `exp` is supported!'
        lr_params['stop_steps'] = self.num_steps
        self.lr_exp = LearningRateExp(**lr_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_exp.start_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: self.lr_exp.value(step)/self.lr_exp.start_lr)

        # Loss
        loss_params = config.pop('loss')
        assert loss_params.pop('type', 'ener'), 'Only loss `ener` is supported!'
        loss_params['starter_learning_rate'] = lr_params['start_lr']
        self.loss = EnergyStdLoss(**loss_params)

        if resume_from is not None:
            state_dict = torch.load(resume_from)
            self.model.load_state_dict(state_dict)
            logging.info(f"Resuming from {resume_from}.")

    def run(self):
        fout = open(self.disp_file, 'w')
        logging.info('Start to train %d steps.', self.num_steps)

        def step(step_id):
            bdata = self.training_data.get_training_batch()
            self.optimizer.zero_grad()
            cur_lr = self.lr_exp.value(step_id)
            l_energy = bdata['energy']
            l_force = bdata['force']

            # Compute prediction error
            coord, atype, natoms = bdata['coord'], bdata['atype'], bdata['natoms']
            mapping, shift, selected, box = bdata['mapping'], bdata['shift'], bdata['selected'], bdata['box']
            p_energy, p_force, p_virial = self.model(coord, atype, natoms, mapping, shift, selected, box)
            l_force = l_force.view(-1, bdata['natoms'][0,0], 3)
            assert l_energy.shape == p_energy.shape
            assert l_force.shape == p_force.shape
            loss, rmse_e, rmse_f = self.loss(cur_lr, natoms, p_energy, p_force, l_energy, l_force)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log and persist
            if step_id % self.disp_freq == 0:
                train_time = time.time() - self.t0
                logging.info(f'step={step_id}, lr={cur_lr:.4f}, loss={loss:.4f}, rmse_e={rmse_e:.4f}, rmse_f={rmse_f:.4f}, speed={train_time:.2f} s/{self.disp_freq} batches')
                record = f'step={step_id}, lr={cur_lr}, loss={loss}, rmse_e={rmse_e}, rmse_f={rmse_f}, speed={train_time} s/{self.disp_freq} batches\n'
                fout.write(record)
                fout.flush()
                self.t0 = time.time()
            if step_id > 0:
                if step_id % self.save_freq == 0:
                    torch.save(self.model.state_dict(), self.save_ckpt)

        self.t0 = time.time()
        with logging_redirect_tqdm():
            for step_id in tqdm(range(self.num_steps), disable=None): # set to None to disable on non-TTY
                step(step_id)

        if self.rank == 0:
            module = self.model
            if isinstance(module, DDP):
                module = module.module
            if JIT:
                module.save("torchscript_model.pt")
            torch.save(module.state_dict(), self.save_ckpt)
        fout.close()
        logging.info('Saving model after all steps...')