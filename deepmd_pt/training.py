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

from env import DEVICE, JIT, LOCAL_RANK
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
        self.save_ckpt = training_params.get('save_ckpt', 'model.pt')
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
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        if (resume_from is not None) and (self.rank == 0):
            state_dict = torch.load(resume_from)
            self.model.load_state_dict(state_dict)
            logging.info(f"Resuming from {resume_from}.")

        if dist.is_initialized():
            # DDP will guarantee the model parameters are identical across all processes
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             output_device=local_rank)
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

    def run(self):
        fout = open(self.disp_file, mode='w', buffering=1) if self.rank == 0 else None # line buffered

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
                if fout:
                    fout.write(record)
                self.t0 = time.time()

            if ((step_id % self.save_freq == 0 and step_id != 0) \
                or step_id == self.num_steps - 1) \
                and (self.rank == 0 or dist.get_rank() == 0 ):
                # Handle the case if rank 0 aborted and re-assigned
                logging.info(f"Saving model to {self.save_ckpt}")
                module=self.model.module if dist.is_initialized() else self.model
                torch.save(module.state_dict(), self.save_ckpt)

        self.t0 = time.time()
        with logging_redirect_tqdm():
            for step_id in tqdm(range(self.num_steps), disable=None): # set to None to disable on non-TTY
                step(step_id)

        if self.rank == 0 or dist.get_rank() == 0: # Handle the case if rank 0 aborted and re-assigned
            if JIT:
                self.model.save("torchscript_model.pt")
        if fout:
            fout.close()
