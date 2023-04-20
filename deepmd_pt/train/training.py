import logging
import os
import torch
import time

from typing import Any, Dict
from deepmd_pt.utils import dp_random
from deepmd_pt.utils.env import DEVICE, JIT
from deepmd_pt.optimizer.KFWrapper import KFOptimizerWrapper
from deepmd_pt.optimizer.LKF import LKFOptimizer
from deepmd_pt.utils.learning_rate import LearningRateExp
from deepmd_pt.loss.ener import EnergyStdLoss
from deepmd_pt.model.model import EnergyModelSeA, EnergyModelDPA1
from deepmd_pt.train.wrapper import ModelWrapper
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb as wb

if torch.__version__.startswith("2"):
    import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Trainer(object):

    def __init__(self, config: Dict[str, Any], training_data, sampled, validation_data=None, resume_from=None):
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        model_params = config['model']
        training_params = config['training']

        # Iteration config
        self.num_steps = training_params['numb_steps']
        self.disp_file = training_params.get('disp_file', 'lcurve.out')
        self.disp_freq = training_params.get('disp_freq', 1000)
        self.save_ckpt = training_params.get('save_ckpt', 'model.pt')
        self.save_freq = training_params.get('save_freq', 1000)
        self.opt_type = training_params.get('opt_type', 'Adam')
        self.wandb_config = training_params.get('wandb_config', {})
        self.wandb_enabled = self.wandb_config.get('wandb_enabled', False)
        if self.wandb_enabled:
            entity = self.wandb_config.get('entity', None)
            assert entity is not None, "The parameter 'entity' of wandb must be specified."

            project = self.wandb_config.get('project', '')
            job_name = self.wandb_config.get('job_name', '')
            wb.init(project=project, entity=entity, config=training_params,
                    name=job_name, settings=wb.Settings(start_method="fork"))

        # Data + Model
        dp_random.seed(training_params['seed'])
        self.training_data = training_data
        self.validation_data = validation_data
        if training_params.get("validation_data", None) is not None:
            self.valid_numb_batch = training_params["validation_data"].get("numb_btch", 1)
        else:
            self.valid_numb_batch = 1
        if model_params['descriptor']['type'] == 'se_e2_a':
            self.model = EnergyModelSeA(model_params, sampled).to(DEVICE)
        elif model_params['descriptor']['type'] == 'se_atten':
            self.model = EnergyModelDPA1(model_params, sampled).to(DEVICE)
        else:
            raise NotImplementedError

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

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model, self.loss)
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)

        # Initialize DDP
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is not None:
            local_rank = int(local_rank)
            assert dist.is_nccl_available()
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        if (resume_from is not None) and (self.rank == 0):
            state_dict = torch.load(resume_from)
            self.wrapper.load_state_dict(state_dict)
            logging.info(f"Resuming from {resume_from}.")

        if dist.is_initialized():
            # DDP will guarantee the model parameters are identical across all processes
            self.wrapper = DDP(self.wrapper,
                               device_ids=[local_rank],
                               output_device=local_rank)

        if self.opt_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.wrapper.parameters(), lr=self.lr_exp.start_lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: self.lr_exp.value(step) / self.lr_exp.start_lr
            )
        elif self.opt_type == 'LKF':
            self.optimizer = LKFOptimizer(self.wrapper.parameters(), 0.98, 0.99870, 10240)
            self.wrapper.inference_only = True
        else:
            raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

        self.multi_task_mode = False
        self.task_keys = ['Default']

    def run(self):
        fout = open(self.disp_file, mode='a', buffering=1) if self.rank == 0 else None  # line buffered
        logging.info('Start to train %d steps.', self.num_steps)
        if dist.is_initialized():
            logging.info(f'Rank: {dist.get_rank()}/{dist.get_world_size()}')

        def step(_step_id, task_key="Default"):
            cur_lr = self.lr_exp.value(_step_id)
            self.optimizer.zero_grad()
            input_dict, label_dict = self.get_data(is_train=True)

            if self.opt_type == 'Adam':
                model_pred, loss, more_loss = self.wrapper(**input_dict,
                                                           cur_lr=cur_lr, label=label_dict, task_key=task_key)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            elif self.opt_type == 'LKF':
                KFOptWrapper = KFOptimizerWrapper(self.wrapper, self.optimizer, 24, 6, False)
                _ = KFOptWrapper.update_energy(input_dict, label_dict['energy'])
                p_energy, p_force = KFOptWrapper.update_force(input_dict, label_dict['force'])
                # [coord, atype, natoms, mapping, shift, selected, box]
                model_pred = {'energy': p_energy,
                              'force': p_force}
                loss, more_loss = self.wrapper.loss[task_key](model_pred, label_dict, input_dict['natoms'], learning_rate=cur_lr)
            else:
                raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

            # Log and persist
            if _step_id % self.disp_freq == 0:
                # training
                train_time = time.time() - self.t0
                msg = f'step={_step_id}, lr={cur_lr:.4f}, loss={loss:.4f}'
                record = f'step={_step_id}, lr={cur_lr}, loss={loss}'
                rmse_val = {item: more_loss[item] for item in more_loss if 'rmse' in item}
                for item in ['rmse_e', 'rmse_f', 'rmse_v']:
                    if item in rmse_val:
                        msg += f', {item}_train={rmse_val[item]:.4f}'
                        record += f', {item}_train={rmse_val[item]}'
                        self.wandb_log({item: rmse_val[item]}, _step_id, '_train')
                        rmse_val.pop(item)
                for rest_item in sorted(list(rmse_val.keys())):
                    if rest_item in rmse_val:
                        msg += f', {rest_item}={rmse_val[rest_item]:.4f}'
                        record += f', {rest_item}={rmse_val[rest_item]}'
                        rmse_val.pop(rest_item)

                # validation
                if self.validation_data is not None:
                    single_results = {}
                    sum_natoms = 0
                    for ii in range(self.valid_numb_batch):
                        self.optimizer.zero_grad()
                        input_dict, label_dict = self.get_data(is_train=False)
                        model_pred, loss, more_loss = self.wrapper(**input_dict,
                                                                   cur_lr=cur_lr, label=label_dict, task_key=task_key)
                        natoms = input_dict['natoms'][0, 0]
                        sum_natoms += natoms
                        for k, v in more_loss.items():
                            if 'rmse' in k:
                                single_results[k] = single_results.get(k, 0.0) + v * natoms
                    valid_results = {
                        k: v / sum_natoms
                        for k, v in single_results.items()
                    }
                    for item in sorted(list(valid_results.keys())):
                        msg += f', {item}_valid={valid_results[item]:.4f}'
                        record += f', {item}_valid={valid_results[item]}'
                        self.wandb_log({item: valid_results[item]}, _step_id, '_valid')

                msg += f', speed={train_time:.2f} s/{self.disp_freq} batches'
                record += f', speed={train_time} s/{self.disp_freq} batches\n'
                logging.info(msg)
                self.wandb_log({'lr': cur_lr}, step_id)

                if fout:
                    fout.write(record)
                self.t0 = time.time()

            if ((_step_id % self.save_freq == 0 and _step_id != 0) \
                or _step_id == self.num_steps - 1) \
                    and (self.rank == 0 or dist.get_rank() == 0):
                # Handle the case if rank 0 aborted and re-assigned
                logging.info(f"Saving model to {self.save_ckpt}")
                module = self.wrapper.module if dist.is_initialized() else self.wrapper
                torch.save(module.state_dict(), self.save_ckpt)

        self.t0 = time.time()
        with logging_redirect_tqdm():
            for step_id in tqdm(range(self.num_steps), disable=None):  # set to None to disable on non-TTY
                step(step_id)

        if self.rank == 0 or dist.get_rank() == 0:  # Handle the case if rank 0 aborted and re-assigned
            if JIT:
                self.wrapper.save("torchscript_model.pt")
        if fout:
            fout.close()

    def get_data(self, is_train=True):
        if is_train:
            batch_data = self.training_data.get_training_batch()
        else:
            batch_data = self.validation_data.get_training_batch()
        input_dict = {}
        for item in ['coord', 'atype', 'natoms', 'mapping', 'shift', 'selected', 'selected_type', 'box']:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ['energy', 'force', 'virial']:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        return input_dict, label_dict

    def wandb_log(self, data: dict, step, type_suffix=''):
        if not self.wandb_enabled:
            return
        for k, v in data.items():
            wb.log({k + type_suffix: v}, step=step)
