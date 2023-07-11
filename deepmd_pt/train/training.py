import logging
import os
import torch
import time
import math
from copy import deepcopy

from typing import Any, Dict
import numpy as np
from deepmd_pt.utils import dp_random
from deepmd_pt.utils.env import DEVICE, JIT, LOCAL_RANK
from deepmd_pt.optimizer import KFOptimizerWrapper, LKFOptimizer
from deepmd_pt.utils.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss, DenoiseLoss
from deepmd_pt.model.model import get_model
from deepmd_pt.train.wrapper import ModelWrapper
from deepmd_pt.utils.dataloader import BufferedIterator
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


import wandb as wb

if torch.__version__.startswith("2"):
    import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(
            self,
            config: Dict[str, Any],
            training_data,
            sampled,
            validation_data=None,
            resume_from=None,
            force_load=False,
            finetune_model=None,
            shared_links=None,
    ):
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        model_params = config["model"]
        training_params = config["training"]
        self.multi_task = "model_dict" in model_params
        self.model_keys = [key for key in model_params["model_dict"]] if self.multi_task else ["Default"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.pt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.lcurve_should_print_header = True

        # Init wandb
        self.wandb_config = training_params.get("wandb_config", {})
        self.wandb_enabled = self.wandb_config.get("wandb_enabled", False)
        if self.wandb_enabled:
            entity = self.wandb_config.get("entity", None)
            assert (
                    entity is not None
            ), "The parameter 'entity' of wandb must be specified."
            project = self.wandb_config.get("project", None)
            assert (
                    project is not None
            ), "The parameter 'project' of wandb must be specified."
            job_name = self.wandb_config.get("job_name", None)
            if job_name is None:
                name_path = os.path.abspath(".").split("/")
                job_name = name_path[-2] + "/" + name_path[-1]
            if self.rank == 0:
                wb.init(
                    project=project,
                    entity=entity,
                    config=model_params,
                    name=job_name,
                    settings=wb.Settings(start_method="fork"),
                )

        def get_opt_param(params):
            opt_type = params.get("opt_type", "Adam")
            opt_param = {'kf_blocksize': params.get("kf_blocksize", 5120),
                         'kf_start_pref_e': params.get("kf_start_pref_e", 1),
                         'kf_limit_pref_e': params.get("kf_limit_pref_e", 1),
                         'kf_start_pref_f': params.get("kf_start_pref_f", 1),
                         'kf_limit_pref_f': params.get("kf_limit_pref_f", 1)}
            return opt_type, opt_param

        def get_data_loader(_training_data, _validation_data, _training_params):
            training_dataloader = DataLoader(
                _training_data,
                sampler=torch.utils.data.RandomSampler(_training_data),
                batch_size=None,
                num_workers=8,  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
                pin_memory=True,
            )
            training_data_buffered = BufferedIterator(iter(training_dataloader))
            validation_dataloader = DataLoader(
                _validation_data,
                sampler=torch.utils.data.RandomSampler(_validation_data),
                batch_size=None,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )

            validation_data_buffered = BufferedIterator(iter(validation_dataloader))
            if _training_params.get("validation_data", None) is not None:
                valid_numb_batch = _training_params["validation_data"].get(
                    "numb_btch", 1
                )
            else:
                valid_numb_batch = 1
            return training_dataloader, training_data_buffered, \
                   validation_dataloader, validation_data_buffered, valid_numb_batch

        def get_single_model(_model_params, _sampled):
            model = get_model(deepcopy(_model_params), _sampled).to(DEVICE)
            return model

        def get_lr(lr_params):
            assert lr_params.get("type", "exp") == "exp", "Only learning rate `exp` is supported!"
            lr_params["stop_steps"] = self.num_steps - self.warmup_steps
            lr_exp = LearningRateExp(**lr_params)
            return lr_exp

        def get_loss(loss_params, start_lr):
            loss_type = loss_params.get("type", "ener")
            if loss_type == 'ener':
                loss_params["starter_learning_rate"] = start_lr
                return EnergyStdLoss(**loss_params)
            elif loss_type == 'denoise':
                loss_params['ntypes'] = len(model_params['type_map'])
                return DenoiseLoss(**loss_params)
            else:
                raise NotImplementedError

        # Optimizer
        if self.multi_task and training_params.get("optim_dict", None) is not None:
            self.optim_dict = training_params.get("optim_dict")
            missing_keys = [key for key in self.model_keys if key not in self.optim_dict]
            assert not missing_keys, f"These keys are not in optim_dict: {missing_keys}!"
            self.opt_type = {}
            self.opt_param = {}
            for model_key in self.model_keys:
                self.opt_type[model_key], self.opt_param[model_key] = get_opt_param(self.optim_dict[model_key])
        else:
            self.opt_type, self.opt_param = get_opt_param(training_params)

        # Data + Model
        dp_random.seed(training_params["seed"])
        if not self.multi_task:
            self.training_dataloader, self.training_data, \
            self.validation_dataloader, self.validation_data, self.valid_numb_batch = get_data_loader(training_data,
                                                                                                      validation_data,
                                                                                                      training_params)
            self.model = get_single_model(model_params, sampled)
        else:
            self.training_dataloader, self.training_data, \
            self.validation_dataloader, self.validation_data, \
            self.valid_numb_batch, self.model = {}, {}, {}, {}, {}, {}
            for model_key in self.model_keys:
                self.training_dataloader[model_key], self.training_data[model_key], \
                self.validation_dataloader[model_key], self.validation_data[model_key], \
                self.valid_numb_batch[model_key] = get_data_loader(training_data[model_key],
                                                                   validation_data[model_key],
                                                                   training_params['data_dict'][model_key])
                self.model[model_key] = get_single_model(model_params['model_dict'][model_key], sampled[model_key])

        # Learning rate
        self.warmup_steps = training_params.get("warmup_steps", 0)
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.)
        assert self.num_steps - self.warmup_steps > 0, "Warm up steps must be less than total training steps!"
        if self.multi_task and config.get("learning_rate_dict", None) is not None:
            self.lr_exp = {}
            for model_key in self.model_keys:
                self.lr_exp[model_key] = get_lr(config["learning_rate_dict"][model_key])
        else:
            self.lr_exp = get_lr(config["learning_rate"])

        # Loss
        if not self.multi_task:
            self.loss = get_loss(config["loss"], config["learning_rate"]["start_lr"])
        else:
            self.loss = {}
            for model_key in self.model_keys:
                if config.get("learning_rate_dict", None) is not None:
                    self.loss[model_key] = \
                        get_loss(config["loss_dict"][model_key], config["learning_rate_dict"][model_key]["start_lr"])
                else:
                    self.loss[model_key] = \
                        get_loss(config["loss_dict"][model_key], config["learning_rate"]["start_lr"])

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)

        # resuming and finetune
        if model_params["resuming"] and (self.rank == 0):
            ntest = model_params.get("data_bias_nsample", 1)
            origin_model = finetune_model if finetune_model is not None else resume_from
            logging.info(f"Resuming from {origin_model}.")
            state_dict = torch.load(origin_model)
            if force_load:
                input_keys = list(state_dict.keys())
                target_keys = list(self.wrapper.state_dict().keys())
                missing_keys = [item for item in target_keys if item not in input_keys]
                if missing_keys:
                    target_state_dict = self.wrapper.state_dict()
                    slim_keys = []
                    for item in missing_keys:
                        state_dict[item] = target_state_dict[item].clone().detach()
                        new_key = True
                        for slim_key in slim_keys:
                            if slim_key in item:
                                new_key = False
                                break
                        if new_key:
                            tmp_keys = '.'.join(item.split('.')[:3])
                            slim_keys.append(tmp_keys)
                    slim_keys = [i + '.*' for i in slim_keys]
                    logging.warning(
                        f"Force load mode allowed! These keys are not in ckpt and will re-init: {slim_keys}")
            self.wrapper.load_state_dict(state_dict)
            # finetune
            if finetune_model is not None and model_params["fitting_net"].get("type", "ener") in ['ener',
                                                                                                  'direct_force_ener',
                                                                                                  'atten_vec_lcc']:
                old_type_map, new_type_map = model_params['type_map'], model_params['new_type_map']
                self.model.fitting_net.change_energy_bias(
                    config,
                    self.model,
                    old_type_map,
                    new_type_map,
                    ntest=ntest,
                    bias_shift=model_params.get("bias_shift", "delta"),
                )

        # Multi-task share params
        if shared_links is not None:
            self.wrapper.share_params(shared_links, resume=model_params["resuming"])

        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)
        if dist.is_initialized():
            torch.cuda.set_device(LOCAL_RANK)
            # DDP will guarantee the model parameters are identical across all processes
            self.wrapper = DDP(
                self.wrapper,
                device_ids=[LOCAL_RANK],
                find_unused_parameters=True,
                output_device=LOCAL_RANK
            )

        # TODO ZD add lr warmups for multitask
        def warm_up_linear(step, warmup_steps):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return self.lr_exp.value(step - warmup_steps) / self.lr_exp.start_lr

        # TODO ZD add optimizers for multitask
        if self.opt_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.wrapper.parameters(), lr=self.lr_exp.start_lr
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: warm_up_linear(step, self.warmup_steps),
            )
        elif self.opt_type == "LKF":
            self.optimizer = LKFOptimizer(
                self.wrapper.parameters(), 0.98, 0.99870, self.kf_blocksize
            )
        else:
            raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

        # Get model prob for multi-task
        if self.multi_task:
            self.model_prob = np.array([0.0 for key in self.model_keys])
            if training_params.get('model_prob', None) is not None:
                model_prob = training_params['model_prob']
                for ii, model_key in enumerate(self.model_keys):
                    if model_key in model_prob:
                        self.model_prob[ii] += float(model_prob[model_key])
            else:
                for ii, model_key in enumerate(self.model_keys):
                    self.model_prob[ii] += float(len(self.training_data[model_key]))
            sum_prob = np.sum(self.model_prob)
            assert sum_prob > 0., "Sum of model prob must be larger than 0!"
            self.model_prob = self.model_prob / sum_prob

    def run(self):
        fout = (
            open(self.disp_file, mode="w", buffering=1) if self.rank == 0 else None
        )  # line buffered
        logging.info("Start to train %d steps.", self.num_steps)
        if dist.is_initialized():
            logging.info(f"Rank: {dist.get_rank()}/{dist.get_world_size()}")

        def step(_step_id, task_key="Default"):
            self.wrapper.train()
            if isinstance(self.lr_exp, dict):
                _lr = self.lr_exp[task_key]
            else:
                _lr = self.lr_exp
            cur_lr = _lr.value(_step_id)
            pref_lr = cur_lr
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = self.get_data(is_train=True, task_key=task_key)
            if self.opt_type == "Adam":
                cur_lr = self.scheduler.get_last_lr()[0]
                if _step_id < self.warmup_steps:
                    pref_lr = _lr.start_lr
                else:
                    pref_lr = cur_lr
                model_pred, loss, more_loss = self.wrapper(
                    **input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key
                )
                loss.backward()
                if self.gradient_max_norm > 0.:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.gradient_max_norm)
                    if not torch.isfinite(grad_norm).all():
                        # check local gradnorm single GPU case, trigger NanDetector
                        raise FloatingPointError("gradients are Nan/Inf")
                self.optimizer.step()
                self.scheduler.step()
            elif self.opt_type == "LKF":
                if isinstance(self.loss, EnergyStdLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    pref_e = self.kf_start_pref_e * (self.kf_limit_pref_e / self.kf_start_pref_e) ** (
                            _step_id / self.num_steps)
                    _ = KFOptWrapper.update_energy(input_dict, label_dict["energy"], pref_e)
                    pref_f = self.kf_start_pref_f * (self.kf_limit_pref_f / self.kf_start_pref_f) ** (
                            _step_id / self.num_steps)
                    p_energy, p_force = KFOptWrapper.update_force(
                        input_dict, label_dict["force"], pref_f
                    )
                    # [coord, atype, natoms, mapping, shift, selected, box]
                    model_pred = {"energy": p_energy, "force": p_force}
                    module = self.wrapper.module if dist.is_initialized() else self.wrapper
                    loss, more_loss = module.loss[task_key](
                        model_pred, label_dict, input_dict["natoms"], learning_rate=pref_lr
                    )
                elif isinstance(self.loss, DenoiseLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    module = self.wrapper.module if dist.is_initialized() else self.wrapper
                    model_pred = KFOptWrapper.update_denoise_coord(input_dict, label_dict["clean_coord"], 1,
                                                                   module.loss[task_key].mask_loss_coord,
                                                                   label_dict["coord_mask"])
                    loss, more_loss = module.loss[task_key](
                        model_pred, label_dict, input_dict["natoms"], learning_rate=pref_lr
                    )
            else:
                raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

            # Log and persist
            if _step_id % self.disp_freq == 0:
                self.wrapper.eval()
                msg = f"step={_step_id}, lr={cur_lr:.4f}"

                def log_loss_train(_loss, _more_loss, _task_key='Default'):
                    results = {}
                    if not self.multi_task:
                        suffix = ''
                    else:
                        suffix = f'_{_task_key}'
                    _msg = f"loss{suffix}={_loss:.4f}"
                    rmse_val = {
                        item: _more_loss[item] for item in _more_loss if 'l2_' not in item
                    }
                    for item in sorted(list(rmse_val.keys())):
                        _msg += f", {item}_train{suffix}={rmse_val[item]:.4f}"
                        results[item] = rmse_val[item]
                        self.wandb_log({item: rmse_val[item]}, _step_id, f"_train{suffix}")
                    return _msg, results

                def log_loss_valid(_task_key='Default'):
                    single_results = {}
                    sum_natoms = 0
                    if not self.multi_task:
                        suffix = ''
                        valid_numb_batch = self.valid_numb_batch
                    else:
                        suffix = f'_{_task_key}'
                        valid_numb_batch = self.valid_numb_batch[_task_key]
                    for ii in range(valid_numb_batch):
                        self.optimizer.zero_grad()
                        input_dict, label_dict = self.get_data(is_train=False, task_key=_task_key)
                        _, loss, more_loss = self.wrapper(
                            **input_dict,
                            cur_lr=pref_lr,
                            label=label_dict,
                            task_key=_task_key,
                        )
                        # more_loss.update({"rmse": math.sqrt(loss)})
                        natoms = input_dict["natoms"][0, 0]
                        sum_natoms += natoms
                        for k, v in more_loss.items():
                            if 'l2_' not in k:
                                single_results[k] = (
                                        single_results.get(k, 0.0) + v * natoms
                                )
                    results = {
                        k: v / sum_natoms for k, v in single_results.items()
                    }
                    _msg = ""
                    for item in sorted(list(results.keys())):
                        _msg += f", {item}_valid{suffix}={results[item]:.4f}"
                        self.wandb_log({item: results[item]}, _step_id, f"_valid{suffix}")
                    return _msg, results

                if not self.multi_task:
                    temp_msg, train_results = log_loss_train(loss, more_loss)
                    msg += '\n' + temp_msg
                    temp_msg, valid_results = log_loss_valid()
                    msg += temp_msg
                else:
                    train_results = {_key: {} for _key in self.model_keys}
                    valid_results = {_key: {} for _key in self.model_keys}
                    train_msg = {}
                    valid_msg = {}
                    train_msg[task_key], train_results[task_key] = log_loss_train(loss, more_loss, _task_key=task_key)
                    for _key in self.model_keys:
                        if _key != task_key:
                            self.optimizer.zero_grad()
                            input_dict, label_dict = self.get_data(is_train=True, task_key=_key)
                            _, loss, more_loss = self.wrapper(
                                **input_dict,
                                cur_lr=pref_lr,
                                label=label_dict,
                                task_key=_key,
                            )
                            train_msg[_key], train_results[_key] = log_loss_train(loss, more_loss, _task_key=_key)
                        valid_msg[_key], valid_results[_key] = log_loss_valid(_task_key=_key)
                        msg += '\n' + train_msg[_key]
                        msg += valid_msg[_key]

                train_time = time.time() - self.t0
                self.t0 = time.time()
                msg += f", speed={train_time:.2f} s/{self.disp_freq if _step_id else 1} batches"
                logging.info(msg)
                self.wandb_log({"lr": cur_lr}, step_id)

                if fout:
                    if self.lcurve_should_print_header:
                        self.print_header(fout, train_results, valid_results)
                        self.lcurve_should_print_header = False
                    self.print_on_training(fout, _step_id, cur_lr, train_results, valid_results)

            if (
                    ((_step_id + 1) % self.save_freq == 0 and _step_id != 0)
                    or (_step_id + 1) == self.num_steps
            ) and (self.rank == 0 or dist.get_rank() == 0):
                # Handle the case if rank 0 aborted and re-assigned
                self.latest_model = Path(self.save_ckpt)
                self.latest_model = self.latest_model.with_name(
                    f"{self.latest_model.stem}_{_step_id + 1}{self.latest_model.suffix}")
                logging.info(f"Saving model to {self.latest_model}")
                module = self.wrapper.module if dist.is_initialized() else self.wrapper
                self.save_model(self.latest_model, lr=cur_lr, step=_step_id)

        self.t0 = time.time()
        with logging_redirect_tqdm():
            for step_id in tqdm(
                    range(self.num_steps), disable=bool(dist.get_rank()) if dist.is_initialized() else None
            ):  # set to None to disable on non-TTY; disable on not rank 0
                if self.multi_task:
                    model_index = dp_random.choice(
                        np.arange(len(self.model_keys)), p=np.array(self.model_prob)
                    )
                    model_key = self.model_keys[model_index]
                else:
                    model_key = "Default"
                step(step_id, model_key)

        if (
                self.rank == 0 or dist.get_rank() == 0
        ):  # Handle the case if rank 0 aborted and re-assigned
            try:
                os.symlink(self.latest_model, self.save_ckpt)
            except OSError:
                self.save_model(self.save_ckpt, lr=0, step=self.num_steps)
            logging.info(f"Trained model has been saved to: {self.save_ckpt}")

            if JIT:
                self.wrapper.save("torchscript_model.pt")
        if fout:
            fout.close()

    def save_model(self, save_path, lr=0., step=0):
        module = self.wrapper.module if dist.is_initialized() else self.wrapper
        module.train_infos['lr'] = lr
        module.train_infos['step'] = step
        torch.save(module.state_dict(), save_path)

    def get_data(self, is_train=True, task_key="Default"):
        if not self.multi_task:
            if is_train:
                try:
                    batch_data = next(iter(self.training_data))
                except StopIteration:
                    # Refresh the status of the dataloader to start from a new epoch
                    self.training_data = BufferedIterator(iter(self.training_dataloader))
                    batch_data = next(iter(self.training_data))
            else:
                try:
                    batch_data = next(iter(self.validation_data))
                except StopIteration:
                    self.validation_data = BufferedIterator(
                        iter(self.validation_dataloader)
                    )
                    batch_data = next(iter(self.validation_data))
        else:
            if is_train:
                try:
                    batch_data = next(iter(self.training_data[task_key]))
                except StopIteration:
                    # Refresh the status of the dataloader to start from a new epoch
                    self.training_data[task_key] = BufferedIterator(iter(self.training_dataloader[task_key]))
                    batch_data = next(iter(self.training_data[task_key]))
            else:
                try:
                    batch_data = next(iter(self.validation_data[task_key]))
                except StopIteration:
                    self.validation_data[task_key] = BufferedIterator(
                        iter(self.validation_dataloader[task_key])
                    )
                    batch_data = next(iter(self.validation_data[task_key]))

        for key in batch_data.keys():
            if not isinstance(batch_data[key], list):
                batch_data[key] = batch_data[key].to(DEVICE)
            else:
                batch_data[key] = [item.to(DEVICE) for item in batch_data[key]]
        input_dict = {}
        for item in [
            "coord",
            "atype",
            "natoms",
            "mapping",
            "shift",
            "selected",
            "selected_loc",
            "selected_type",
            "box",
        ]:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ["energy", "force", "virial", "clean_coord", "clean_type", "coord_mask", "type_mask"]:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        return input_dict, label_dict

    def wandb_log(self, data: dict, step, type_suffix=""):
        if not self.wandb_enabled or self.rank != 0:
            return
        for k, v in data.items():
            wb.log({k + type_suffix: v}, step=step)

    def print_header(self, fout, train_results, valid_results):
        train_keys = sorted(list(train_results.keys()))
        print_str = ""
        print_str += "# %5s" % "step"
        if not self.multi_task:
            if valid_results is not None:
                prop_fmt = "   %11s %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_trn")
        else:
            for model_key in self.model_keys:
                if valid_results[model_key] is not None:
                    prop_fmt = "   %11s %11s"
                    for k in train_results[model_key].keys():
                        print_str += prop_fmt % (k + f"_val_{model_key}", k + f"_trn_{model_key}")
                else:
                    prop_fmt = "   %11s"
                    for k in train_results[model_key].keys():
                        print_str += prop_fmt % (k + f"_trn_{model_key}")
        print_str += "   %8s\n" % "lr"
        fout.write(print_str)
        fout.flush()

    def print_on_training(self, fout, step_id, cur_lr, train_results, valid_results):
        train_keys = sorted(list(train_results.keys()))
        print_str = ""
        print_str += "%7d" % step_id
        if not self.multi_task:
            if valid_results is not None:
                prop_fmt = "   %11.2e %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (valid_results[k], train_results[k])
            else:
                prop_fmt = "   %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (train_results[k])
        else:
            for model_key in self.model_keys:
                if valid_results[model_key] is not None:
                    prop_fmt = "   %11.2e %11.2e"
                    for k in valid_results[model_key].keys():
                        print_str += prop_fmt % (
                            valid_results[model_key][k],
                            train_results[model_key][k],
                        )
                else:
                    prop_fmt = "   %11.2e"
                    for k in train_results[model_key].keys():
                        print_str += prop_fmt % (train_results[model_key][k])
        print_str += "   %8.1e\n" % cur_lr
        fout.write(print_str)
        fout.flush()
