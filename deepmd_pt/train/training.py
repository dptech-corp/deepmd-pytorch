import logging
import os
import torch
import time
import math

from typing import Any, Dict
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
    ):
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        model_params = config["model"]
        training_params = config["training"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.pt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.opt_type = training_params.get("opt_type", "Adam")
        self.kf_blocksize = training_params.get("kf_blocksize", 5120)
        self.kf_start_pref_e = training_params.get("kf_start_pref_e", 1)
        self.kf_limit_pref_e = training_params.get("kf_limit_pref_e", 1)
        self.kf_start_pref_f = training_params.get("kf_start_pref_f", 1)
        self.kf_limit_pref_f = training_params.get("kf_limit_pref_f", 1)
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
                    config=training_params,
                    name=job_name,
                    settings=wb.Settings(start_method="fork"),
                )

        # Data + Model
        dp_random.seed(training_params["seed"])
        self.training_dataloader = DataLoader(
            training_data,
            sampler=torch.utils.data.RandomSampler(training_data),
            batch_size=None,
            num_workers=8,  # setting to 0 diverges the behavior of its iterator; should be >=1
            drop_last=False,
            pin_memory=True,
        )
        self.training_data = BufferedIterator(iter(self.training_dataloader))
        self.validation_dataloader = DataLoader(
            validation_data,
            sampler=torch.utils.data.RandomSampler(validation_data),
            batch_size=None,
            num_workers=1,
            drop_last=False,
            pin_memory=True,
        )

        self.validation_data = BufferedIterator(iter(self.validation_dataloader))
        if training_params.get("validation_data", None) is not None:
            self.valid_numb_batch = training_params["validation_data"].get(
                "numb_btch", 1
            )
        else:
            self.valid_numb_batch = 1
        model_params["resuming"] = (resume_from is not None)
        self.model = get_model(model_params, sampled).to(DEVICE)

        # Learning rate
        lr_params = config.pop("learning_rate")
        assert lr_params.pop("type", "exp"), "Only learning rate `exp` is supported!"
        lr_params["stop_steps"] = self.num_steps
        self.lr_exp = LearningRateExp(**lr_params)

        # Loss
        loss_params = config.pop("loss")
        loss_type = loss_params.pop("type", "ener")
        if loss_type == 'ener':
            loss_params["starter_learning_rate"] = lr_params["start_lr"]
            self.loss = EnergyStdLoss(**loss_params)
        elif loss_type == 'denoise':
            loss_params['ntypes'] = len(model_params['type_map'])
            self.loss = DenoiseLoss(**loss_params)
        else:
            raise NotImplementedError

        # Model Wrapper
        if JIT:
            self.model = torch.jit.script(self.model)
        self.wrapper = ModelWrapper(self.model, self.loss)

        if (resume_from is not None) and (self.rank == 0):
            logging.info(f"Resuming from {resume_from}.")
            state_dict = torch.load(resume_from)
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
                    logging.warning(f"Force load mode allowed! These keys are not in ckpt and will re-init: {slim_keys}")
            self.wrapper.load_state_dict(state_dict)

        if dist.is_initialized():
            torch.cuda.set_device(LOCAL_RANK)
            # DDP will guarantee the model parameters are identical across all processes
            self.wrapper = DDP(
                self.wrapper,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK
            )

        if self.opt_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.wrapper.parameters(), lr=self.lr_exp.start_lr
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: self.lr_exp.value(step) / self.lr_exp.start_lr,
            )
        elif self.opt_type == "LKF":
            self.optimizer = LKFOptimizer(
                self.wrapper.parameters(), 0.98, 0.99870, self.kf_blocksize
            )
        else:
            raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

        self.multi_task_mode = False
        self.task_keys = ["Default"]

        self.lcurve_should_print_header = True

    def run(self):
        fout = (
            open(self.disp_file, mode="w", buffering=1) if self.rank == 0 else None
        )  # line buffered
        logging.info("Start to train %d steps.", self.num_steps)
        if dist.is_initialized():
            logging.info(f"Rank: {dist.get_rank()}/{dist.get_world_size()}")

        def step(_step_id, task_key="Default"):
            cur_lr = self.lr_exp.value(_step_id)
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = self.get_data(is_train=True)
            if self.opt_type == "Adam":
                model_pred, loss, more_loss = self.wrapper(
                    **input_dict, cur_lr=cur_lr, label=label_dict, task_key=task_key
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            elif self.opt_type == "LKF":
                if isinstance(self.loss, EnergyStdLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    pref_e = self.kf_start_pref_e * (self.kf_limit_pref_e/self.kf_start_pref_e)**(_step_id/self.num_steps)
                    _ = KFOptWrapper.update_energy(input_dict, label_dict["energy"], pref_e)
                    pref_f = self.kf_start_pref_f * (self.kf_limit_pref_f/self.kf_start_pref_f)**(_step_id/self.num_steps)
                    p_energy, p_force = KFOptWrapper.update_force(
                        input_dict, label_dict["force"], pref_f
                    )
                    # [coord, atype, natoms, mapping, shift, selected, box]
                    model_pred = {"energy": p_energy, "force": p_force}
                    module = self.wrapper.module if dist.is_initialized() else self.wrapper
                    loss, more_loss = module.loss[task_key](
                            model_pred, label_dict, input_dict["natoms"], learning_rate=cur_lr
                        )
                elif isinstance(self.loss, DenoiseLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    module = self.wrapper.module if dist.is_initialized() else self.wrapper
                    model_pred = KFOptWrapper.update_denoise_coord(input_dict, label_dict["clean_coord"], 1, module.loss[task_key].mask_loss_coord, label_dict["coord_mask"])
                    loss, more_loss = module.loss[task_key](
                        model_pred, label_dict, input_dict["natoms"], learning_rate=cur_lr
                    )
            else:
                raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

            # Log and persist
            if _step_id % self.disp_freq == 0:
                # training
                train_results = {}
                valid_results = {}

                msg = f"step={_step_id}, lr={cur_lr:.4f}, loss={loss:.4f}"
                rmse_val = {
                    item: more_loss[item] for item in more_loss if 'l2_' not in item
                }
                for item in sorted(list(rmse_val.keys())):
                    if item in rmse_val:
                        msg += f", {item}_train={rmse_val[item]:.4f}"
                        train_results[item] = rmse_val[item]
                        self.wandb_log({item: rmse_val[item]}, _step_id, "_train")
                        rmse_val.pop(item)
                for rest_item in sorted(list(rmse_val.keys())):
                    if rest_item in rmse_val:
                        msg += f", {rest_item}={rmse_val[rest_item]:.4f}"
                        rmse_val.pop(rest_item)

                # validation
                if self.validation_data is not None:
                    single_results = {}
                    sum_natoms = 0
                    for ii in range(self.valid_numb_batch):
                        self.optimizer.zero_grad()
                        input_dict, label_dict = self.get_data(is_train=False)
                        _, loss, more_loss = self.wrapper(
                            **input_dict,
                            cur_lr=cur_lr,
                            label=label_dict,
                            task_key=task_key,
                        )
                        # more_loss.update({"rmse": math.sqrt(loss)})
                        natoms = input_dict["natoms"][0, 0]
                        sum_natoms += natoms
                        for k, v in more_loss.items():
                            if 'l2_' not in k:
                                single_results[k] = (
                                    single_results.get(k, 0.0) + v * natoms
                                )
                    valid_results = {
                        k: v / sum_natoms for k, v in single_results.items()
                    }
                    for item in sorted(list(valid_results.keys())):
                        msg += f", {item}_valid={valid_results[item]:.4f}"
                        self.wandb_log({item: valid_results[item]}, _step_id, "_valid")

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
                ((_step_id+1) % self.save_freq == 0 and _step_id != 0)
                or (_step_id+1) == self.num_steps
            ) and (self.rank == 0 or dist.get_rank() == 0):
                # Handle the case if rank 0 aborted and re-assigned
                self.latest_model = Path(self.save_ckpt)
                self.latest_model = self.latest_model.with_name(f"{self.latest_model.stem}_{_step_id+1}{self.latest_model.suffix}")
                logging.info(f"Saving model to {self.latest_model}")
                module = self.wrapper.module if dist.is_initialized() else self.wrapper
                torch.save(module.state_dict(), self.latest_model)

        self.t0 = time.time()
        with logging_redirect_tqdm():
            for step_id in tqdm(
                range(self.num_steps), disable=bool(dist.get_rank()) if dist.is_initialized() else None
            ):  # set to None to disable on non-TTY; disable on not rank 0
                step(step_id)

        if (
            self.rank == 0 or dist.get_rank() == 0
        ):  # Handle the case if rank 0 aborted and re-assigned
            if JIT:
                pth_model_path = "frozen_model.pth" # We use .pth to denote the frozen model
                self.model.save(pth_model_path)
                logging.info(f"Frozen model for inferencing has been saved to {pth_model_path}")
            try:
                os.symlink(self.latest_model, self.save_ckpt)
            except OSError:
                module = self.wrapper.module if dist.is_initialized() else self.wrapper
                torch.save(module.state_dict(), self.save_ckpt)
            logging.info(f"Trained model weight has been saved to {self.save_ckpt}")

        if fout:
            fout.close()

    def get_data(self, is_train=True):
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
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(DEVICE)
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
        if not self.multi_task_mode:
            if valid_results is not None:
                prop_fmt = "   %11s %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_trn")
        else:
            for fitting_key in train_keys:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11s %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_val", k + "_trn")
                else:
                    prop_fmt = "   %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_trn")
        print_str += "   %8s\n" % "lr"
        fout.write(print_str)
        fout.flush()

    def print_on_training(self, fout, step_id, cur_lr, train_results, valid_results):
        train_keys = sorted(list(train_results.keys()))
        print_str = ""
        print_str += "%7d" % step_id
        if not self.multi_task_mode:
            if valid_results is not None:
                prop_fmt = "   %11.2e %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (valid_results[k], train_results[k])
            else:
                prop_fmt = "   %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (train_results[k])
        else:
            for fitting_key in train_keys:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11.2e %11.2e"
                    for k in valid_results[fitting_key].keys():
                        print_str += prop_fmt % (
                            valid_results[fitting_key][k],
                            train_results[fitting_key][k],
                        )
                else:
                    prop_fmt = "   %11.2e"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (train_results[fitting_key][k])
        print_str += "   %8.1e\n" % cur_lr
        fout.write(print_str)
        fout.flush()
