import logging
import torch

from typing import Any, Dict

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel
from env import DEVICE, JIT
if torch.__version__.startswith("2"):
    import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
      
class Trainer(object):

    def __init__(self, config: Dict[str, Any], resume_from = None):
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
        dataset_params = training_params.pop('training_data')
        self.training_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map'],
            rcut=model_params['descriptor']['rcut'],
            sel=model_params['descriptor']['sel']
        )  
        self.model = EnergyModel(model_params, self.training_data).to(DEVICE)
        if JIT:
            self.model = torch.jit.script(self.model)
        self.rank = 0
        if dist.is_initialized() and dist.get_world_size()>1:
            self.model = DDP(self.model)
            self.rank = dist.get_rank()
            module = self.model.module
            logging.basicConfig()
            if self.rank == 0:
                logging.getLogger().setLevel(logging.INFO)
                torch.save(module.state_dict(), self.save_ckpt)
                dist.barrier()
            else:
                logging.getLogger().setLevel(logging.ERROR)
                dist.barrier()
                state_dict = torch.load(self.save_ckpt)
                self.model.module.load_state_dict(state_dict)
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
            bdata = self.training_data.__getitem__()
            self.optimizer.zero_grad()
            cur_lr = self.lr_exp.value(step_id)
            l_energy = bdata['energy']
            l_force = bdata['force']

            # Compute prediction error
            coord, atype, natoms = bdata['coord'], bdata['atype'], bdata['natoms']
            mapping, shift, selected, box = bdata['mapping'], bdata['shift'], bdata['selected'], bdata['box']
            p_energy, p_force, stress = self.model(coord, atype, natoms, mapping, shift, selected, box)
            l_force = l_force.view(-1, bdata['natoms'][0,0], 3)
            assert l_energy.shape == p_energy.shape
            assert l_force.shape == p_force.shape
            loss, rmse_e, rmse_f = self.loss(cur_lr, natoms, p_energy, p_force, l_energy, l_force)
            loss_val = loss.cpu().detach().numpy().tolist()
            logging.info('step=%d, lr=%f, loss=%f', step_id, cur_lr, loss_val)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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

        if self.rank == 0:
            module = self.model
            if isinstance(module, DDP):
                module = module.module
            if JIT:
                module.save("torchscript_model.pt")
            torch.save(module.state_dict(), self.save_ckpt)
        fout.close()
        logging.info('Saving model after all steps...')