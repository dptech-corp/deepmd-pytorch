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

        # Data + Model
        my_random.seed(training_params['seed'])
        dataset_params = training_params.pop('validation_data')
        self.test_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map'],
            rcut=model_params['descriptor']['rcut'],
            sel=model_params['descriptor']['sel']
        )   
        dataset_params = training_params.pop('training_data')
        training_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map'],
            rcut=model_params['descriptor']['rcut'],
            sel=model_params['descriptor']['sel']
        )   
        #self.test_data = training_data
        self.model = EnergyModel(model_params, self.test_data).to(DEVICE)
        state_dict = torch.load("model.ckpt")
        self.model.load_state_dict(state_dict)

        # Loss
        loss_params = config.pop('loss')
        assert loss_params.pop('type', 'ener'), 'Only loss `ener` is supported!'
        loss_params['starter_learning_rate'] = 1.
        self.loss = EnergyStdLoss(**loss_params)

    def run(self):
        def step(step_id):
            bdata = self.test_data.__getitem__()
            # Prepare inputs
            coord = bdata['coord']
            atype = bdata['atype']
            natoms = bdata['natoms']
            box = bdata['box']
            l_energy = bdata['energy']
            l_force = bdata['force']

            # Compute prediction error
            coord.requires_grad_(True)
            p_energy, p_force = self.model(coord, atype, natoms, bdata['mapping'], bdata['shift'], bdata['selected'])
            l_force = l_force.view(-1, bdata['natoms'][0,0], 3)
            assert l_energy.shape == p_energy.shape
            assert l_force.shape == p_force.shape
            loss, rmse_e, rmse_f = self.loss(0., natoms, p_energy, p_force, l_energy, l_force)
            return rmse_e, rmse_f

        rmse_e, rmse_f = [], []
        for step_id in range(1000):
            e, f = step(step_id)
            rmse_e.append(e)
            rmse_f.append(f)
        e = sum(rmse_e)/len(rmse_e)
        f = sum(rmse_f)/len(rmse_f)
        print(e.item())
        print(f.item())
