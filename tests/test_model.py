import collections
import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd import op
from deepmd.common import data_requirement, expand_sys_str
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.loss import EnerStdLoss
from deepmd.model import EnerModel
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.learning_rate import LearningRateExp

from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp as MyLRExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel
from deepmd_pt.env import *
from deepmd_pt import my_random

CUR_DIR = os.path.dirname(__file__)

if TEST_DATASET == 'water':
    kDataSystems = [
        os.path.join(CUR_DIR, 'water/data/data_0'),
        os.path.join(CUR_DIR, 'water/data/data_1'),
        os.path.join(CUR_DIR, 'water/data/data_2')
    ]
elif TEST_DATASET == 'Cu':
    kDataSystems = "/data/cu_test.hdf5#/Cu16"

VariableState = collections.namedtuple('VariableState', ['value', 'gradient'])


def torch2tf(torch_name):
    fields = torch_name.split('.')
    element_id = int(fields[2])
    if fields[0] == 'embedding_net':
        layer_id = int(fields[4]) + 1
        weight_type = fields[5]
        return 'filter_type_all/%s_%d_%d:0' % (weight_type, layer_id, element_id)
    elif fields[3] == 'deep_layers':
        layer_id = int(fields[4])
        weight_type = fields[5]
        return 'layer_%d_type_%d/%s:0' % (layer_id, element_id, weight_type)
    elif fields[3] == 'final_layer':
        weight_type = fields[4]
        return 'final_layer_type_%d/%s:0' % (element_id, weight_type)
    else:
        raise RuntimeError('Unexpected parameter name: %s' % torch_name)


class DpTrainer(object):

    def __init__(self):
        self.batch_size = 3
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16
        self.n_neuron = [32, 32, 32]
        self.data_stat_nbatch = 3
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600
        self.start_pref_e = 1.
        self.limit_pref_e = 2.
        self.start_pref_f = 2.
        self.limit_pref_f = 1.
        if TEST_DATASET == 'water':
            self.type_map = ['O', 'H']
            self.sel = [46, 92]
        elif TEST_DATASET == 'Cu':
            self.type_map = ['Cu']
            self.sel = [138]
            self.n_neuron = [240, 240, 240]
        self.ntypes = len(self.type_map)

    def get_intermediate_state(self, num_steps=1):
        dp_model = self._get_dp_model()
        dp_loss = self._get_dp_loss()
        dp_lr = self._get_dp_lr()
        dp_ds = self._get_dp_dataset()
        dp_model.data_stat(dp_ds)

        # Build graph
        g = tf.Graph()
        with g.as_default():
            place_holders = self._get_dp_placeholders(dp_ds)
            model_pred = dp_model.build(
                coord_=place_holders['coord'],
                atype_=place_holders['type'],
                natoms=place_holders['natoms_vec'],
                box=place_holders['box'],
                mesh=place_holders['default_mesh'],
                input_dict=place_holders
            )
            global_step = tf.train.get_or_create_global_step()
            learning_rate = dp_lr.build(global_step, self.stop_steps)
            l2_l, _ = dp_loss.build(
                learning_rate=learning_rate,
                natoms=place_holders['natoms_vec'],
                model_dict=model_pred,
                label_dict=place_holders,
                suffix='test'
            )
            t_vars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            t_grad_and_vars = optimizer.compute_gradients(l2_l, t_vars)
            train_op = optimizer.apply_gradients(t_grad_and_vars, global_step)
            init_op = tf.global_variables_initializer()
            t_heads = {
                'loss': l2_l,
                'energy': model_pred['energy'],
                'force': model_pred['force']
            }

        # Get statistics of each component
        stat_dict = {
            'descriptor.mean': dp_model.descrpt.davg,
            'descriptor.stddev': dp_model.descrpt.dstd,
            'fitting_net.bias_atom_e': dp_model.fitting.bias_atom_e
        }

        # Get variables and their gradients
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            for _ in range(num_steps):
                batch = dp_ds.get_batch()
                feeds = self._get_feed_dict(batch, place_holders)
                sess.run(train_op, feed_dict=feeds)

            batch = dp_ds.get_batch()
            feeds = self._get_feed_dict(batch, place_holders)
            grads_and_vars, head_dict = sess.run([t_grad_and_vars, t_heads], feed_dict=feeds)
            vs_dict = {}
            for idx, one in enumerate(t_vars):
                grad, var = grads_and_vars[idx]
                vs_dict[one.name] = VariableState(var, grad)

        # Used for reproducing
        return batch, head_dict, stat_dict, vs_dict

    def _get_dp_dataset(self):
        global kDataSystems
        if TEST_DATASET == 'Cu':
            kDataSystems = expand_sys_str(kDataSystems)
        data = DeepmdDataSystem(
            systems=kDataSystems,
            batch_size=self.batch_size,
            test_size=1,
            rcut=self.rcut,
            type_map=self.type_map,
            trn_all_set=True
        )
        data.add_dict(data_requirement)
        return data

    def _get_dp_model(self):
        dp_descrpt = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron
        )
        dp_fitting = EnerFitting(
            descrpt=dp_descrpt,
            neuron=self.n_neuron
        )
        return EnerModel(
            descrpt=dp_descrpt,
            fitting=dp_fitting,
            type_map=self.type_map,
            data_stat_nbatch=self.data_stat_nbatch
        )

    def _get_dp_loss(self):
        return EnerStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f
        )

    def _get_dp_lr(self):
        return LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=self.stop_lr,
            decay_steps=self.decay_steps
        )

    def _get_dp_placeholders(self, dataset):
        place_holders = {}
        data_dict = dataset.get_data_dict()
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = tf.float64
            place_holders[kk] = tf.placeholder(prec, [None], name = 't_' + kk)
            place_holders['find_'+kk] = tf.placeholder(tf.float32, name = 't_find_' + kk)
        place_holders['type'] = tf.placeholder(tf.int32, [None], name='t_type')
        place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name='t_natoms')
        place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name='t_mesh')
        place_holders['is_training'] = tf.placeholder(tf.bool)
        return place_holders

    def _get_feed_dict(self, batch, place_holders):
        feed_dict = {}
        for kk in batch.keys():
            if kk == 'find_type' or kk == 'type':
                continue
            if 'find_' in kk:
                feed_dict[place_holders[kk]] = batch[kk]
            else:
                feed_dict[place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ['type']:
            feed_dict[place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh']:
            feed_dict[place_holders[ii]] = batch[ii]
        feed_dict[place_holders['is_training']] = True
        return feed_dict


class TestEnergy(unittest.TestCase):

    def setUp(self):
        self.dp_trainer = DpTrainer()
        self.wanted_step = 0
        for key in dir(self.dp_trainer):
            if not key.startswith('_') or key == 'get_intermediate_state':
                value = getattr(self.dp_trainer, key)
                setattr(self, key, value)

    def test_consistency(self):
        batch, head_dict, stat_dict, vs_dict = self.dp_trainer.get_intermediate_state(self.wanted_step)
        # Build DeePMD graph
        my_ds = DeepmdDataSet(kDataSystems, self.batch_size, self.type_map, self.rcut, self.sel)
        my_model = EnergyModel(
            model_params={
                'descriptor': {
                    'type': 'se_e2_a',
                    'sel': self.sel,
                    'rcut_smth': self.rcut_smth,
                    'rcut': self.rcut,
                    'neuron': self.filter_neuron,
                    'axis_neuron': self.axis_neuron,
                },
                'fitting_net': {
                    'neuron': self.n_neuron
                },
                'data_stat_nbatch': self.data_stat_nbatch
            },
            training_data=my_ds
        )
        my_model.to(DEVICE)
        my_lr = MyLRExp(self.start_lr, self.stop_lr, self.decay_steps, self.stop_steps)
        my_loss = EnergyStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f
        )

        # Keep statistics consistency between 2 implentations
        my_em = my_model.embedding_net
        my_em.mean = stat_dict['descriptor.mean'].reshape([self.ntypes, my_em.nnei, 4])
        my_em.mean = torch.tensor(my_em.mean, device=DEVICE)
        my_em.stddev = stat_dict['descriptor.stddev'].reshape([self.ntypes, my_em.nnei, 4])
        my_em.stddev = torch.tensor(my_em.stddev, device=DEVICE)
        my_model.fitting_net.bias_atom_e = stat_dict['fitting_net.bias_atom_e']

        # Keep parameter value consistency between 2 implentations
        for name, param in my_model.named_parameters():
            var_name = torch2tf(name)
            var = vs_dict[var_name].value
            with torch.no_grad():
                src = torch.from_numpy(var)
                dst = param.data
                print(name)
                print(src.mean(), src.std())
                print(dst.mean(), dst.std())
                dst.copy_(src)
        # Start forward computing
        batch = my_ds._data_systems[0].preprocess(batch)
        batch['coord'].requires_grad_(True)
        batch['natoms'] = torch.tensor(batch['natoms_vec'], device=batch['coord'].device).unsqueeze(0)
        p_energy, p_force = my_model(batch['coord'], batch['atype'], batch['natoms'],
        batch['mapping'], batch['shift'], batch['selected'])
        cur_lr = my_lr.value(self.wanted_step)
        loss = my_loss(cur_lr, batch['natoms'], p_energy, p_force, batch['energy'], batch['force'])[0]
        self.assertTrue(np.allclose(head_dict['energy'], p_energy.view(-1).cpu().detach().numpy()))
        self.assertTrue(np.allclose(head_dict['force'], p_force.view(self.batch_size, -1).cpu().detach().numpy()))
        rtol= 1e-5; atol=1e-8
        self.assertTrue(np.allclose(head_dict['loss'], loss.cpu().detach().numpy(), rtol=rtol,atol=atol))
        optimizer = torch.optim.Adam(my_model.parameters(), lr=cur_lr)
        optimizer.zero_grad()
        def step(step_id):
            bdata = self.training_data.get_batch()
            optimizer.zero_grad()
        # Compare gradient for consistency
        loss.backward()

        for name, param in my_model.named_parameters():
            var_name = torch2tf(name)
            var_grad = vs_dict[var_name].gradient
            param_grad = param.grad.cpu()
            var_grad = torch.tensor(var_grad)
            assert np.allclose(var_grad, param_grad, rtol=rtol, atol=atol)
if __name__ == '__main__':
    unittest.main()
