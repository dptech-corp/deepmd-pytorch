import collections
import torch
import re
import json
import numpy as np
import tensorflow.compat.v1 as tf
import copy
import unittest

from deepmd.train.trainer import DPTrainer
from deepmd.train.run_options import RunOptions
from deepmd.utils import random as dp_random
from deepmd.entrypoints.train import get_modifier, get_data
from deepmd.utils.argcheck import normalize

from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.env import DEVICE
from deepmd_pt.utils.dataloader import BufferedIterator

VariableState = collections.namedtuple('VariableState', ['value', 'gradient'])

# change value to torch tensor and add dim if need
def totorch(value, add_dim=False):
    if add_dim:
        return torch.from_numpy(value).unsqueeze(0)
    else:
        return torch.from_numpy(value)

def tfdict2ptdict(my_model, tf_dict, config):
    pt_dict = {}
    type_split = config['model']['descriptor']['type'] not in ['se_atten']
    ntypes = len(config['model']['type_map'])
    neuron = len(config['model']['descriptor']['neuron'])
    for pt_key in my_model.state_dict().keys():
        # stable value
        if pt_key in ['descriptor.mean','descriptor.stddev']:
            pt_dict[pt_key] = totorch(tf_dict[pt_key].reshape((ntypes,-1,4)))
        elif pt_key in ['fitting_net.bias_atom_e']:
            pt_dict[pt_key] = totorch(tf_dict[pt_key])
        elif pt_key in ['type_embedding.embedding.weight']:
            tmp = tf_dict['type_embed_net/matrix_1:0'].value + tf_dict['type_embed_net/bias_1:0'].value
            # if ntypes == 8:
            #     tmp += np.eye(8)
            type_embed_net = np.vstack((tmp, np.zeros((1, tmp.shape[1]))))
            pt_dict[pt_key] = totorch(type_embed_net)
        elif 'descriptor.dpa1_attention.attention_layers' in pt_key:
            pt_words = pt_key.split('.')
            i = pt_words[3]
            if pt_words[-2] in ['in_proj']:
                qm, km, vm = [tf_dict['attention_layer_%s/%s/%s:0'%(i,this, pt_words[-1])].value for this in ['c_query', 'c_key', 'c_value']]
                in_proj_value = np.concatenate((qm, km, vm), axis=-1)
                pt_dict[pt_key] = totorch(in_proj_value, pt_words[-1] not in ['matrix'])
            else:
                add_dim = False
                if 'out_proj' in pt_words:
                    key1 = 'c_out'
                    key2 = pt_words[-1]
                    add_dim = key2 in ['bias']
                else:
                    key1 = 'layer_normalization' if i == '0' else 'layer_normalization_%s'%i
                    key2 = 'gamma' if pt_words[-1] == 'weight' else 'beta'
                tf_key = 'attention_layer_%s/%s/%s:0'%(i, key1, key2)
                pt_dict[pt_key] = totorch(tf_dict[tf_key].value, add_dim)
        else:
            pt_words = pt_key.split('.')
            # type_str = '_type_%s'%pt_words[2] if type_split else '_type_all'
            # descriptor.filter_layers
            if 'descriptor.filter_layers' in pt_key:
                type_str = '_%s'%pt_words[2] if type_split else ''
                tf_key = 'filter_type_all/%s_%d%s:0'%(pt_words[-1], int(pt_words[-2])+1, type_str)
                # if type_split:
                #     neuron_i = int(pt_words[-2]) + 1
                #     tf_key = 'filter_type_%s/%s_%d:0'%(pt_words[2], pt_words[-1], neuron_i)
                # else:
                #     tf_key = 'filter_type_all/%s_%d:0'%(pt_words[-1], neuron_i)
                # pt_dict[pt_key] = totorch(tf_dict[tf_key].value, pt_words[-1] not in ['matrix'])
            # fitting_net.filter_layers
            elif 'fitting_net.filter_layers' in pt_key:
                type_str = '_type_%s'%pt_words[2] if type_split else ''
                neuron_str = 'final_layer' if pt_words[-2]=='final_layer' else 'layer_%s'%pt_words[-2]
                tf_key = '%s%s/%s:0'%(neuron_str, type_str, pt_words[-1])
                
                # if 'final_layer' in pt_key:
                #     tf_key = 'final_layer%s/%s:0'%(type_str, pt_words[-1])
                # else:
                #     tf_key = 'layer_%s%s/%s:0'%(pt_words[2], type_str, pt_words[-1])
            pt_dict[pt_key] = totorch(tf_dict[tf_key].value, pt_words[-1] not in ['matrix'])
    return pt_dict

class TrainerForTest():
    def __init__(self, pt_file):
        self.read_config(pt_file)
        self.get_tfmodel()
        self.stop_steps = 0
        self.pbc = np.all([i.pbc for i in self.train_data.data_systems])

    def read_config(self, pt_file):
        with open(pt_file)as f:
            self.config = json.load(f)
        is_sea = self.config['model']['descriptor'].get('type', 'se_e2_a') not in ['se_atten']
        # temp method to skip the error when ntypes is 8 and copying parameters from tf to pt.
        if len(self.config['model']['type_map']) == 8:
            self.config['model']['type_map'].append('new')
        self.config['model']['descriptor']['type_one_side'] = is_sea
        config = copy.deepcopy(self.config)
        if is_sea:
            self.jdata = normalize(config)
        else:
            self.config['model']['descriptor']['normalize'] = True
            for del_key in ['post_ln','ffn','ffn_embed_dim','activation','scaling_factor','head_num','normalize','temperature']:
                del config['model']['descriptor'][del_key]
            self.jdata = normalize(config)

    def get_tfmodel(self):
        jdata = self.jdata
        run_opt = RunOptions(log_level=20)
        self.trainer = DPTrainer(jdata, run_opt=run_opt, is_compress=False)
        rcut = self.trainer.model.get_rcut()
        type_map = self.trainer.model.get_type_map()
        if len(type_map) == 0:
            ipt_type_map = None
        else:
            ipt_type_map = type_map
        seed = jdata["training"].get("seed", None)
        if seed is not None:
            # avoid the same batch sequence among workers
            seed += run_opt.my_rank
            seed = seed % (2**32)
        modifier = get_modifier(jdata["model"].get("modifier", None))
        dp_random.seed(seed)
        self.train_data = get_data(jdata["training"]["training_data"], rcut, ipt_type_map, modifier)
        self.valid_data = None

    def _get_dp_placeholders(self, dataset):
        place_holders = {}
        data_dict = dataset.get_data_dict()
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = tf.float64
            place_holders[kk] = tf.placeholder(prec, [None], name='t_' + kk)
            place_holders['find_' + kk] = tf.placeholder(tf.float32, name='t_find_' + kk)
        place_holders['type'] = tf.placeholder(tf.int32, [None], name='t_type')
        ntypes = len(self.jdata['model']['type_map'])
        place_holders['natoms_vec'] = tf.placeholder(tf.int32, [ntypes + 2], name='t_natoms')
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
                if kk not in place_holders:
                    continue
                feed_dict[place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ['type']:
            feed_dict[place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh']:
            feed_dict[place_holders[ii]] = batch[ii]
        feed_dict[place_holders['is_training']] = True
        return feed_dict

    def get_tfdata(self):
        self.trainer.model.data_stat(self.train_data)
        # Build graph
        g = tf.Graph()
        with g.as_default():
            place_holders = self._get_dp_placeholders(self.train_data)
            model_pred = self.trainer.model.build(
                coord_=place_holders['coord'],
                atype_=place_holders['type'],
                natoms=place_holders['natoms_vec'],
                box=place_holders['box'],
                mesh=place_holders['default_mesh'],
                input_dict=place_holders,
            )
            # temp_val = self.model.model.temp_val
            init_op = tf.global_variables_initializer()
            global_step = tf.train.get_or_create_global_step()
            learning_rate = self.trainer.lr.build(global_step, self.stop_steps)
            l2_l, _ = self.trainer.loss.build(
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
                'force': model_pred['force'],
                'virial': model_pred['virial'],
            }

        # Get statistics of each component
        bias_atom_e = self.trainer.model.fitting.bias_atom_e if self.trainer.model.fitting.bias_atom_e is not None else None
        vs_dict = {
            'descriptor.mean': self.trainer.model.descrpt.davg,
            'descriptor.stddev': self.trainer.model.descrpt.dstd,
            'fitting_net.bias_atom_e': bias_atom_e
        }

        # Get variables and their gradients
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            for _ in range(self.stop_steps):
                batch = self.train_data.get_batch()
                feeds = self._get_feed_dict(batch, place_holders)
                sess.run(train_op, feed_dict=feeds)

            batch = self.train_data.get_batch()
            feeds = self._get_feed_dict(batch, place_holders)
            grads_and_vars, head_dict = sess.run([t_grad_and_vars, t_heads], feed_dict=feeds)
            # vs_dict = {}
            for idx, one in enumerate(t_vars):
                grad, var = grads_and_vars[idx]
                vs_dict[one.name] = VariableState(var, grad)
        return batch, head_dict, vs_dict

    def load_tf_to_pt(self, my_model, vs_dict):
        ntypes = len(self.jdata['model']['type_map'])
        x = my_model.state_dict()
        pt_dict = tfdict2ptdict(my_model, tf_dict=vs_dict, config=self.jdata)
        my_model.load_state_dict(pt_dict, strict=False)
        for key, value in my_model.state_dict().items():
            if not (key in pt_dict and torch.allclose(value, pt_dict[key])):
                print('Error! key = %s'%key)
        return my_model

    def set_pt_model(self, vs_dict, tfbatch):
        # Build DeePMD graph
        systems = self.config['training']['training_data']['systems']
        batch_size = self.config['training']['training_data']['batch_size']
        model_params = self.config['model']
        type_split = model_params['descriptor']['type'] not in ['se_atten']
        my_ds = DpLoaderSet(systems, batch_size, model_params=model_params, type_split=type_split)
        data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
        sampled = make_stat_input(my_ds.systems, my_ds.dataloaders, data_stat_nbatch)
        my_model = get_model(model_params, sampled).to(DEVICE)
        # Keep statistics consistency between 2 implentations
        my_model = self.load_tf_to_pt(my_model, vs_dict)
        # Get pytorch data to train
        batch = my_ds.systems[0]._data_system.preprocess(tfbatch, self.pbc)
        batch['coord'].requires_grad_(True)
        batch['natoms'] = torch.tensor(batch['natoms_vec'], device=batch['coord'].device).unsqueeze(0)
        return my_model, batch
    
    def train_result(self, rtol=1E-4, atol=0):
        if self.config['model']['descriptor'].get('type', 'se_e2_a') in ['se_atten'] and self.config['model']['descriptor'].get('attn_layer', 0):
            rtol = rtol * 100
            atol = 0.02
        tfbatch, head_dict, vs_dict = self.get_tfdata()
        nframes = tfbatch['energy'].shape[0]
        # Build DeePMD graph
        my_model, batch = self.set_pt_model(vs_dict, tfbatch)
        model_predict = my_model(batch['coord'], batch['atype'], batch['natoms'],
                                 batch['mapping'], batch['shift'], batch['selected'], batch['selected_type'],
                                 batch['selected_loc'], batch['box'])
        
        return head_dict, model_predict, rtol, atol, nframes

class TestModel(unittest.TestCase):
    def setUp(self):
        self.json_files = [
            'tests/water/se_e2_a.json',
            'tests/water/se_atten.json',
            'tests/NoPBC/se_atten.json',
        ]
    
    def test_models(self):
        for json_file in self.json_files:
            print('testing json: %s'%json_file)
            trainer = TrainerForTest(json_file)
            head_dict, model_predict, rtol, atol, nframes = trainer.train_result()
            p_energy, p_force, p_virial = model_predict['energy'], model_predict['force'], model_predict['virial']
            np.testing.assert_allclose(head_dict['energy'], p_energy.view(-1).cpu().detach().numpy(), rtol=rtol/100, atol=atol)
            np.testing.assert_allclose(head_dict['force'].reshape((nframes, -1, 3)), p_force.cpu().detach().numpy(), rtol=rtol, atol=atol)
            np.testing.assert_allclose(head_dict['virial'].reshape(nframes, 3, 3), p_virial.cpu().detach().numpy(), rtol=rtol, atol=atol*2)

if __name__ == '__main__':
    unittest.main()