import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd_pt.dataset import DeepmdDataSet
from deepmd.loss.ener import EnerStdLoss

from deepmd_pt.loss import EnergyStdLoss
import json
from deepmd_pt.env import TEST_CONFIG
from deepmd.common import expand_sys_str


CUR_DIR = os.path.dirname(__file__)


class TestLearningRate(unittest.TestCase):

    def setUp(self):
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.
        self.start_pref_f = 1000.
        self.limit_pref_f = 1.

        with open(TEST_CONFIG, 'r') as fin:
            content = fin.read()
        config = json.loads(content)
        model_config = config['model']
        self.rcut = model_config['descriptor']['rcut']
        self.rcut_smth = model_config['descriptor']['rcut_smth']
        self.sel = model_config['descriptor']['sel']
        self.batch_size = config['training']['training_data']['batch_size']
        self.systems = config['training']['validation_data']['systems']
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.dataset = DeepmdDataSet(self.systems,
         self.batch_size, model_config['type_map'], self.rcut, self.sel)
        self.filter_neuron = model_config['descriptor']['neuron']
        self.axis_neuron = model_config['descriptor']['axis_neuron']

    def test_consistency(self):
        base = EnerStdLoss(self.start_lr, self.start_pref_e, self.limit_pref_e, self.start_pref_f, self.limit_pref_f)
        g = tf.Graph()
        with g.as_default():
            t_cur_lr = tf.placeholder(shape=[], dtype=tf.float64)
            t_natoms = tf.placeholder(shape=[None], dtype=tf.int32)
            t_penergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_pforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_pvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_patom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lenergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_lforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_latom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_atom_pref = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_energy = tf.constant(1., dtype=tf.float64)
            find_force = tf.constant(1., dtype=tf.float64)
            find_virial = tf.constant(0., dtype=tf.float64)
            find_atom_energy = tf.constant(0., dtype=tf.float64)
            find_atom_pref = tf.constant(0., dtype=tf.float64)
            model_dict = {
                'energy': t_penergy,
                'force': t_pforce,
                'virial': t_pvirial,
                'atom_ener': t_patom_energy
            }
            label_dict = {
                'energy': t_lenergy,
                'force': t_lforce,
                'virial': t_lvirial,
                'atom_ener': t_latom_energy,
                'atom_pref': t_atom_pref,
                'find_energy': find_energy,
                'find_force': find_force,
                'find_virial': find_virial,
                'find_atom_ener': find_atom_energy,
                'find_atom_pref': find_atom_pref
            }
            t_loss = base.build(t_cur_lr, t_natoms, model_dict, label_dict, '')

        np_batch, pt_batch = self.dataset.get_batch()
        mine = EnergyStdLoss(self.start_lr, self.start_pref_e, self.limit_pref_e, self.start_pref_f, self.limit_pref_f)
        cur_lr = 1.2
        natoms = np_batch['natoms']
        l_energy = np_batch['energy']
        l_force = np_batch['force']
        p_energy = np.ones_like(l_energy)
        p_force = np.ones_like(l_force)
        nloc = natoms[0]
        batch_size = pt_batch['coord'].shape[0]
        virial = np.zeros(shape=[batch_size, 9])
        atom_energy = np.zeros(shape=[batch_size, nloc])
        atom_pref = np.zeros(shape=[batch_size, nloc*3])

        with tf.Session(graph=g) as sess:
            base_loss, _ = sess.run(t_loss, feed_dict={
                t_cur_lr: cur_lr,
                t_natoms: natoms,
                t_penergy: p_energy,
                t_pforce: p_force,
                t_pvirial: virial,
                t_patom_energy: atom_energy,
                t_lenergy: l_energy,
                t_lforce: l_force,
                t_lvirial: virial,
                t_latom_energy: atom_energy,
                t_atom_pref: atom_pref
            })
        my_loss = mine(
            cur_lr,
            pt_batch['natoms'],
            torch.from_numpy(p_energy),
            torch.from_numpy(p_force),
            torch.from_numpy(l_energy),
            torch.from_numpy(l_force)
        )
        my_loss = my_loss[0].detach().cpu()
        self.assertTrue(np.allclose(base_loss, my_loss.numpy()))


if __name__ == '__main__':
    unittest.main()
