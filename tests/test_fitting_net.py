import numpy as np
import re
import torch
import unittest
import json

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.fit.ener import EnerFitting

from deepmd_pt.utils.env import GLOBAL_NP_FLOAT_PRECISION, TEST_CONFIG
from deepmd_pt.model.task import EnergyFittingNet


class FakeDescriptor(object):

    def __init__(self, ntypes, embedding_width):
        self._ntypes = ntypes
        self._dim_out = embedding_width

    def get_ntypes(self):
        return self._ntypes

    def get_dim_out(self):
        return self._dim_out


def gen_key(type_id, layer_id, w_or_b):
    return (type_id, layer_id, w_or_b)


def base_fitting_net(dp_fn, embedding, natoms):
    g = tf.Graph()
    with g.as_default():
        t_embedding = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        t_natoms = tf.placeholder(tf.int32, [None])
        t_energy = dp_fn.build(t_embedding, t_natoms, {})
        init_op = tf.global_variables_initializer()
        t_vars = {}
        for var in tf.global_variables():
            key = None
            matched = re.match(r'layer_(\d)_type_(\d)/([a-z]+)', var.name)
            if matched:
                key = gen_key(type_id=matched.group(2), layer_id=matched.group(1), w_or_b=matched.group(3))
            else:
                matched = re.match(r'final_layer_type_(\d)/([a-z]+)', var.name)
                if matched:
                    key = gen_key(type_id=matched.group(1), layer_id=-1, w_or_b=matched.group(2))
            if key is not None:
                t_vars[key] = var

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        energy, values = sess.run([t_energy, t_vars], feed_dict={
            t_embedding: embedding,
            t_natoms: natoms
        })
        return energy, values


class TestFittingNet(unittest.TestCase):

    def setUp(self):
        nloc = 7
        self.embedding_width = 30
        self.natoms = np.array([nloc, nloc, 2, 5], dtype=np.int32)
        self.embedding = np.random.uniform(size=[4, nloc * self.embedding_width])
        self.ntypes = self.natoms.size - 2
        self.n_neuron = [32, 32, 32]

        fake_d = FakeDescriptor(2, 30)
        self.dp_fn = EnerFitting(fake_d, self.n_neuron)
        self.dp_fn.bias_atom_e = np.random.uniform(size=[self.ntypes])
        self.dp_fn.bias_atom_e = [1e8, 0]

    def test_consistency(self):
        dp_energy, values = base_fitting_net(self.dp_fn, self.embedding, self.natoms)
        my_fn = EnergyFittingNet(self.ntypes, self.embedding_width, self.n_neuron, self.dp_fn.bias_atom_e)
        for name, param in my_fn.named_parameters():
            matched = re.match('filter_layers\.(\d).deep_layers\.(\d)\.([a-z]+)', name)
            key = None
            if matched:
                key = gen_key(type_id=matched.group(1), layer_id=matched.group(2), w_or_b=matched.group(3))
            else:
                matched = re.match('filter_layers\.(\d).final_layer\.([a-z]+)', name)
                if matched:
                    key = gen_key(type_id=matched.group(1), layer_id=-1, w_or_b=matched.group(2))
            assert key is not None
            var = values[key]
            with torch.no_grad():
                # Keep parameter value consistency between 2 implentations
                param.data.copy_(torch.from_numpy(var))
        embedding = torch.from_numpy(self.embedding)
        embedding = embedding.view(4, -1, self.embedding_width)
        natoms = torch.from_numpy(self.natoms)
        atype = torch.zeros(1, natoms[0], dtype=torch.long)
        cnt = 0
        for i in range(natoms.shape[0] - 2):
            atype[:, cnt:cnt + natoms[i + 2]] = i
            cnt += natoms[i + 2]
        my_energy = my_fn(embedding, atype).detach()
        self.assertTrue(np.allclose(dp_energy, my_energy.numpy().reshape([-1])))


if __name__ == '__main__':
    unittest.main()
