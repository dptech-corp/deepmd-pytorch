import numpy as np
import os
import re
import torch
import json
import unittest

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.descriptor import DescrptSeA as DescrptSeA_tf

from deepmd_pt.utils import dp_random
from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.model.descriptor import DescrptSeA
from deepmd_pt.utils.env import GLOBAL_NP_FLOAT_PRECISION, DEVICE, TEST_CONFIG
from deepmd.common import expand_sys_str

CUR_DIR = os.path.dirname(__file__)


def gen_key(worb, depth, elemid):
    return (worb, depth, elemid)


def base_se_a(descriptor, coord, atype, natoms, box):
    g = tf.Graph()
    with g.as_default():
        name_pfx = 'd_sea_'
        t_coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + 't_coord')
        t_atype = tf.placeholder(tf.int32, [None, None], name=name_pfx + 't_type')
        t_natoms = tf.placeholder(tf.int32, [descriptor.ntypes + 2], name=name_pfx + 't_natoms')
        t_box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + 't_box')
        t_default_mesh = tf.placeholder(tf.int32, [None], name=name_pfx + 't_mesh')
        t_embedding = descriptor.build(t_coord, t_atype, t_natoms, t_box, t_default_mesh, input_dict={})
        fake_energy = tf.reduce_sum(t_embedding)
        t_force = descriptor.prod_force_virial(fake_energy, t_natoms)[0]
        t_vars = {}
        for var in tf.global_variables():
            ms = re.findall(r'([a-z]+)_(\d)_(\d)', var.name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[0], depth=int(m[1]), elemid=int(m[2]))
                t_vars[key] = var
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        embedding, force, values = sess.run([t_embedding, t_force, t_vars], feed_dict={
            t_coord: coord,
            t_atype: atype,
            t_natoms: natoms,
            t_box: box,
            t_default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })
        return embedding, force, values


class TestSeA(unittest.TestCase):

    def setUp(self):
        dp_random.seed(0)
        with open(TEST_CONFIG, 'r') as fin:
            content = fin.read()
        config = json.loads(content)
        model_config = config['model']
        self.rcut = model_config['descriptor']['rcut']
        self.rcut_smth = model_config['descriptor']['rcut_smth']
        self.sel = model_config['descriptor']['sel']
        self.bsz = config['training']['training_data']['batch_size']
        self.systems = config['training']['validation_data']['systems']
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        ds = DeepmdDataSet(self.systems, self.bsz, model_config['type_map'], self.rcut, self.sel)
        self.filter_neuron = model_config['descriptor']['neuron']
        self.axis_neuron = model_config['descriptor']['axis_neuron']
        self.np_batch, self.torch_batch = ds.get_batch()

    def test_consistency(self):
        dp_d = DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
            seed=1
        )
        dp_embedding, dp_force, dp_vars = base_se_a(
            descriptor=dp_d,
            coord=self.np_batch['coord'],
            atype=self.np_batch['atype'],
            natoms=self.np_batch['natoms'],
            box=self.np_batch['box'],
        )

        # Reproduced
        descriptor = DescrptSeA(
            self.rcut, self.rcut_smth, self.sel,
            self.filter_neuron, self.axis_neuron
        ).to(DEVICE)
        for name, param in descriptor.named_parameters():
            ms = re.findall(r'(\d)\.deep_layers\.(\d)\.([a-z]+)', name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[2], depth=int(m[1]) + 1, elemid=int(m[0]))
                var = dp_vars[key]
                with torch.no_grad():
                    # Keep parameter value consistency between 2 implentations
                    param.data.copy_(torch.from_numpy(var))

        pt_coord = self.torch_batch['coord']
        pt_coord.requires_grad_(True)
        index = self.torch_batch['mapping'].unsqueeze(-1).expand(-1, -1, 3)
        extended_coord = torch.gather(pt_coord, dim=1, index=index)
        extended_coord = extended_coord - self.torch_batch['shift']
        descriptor_out = descriptor(
            extended_coord,
            self.torch_batch['selected'],
            self.torch_batch['atype']
        )
        my_embedding = descriptor_out.cpu().detach().numpy()
        fake_energy = torch.sum(descriptor_out)
        fake_energy.backward()
        my_force = -pt_coord.grad.cpu().numpy()

        # Check
        self.assertTrue(np.allclose(dp_embedding, my_embedding))
        dp_force = dp_force.reshape(*my_force.shape)
        self.assertTrue(np.allclose(dp_force, my_force))


if __name__ == '__main__':
    unittest.main()
