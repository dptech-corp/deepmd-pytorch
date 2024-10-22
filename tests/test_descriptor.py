import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.env import op_module

from deepmd_pt.utils import dp_random
from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.model.descriptor import prod_env_mat_se_a
from deepmd_pt.utils.env import *
from deepmd.common import expand_sys_str
import json

CUR_DIR = os.path.dirname(__file__)


def base_se_a(rcut, rcut_smth, sel, batch, mean, stddev):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        stat_descrpt, descrpt_deriv, rij, nlist \
            = op_module.prod_env_mat_a(coord,
                                       atype,
                                       natoms_vec,
                                       box,
                                       default_mesh,
                                       tf.constant(mean),
                                       tf.constant(stddev),
                                       rcut_a=-1.,
                                       rcut_r=rcut,
                                       rcut_r_smth=rcut_smth,
                                       sel_a=sel,
                                       sel_r=[0 for i in sel])

        net_deriv_reshape = tf.ones_like(stat_descrpt)
        force = op_module.prod_force_se_a(net_deriv_reshape,
                                          descrpt_deriv,
                                          nlist,
                                          natoms_vec,
                                          n_a_sel=sum(sel),
                                          n_r_sel=0)

    with tf.Session(graph=g) as sess:
        return sess.run([stat_descrpt, force, nlist], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms'],
            atype: batch['atype'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


class TestSeA(unittest.TestCase):

    def setUp(self):
        dp_random.seed(20)
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
        self.np_batch, self.pt_batch = ds.get_batch()
        self.sec = np.cumsum(self.sel)
        self.ntypes = len(self.sel)
        self.nnei = sum(self.sel)

    def test_consistency(self):
        avg_zero = torch.zeros([self.ntypes, self.nnei * 4], dtype=GLOBAL_PT_FLOAT_PRECISION)
        std_ones = torch.ones([self.ntypes, self.nnei * 4], dtype=GLOBAL_PT_FLOAT_PRECISION)
        base_d, base_force, nlist = base_se_a(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            batch=self.np_batch,
            mean=avg_zero,
            stddev=std_ones
        )

        pt_coord = self.pt_batch['coord']
        pt_coord.requires_grad_(True)
        index = self.pt_batch['mapping'].unsqueeze(-1).expand(-1, -1, 3)
        extended_coord = torch.gather(pt_coord, dim=1, index=index)
        extended_coord = extended_coord - self.pt_batch['shift']
        my_d, _ = prod_env_mat_se_a(
            extended_coord.to(DEVICE),
            self.pt_batch['selected'],
            self.pt_batch['atype'],
            avg_zero.reshape([-1, self.nnei, 4]).to(DEVICE),
            std_ones.reshape([-1, self.nnei, 4]).to(DEVICE),
            self.rcut,
            self.rcut_smth,
        )
        my_d.sum().backward()
        bsz = pt_coord.shape[0]
        my_force = pt_coord.grad.view(bsz, -1, 3).cpu().detach().numpy()
        base_force = base_force.reshape(bsz, -1, 3)
        base_d = base_d.reshape(bsz, -1, self.nnei, 4)
        my_d = my_d.view(bsz, -1, self.nnei, 4).cpu().detach().numpy()
        nlist = nlist.reshape(bsz, -1, self.nnei)

        mapping = self.pt_batch['mapping'].cpu()
        selected = self.pt_batch['selected'].view(bsz, -1).cpu()
        mask = selected == -1
        selected = selected * ~mask
        my_nlist = torch.gather(mapping, dim=-1, index=selected)
        my_nlist = my_nlist * ~mask - mask.long()
        my_nlist = my_nlist.cpu().view(bsz, -1, self.nnei).numpy()
        self.assertTrue(np.allclose(nlist, my_nlist))
        self.assertTrue(np.allclose(np.mean(base_d, axis=2), np.mean(my_d, axis=2)))
        self.assertTrue(np.allclose(np.std(base_d, axis=2), np.std(my_d, axis=2)))
        # descriptors may be different when there are multiple neighbors in the same distance
        self.assertTrue(np.allclose(base_force, -my_force))


if __name__ == '__main__':
    unittest.main()
