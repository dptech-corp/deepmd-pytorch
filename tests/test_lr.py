import numpy as np
import unittest

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.utils import learning_rate

from deepmd_pt.utils.learning_rate import LearningRateExp, LearningRatePolynomial


class TestLearningRate(unittest.TestCase):

    def setUp(self):
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = np.arange(400, 601, 100)
        self.stop_steps = np.arange(500, 1600, 500)

    def test_consistency(self):
        for decay_step in self.decay_steps:
            for stop_step in self.stop_steps:
                self.decay_step = decay_step
                self.stop_step = stop_step
                self.judge_it()

    def judge_it(self):
        base_lr = learning_rate.LearningRateExp(self.start_lr, self.stop_lr, self.decay_step)
        g = tf.Graph()
        with g.as_default():
            global_step = tf.placeholder(shape=[], dtype=tf.int32)
            t_lr = base_lr.build(global_step, self.stop_step)

        my_lr = LearningRateExp(self.start_lr, self.stop_lr, self.decay_step, self.stop_step)
        with tf.Session(graph=g) as sess:
            base_vals = [sess.run(t_lr, feed_dict={global_step: step_id}) for step_id in range(self.stop_step) if
                         step_id % self.decay_step != 0]
        my_vals = [my_lr.value(step_id) for step_id in range(self.stop_step) if step_id % self.decay_step != 0]
        self.assertTrue(np.allclose(base_vals, my_vals))


class TestLearningRatePolynomial(unittest.TestCase):
    def setUp(self):
        self.start_lr = 2e-4
        self.stop_lr = 1e-4
        self.stop_steps = 10
    
    def test_power_one_1(self):
        my_lr = LearningRatePolynomial(self.start_lr, self.stop_lr, self.stop_steps)
        my_vals = [my_lr.value(step_id) for step_id in range(self.stop_steps+1)]
        base_vals = [0.0002, 0.00019, 0.00018, 0.00017, 0.00016, 0.00015, 0.00014, 0.00013, 0.00012, 0.00011, 0.00010]
        self.assertTrue(np.allclose(base_vals, my_vals))

    def test_power_one_2(self):
        my_lr = LearningRatePolynomial(self.start_lr, self.stop_lr, self.stop_steps, decay_steps=5)
        my_vals = [my_lr.value(step_id) for step_id in range(self.stop_steps+1)]
        base_vals = [0.0002, 0.00018, 0.00016, 0.00014, 0.00012, 0.00010, 0.00010, 0.00010, 0.00010, 0.00010, 0.00010]
        self.assertTrue(np.allclose(base_vals, my_vals))
    
    def test_power_one_3(self):
        my_lr = LearningRatePolynomial(self.start_lr, self.stop_lr, self.stop_steps, decay_steps=5, cycle=True)
        my_vals = [my_lr.value(step_id) for step_id in range(self.stop_steps+1)]
        base_vals = [0.0002, 0.00018, 0.00016, 0.00014, 0.00012, 0.00010, 0.00014, 0.00013, 0.00012, 0.00011, 0.00010]
        self.assertTrue(np.allclose(base_vals, my_vals))
    
    def test_power_two(self):
        my_lr = LearningRatePolynomial(self.start_lr, self.stop_lr, self.stop_steps, power=2)
        my_vals = [my_lr.value(step_id) for step_id in range(self.stop_steps+1)]
        base_vals = [0.0002, 0.000181, 0.000164, 0.000149, 0.000136, 0.000125, 0.000116, 0.000109, 0.000104, 0.000101, 0.00010]
        self.assertTrue(np.allclose(base_vals, my_vals))

if __name__ == '__main__':
    unittest.main()
