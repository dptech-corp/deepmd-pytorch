import argparse
import os

from collections import namedtuple
from matplotlib import pyplot
import numpy as np

CurveRecord = namedtuple('CurveRecord', ['step', 'rmse_e', 'rmse_f'])


def load_lcurve_file(path):
    with open(path, 'r') as fin:
        lines = fin.readlines()
    lcurve = []
    for item in lines:
        line = item.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        lcurve.append(fields)
    return lcurve

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def draw_line_2d(path, step_interval, rolling=100):
    lcurve = load_lcurve_file(path)
    steps = []
    rmse_e = []
    rmse_f = []
    for idx in range(0, len(lcurve), step_interval):
        item = lcurve[idx]
        step = item[0].rstrip(',').split('=')[-1]
        e = item[1].rstrip(',').split('=')[-1]
        f = item[2].rstrip(',').split('=')[-1]
        steps.append(int(step))
        rmse_e.append(float(e))
        rmse_f.append(float(f))
    pyplot.yscale("log")
    rmse_e = rolling_window(np.array(rmse_e), rolling).mean(axis=-1)
    rmse_f = rolling_window(np.array(rmse_f), rolling).mean(axis=-1)
    pyplot.plot(steps[rolling-1:], rmse_e, label='Energy')
    pyplot.plot(steps[rolling-1:], rmse_f, label='Force')


def run(FLAGS):
    pyplot.figure(figsize=(15,5))
    pyplot.title('RMSE over step')
    pyplot.xlabel('Step')
    pyplot.ylabel('RMSE')
    draw_line_2d('lcurve.out', FLAGS.sample_every_steps)
    pyplot.legend()

    if os.path.isfile(FLAGS.output_path):
        os.remove(FLAGS.output_path)
    pyplot.savefig(FLAGS.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw lines from a lcurve file.')
    parser.add_argument('-n', '--sample_every_steps', type=int, default=1, help='Sample every N steps.')
    parser.add_argument('-o', '--output_path', default='rmse_over_step.png', help='Where to write image.')
    FLAGS = parser.parse_args()
    run(FLAGS)
