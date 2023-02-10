import argparse
import os

from collections import namedtuple
from matplotlib import pyplot


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


def draw_line_2d(path, step_interval, sacle_ratio=1):
    lcurve = load_lcurve_file(path)
    steps = []
    rmse_e = []
    rmse_f = []
    for idx in range(0, len(lcurve), step_interval):
        item = lcurve[idx]
        steps.append(int(int(item[0])*sacle_ratio))
        rmse_e.append(float(item[3]))
        rmse_f.append(float(item[5]))
    pyplot.yscale("log")
    pyplot.plot(steps, rmse_e, label='Energy')
    pyplot.plot(steps, rmse_f, label='Force')


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
