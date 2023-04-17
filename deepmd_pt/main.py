

import os
import argparse
import json
import logging
import torch
from deepmd_pt import env
from deepmd_pt import training
from deepmd_pt import inference
import torch.multiprocessing as mp
import torch.distributed as dist

def train(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    trainer = training.Trainer(config, resume_from=FLAGS.CKPT)
    try:
        trainer.run()
    finally:
        dist.destroy_process_group()

def test(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    trainer = inference.Trainer(config, FLAGS.CKPT)
    trainer.run()


def main(args=None):
    parser = argparse.ArgumentParser(description='A tool to manager deep models of potential energy surface.')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('INPUT', help='A Json-format configuration file.')
    train_parser.add_argument('CKPT', nargs='?', help='Resumes from checkpoint.')

    test_parser = subparsers.add_parser('test', help='Test a model.')
    test_parser.add_argument('INPUT', help='A Json-format configuration file.')
    test_parser.add_argument('CKPT', help='Resumes from checkpoint.')
    FLAGS = parser.parse_args(args)
    if FLAGS.command == 'train':
        train(FLAGS)
    elif FLAGS.command == 'test':
        test(FLAGS)
    else:
        logging.error('Invalid command!')
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
    )
    main()
