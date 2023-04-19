import os
import argparse
import json
import logging
import torch
from deepmd_pt.utils import env
from deepmd_pt.train import training
from deepmd_pt.infer import inference
import torch.multiprocessing as mp
import torch.distributed as dist
from deepmd_pt.utils.dataset import DeepmdDataSet
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from deepmd_pt.utils.stat import make_stat_input


def train(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    training_params = config['training']
    model_params = config['model']
    training_dataset_params = training_params.pop('training_data')
    training_data = DeepmdDataSet(
            systems=training_dataset_params['systems'],
            batch_size=training_dataset_params['batch_size'],
            type_map=model_params['type_map'],
            rcut=model_params['descriptor']['rcut'],
            sel=model_params['descriptor']['sel']
        )
    validation_dataset_params = training_params.pop('validation_data')
    validation_data = DeepmdDataSet(
        systems=validation_dataset_params['systems'],
        batch_size=validation_dataset_params['batch_size'],
        type_map=model_params['type_map'],
        rcut=model_params['descriptor']['rcut'],
        sel=model_params['descriptor']['sel']
    )
    data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
    sampled = make_stat_input(training_data, data_stat_nbatch)
    trainer = training.Trainer(config, training_data, sampled, validation_data=validation_data, resume_from=FLAGS.CKPT)
    trainer.run()


def test(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    trainer = inference.Trainer(config, FLAGS.CKPT)
    trainer.run()


@record
def main(args=None):
    logging.basicConfig(
        level=logging.WARNING if env.LOCAL_RANK else logging.INFO,
        format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
    )
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
    main()
