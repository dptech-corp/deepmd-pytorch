      

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
from deepmd_pt.dataset import DeepmdDataSet
from torch.utils.data.distributed import DistributedSampler
from deepmd_pt.stat import make_stat_input

def train(rank, world_size, FLAGS):
    def setup(rank, world_size):
        if os.environ.get('MASTER_ADDR') is None:
            os.environ['MASTER_ADDR'] = 'localhost'
        if os.environ.get('MASTER_PORT') is None:
            os.environ['MASTER_PORT'] = "12345"
        if not env.DEVICE == torch.device('cpu'):
            device_count = torch.cuda.device_count()
            if device_count < world_size:
                logging.warn("There are more processes than GPUs !")
            torch.cuda.set_device(rank%device_count)

    def cleanup():
        dist.destroy_process_group()
    setup(rank, world_size)
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        content = fin.read()
    config = json.loads(content)
    training_params = config['training']
    model_params = config['model']
    dataset_params = training_params.pop('training_data')
    training_data = DeepmdDataSet(
            systems=dataset_params['systems'],
            batch_size=dataset_params['batch_size'],
            type_map=model_params['type_map'],
            rcut=model_params['descriptor']['rcut'],
            sel=model_params['descriptor']['sel']
        )  
    data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
    sampled = make_stat_input(training_data, data_stat_nbatch)
    trainer = training.Trainer(config, training_data,sampled,resume_from=FLAGS.CKPT)
    try:
        trainer.run()
    finally:
        cleanup()

def test(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        content = fin.read()
    config = json.loads(content)
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
        world_size = env.WORLD_SIZE
        if os.environ.get('WORLD_SIZE') is not None:
           world_size = int(os.environ.get('WORLD_SIZE'))
        if world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend,rank=int(os.environ.get('RANK')),world_size=world_size)
            rank = dist.get_rank()
            train(rank,world_size,FLAGS)
        else:
            train(0, world_size, FLAGS)

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