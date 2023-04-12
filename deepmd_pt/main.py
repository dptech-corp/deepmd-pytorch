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

def train(rank, world_size, FLAGS):
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12350'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
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
    trainer = training.Trainer(config, resume_from=FLAGS.CKPT)
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
        if world_size > 1:
            mp.spawn(train,
                    args=(world_size, FLAGS),
                    nprocs=world_size,
                    join=True)
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