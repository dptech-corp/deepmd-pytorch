import argparse
import json
import logging

from deepmd_pt import training
from deepmd_pt import inference


def train(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        content = fin.read()
    config = json.loads(content)
    trainer = training.Trainer(config)
    trainer.run()

def test(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        content = fin.read()
    config = json.loads(content)
    trainer = inference.Trainer(config)
    trainer.run()


def main(args=None):
    parser = argparse.ArgumentParser(description='A tool to manager deep models of potential energy suface.')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('INPUT', help='A Json-format configuration file.')
    train_parser = subparsers.add_parser('test', help='Test a model.')
    train_parser.add_argument('INPUT', help='A Json-format configuration file.')
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