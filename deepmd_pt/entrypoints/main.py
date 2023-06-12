import os
import argparse
import json
import logging
import torch
from deepmd_pt.utils import env
from deepmd_pt.train import training
from deepmd_pt.infer import inference
import torch.distributed as dist
from deepmd_pt.utils.dataset import DeepmdDataSet
from deepmd_pt.utils.dataloader import DpLoaderSet
from torch.distributed.elastic.multiprocessing.errors import record

from deepmd_pt.utils.finetune import load_model_params
from deepmd_pt.utils.stat import make_stat_input


def get_trainer(config, ckpt=None, force_load=False, finetune_model=None):
    config = load_model_params(ckpt, finetune_model, config)
    training_params = config['training']
    model_params = config['model']
    training_dataset_params = training_params['training_data']
    type_split = True
    if model_params['descriptor']['type'] in ['se_atten', 'se_uni']:
        type_split = False
    validation_dataset_params = training_params.pop('validation_data')
    # Initialize DDP
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        local_rank = int(local_rank)
        assert dist.is_nccl_available()
        dist.init_process_group(backend='nccl')

    training_systems = training_dataset_params['systems']
    validation_systems = validation_dataset_params['systems']

    # noise params
    noise_settings = None
    if config['loss'].get('type', 'ener') == 'denoise':
        noise_settings = {"noise_type": config['loss'].pop("noise_type", "uniform"),
                          "noise": config['loss'].pop("noise", 1.0),
                          "noise_mode": config['loss'].pop("noise_mode", "fix_num"),
                          "mask_num": config['loss'].pop("mask_num", 8),
                          "mask_prob": config['loss'].pop("mask_prob", 0.15),
                          "same_mask": config['loss'].pop("same_mask", False),
                          "mask_coord": config['loss'].pop("mask_coord", False),
                          "mask_type": config['loss'].pop("mask_type", False),
                          "max_fail_num": config['loss'].pop("max_fail_num", 10),
                          "mask_type_idx": len(model_params["type_map"]) - 1}
    # noise_settings = None
    validation_data = DpLoaderSet(validation_systems, validation_dataset_params['batch_size'], model_params,
                                  type_split=type_split, noise_settings=noise_settings)

    default_stat_file_name = f'stat_file_rcut{model_params["descriptor"]["rcut"]:.2f}_'\
        f'smth{model_params["descriptor"]["rcut_smth"]:.2f}_'\
        f'sel{model_params["descriptor"]["sel"]}.npz'
    model_params["stat_file_dir"] = training_params.get("stat_file_dir", "stat_files")
    model_params["stat_file"] = training_params.get("stat_file", default_stat_file_name)
    model_params["stat_file_path"] = os.path.join(model_params["stat_file_dir"], model_params["stat_file"])
    if ckpt or os.path.exists(model_params["stat_file_path"]):
        train_data = DpLoaderSet(training_systems, training_dataset_params['batch_size'], model_params,
                                 type_split=type_split, noise_settings=noise_settings)
        sampled = None
    else:
        train_data = DpLoaderSet(training_systems, training_dataset_params['batch_size'], model_params,
                                 type_split=type_split)
        #data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
        #sampled = make_stat_input(train_data.systems, train_data.dataloaders, data_stat_nbatch)
        sampled = True
        if noise_settings is not None:
            train_data = DpLoaderSet(training_systems, training_dataset_params['batch_size'], model_params,
                                     type_split=type_split, noise_settings=noise_settings)
    trainer = training.Trainer(config, train_data, sampled, validation_data=validation_data,
                               resume_from=ckpt, force_load=force_load, finetune_model=finetune_model)
    return trainer


def train(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    trainer = get_trainer(config, FLAGS.CKPT, FLAGS.force_load, FLAGS.finetune)
    trainer.run()


def test(FLAGS):
    logging.info('Configuration path: %s', FLAGS.INPUT)
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    trainer = inference.Tester(config, FLAGS.CKPT, FLAGS.numb_test)
    trainer.run()


def freeze(FLAGS):
    with open(FLAGS.INPUT, 'r') as fin:
        config = json.load(fin)
    model = torch.jit.script(inference.Tester(config, FLAGS.CKPT, 1).model)
    torch.jit.save(model, 'frozen_model.pt', {
        # TODO: _extra_files
    })


@record
def main(args=None):
    logging.basicConfig(
        level=logging.WARNING if env.LOCAL_RANK else logging.INFO,
        format=f"%(asctime)-15s {os.environ.get('RANK') or ''} [%(filename)s:%(lineno)d] %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description='A tool to manager deep models of potential energy surface.')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('INPUT', help='A Json-format configuration file.')
    train_parser.add_argument('CKPT', nargs='?', help='Resumes from checkpoint.')
    train_parser.add_argument("--force-load", action="store_true",
                              help='Force load from ckpt, other missing tensors will init from scratch')
    train_parser.add_argument("--finetune", help="The Finetune model.")

    test_parser = subparsers.add_parser('test', help='Test a model.')
    test_parser.add_argument('INPUT', help='A Json-format configuration file.')
    test_parser.add_argument('CKPT', help='Resumes from checkpoint.')
    test_parser.add_argument("-n", "--numb-test", default=100, type=int, help="The number of data for test")

    freeze_parser = subparsers.add_parser('freeze', help='Freeze a model.')
    freeze_parser.add_argument('INPUT', help='A Json-format configuration file.')
    freeze_parser.add_argument('CKPT', help='Resumes from checkpoint.')

    FLAGS = parser.parse_args(args)
    if FLAGS.command == 'train':
        train(FLAGS)
    elif FLAGS.command == 'test':
        test(FLAGS)
    elif FLAGS.command == 'freeze':
        freeze(FLAGS)
    else:
        logging.error('Invalid command!')
        parser.print_help()


if __name__ == '__main__':
    main()
