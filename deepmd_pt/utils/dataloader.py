import logging
import os
import queue
import time
import numpy as np
from threading import Thread
from typing import Callable, Dict, List, Tuple, Type, Union
from multiprocessing.dummy import Pool

import h5py
import torch
import torch.distributed as dist
from deepmd_pt.utils import env
from deepmd_pt.utils.dataset import DeepmdDataSetForLoader
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DpLoaderSet(Dataset):
    """A dataset for storing DataLoaders to multiple Systems."""

    def __init__(
        self,
        systems,
        batch_size,
        model_params,
        seed=10,
        type_split=True,
        noise_settings=None,
        shuffle=True,
    ):
        setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        self.systems: List[DeepmdDataSetForLoader] = []
        if len(systems) >= 100:
            logging.info(f"Constructing DataLoaders from {len(systems)} systems")

        def construct_dataset(system):
            if model_params["descriptor"].get("type") != "hybrid":
                rcut = model_params["descriptor"]["rcut"]
                sel = model_params["descriptor"]["sel"]
            else:
                rcut = []
                sel = []
                for ii in model_params["descriptor"]["list"]:
                    rcut.append(ii["rcut"])
                    sel.append(ii["sel"])
            return DeepmdDataSetForLoader(
                system=system,
                type_map=model_params["type_map"],
                rcut=rcut,
                sel=sel,
                type_split=type_split,
                noise_settings=noise_settings,
                shuffle=shuffle,
            )

        with Pool(
            os.cpu_count()
            // (int(os.environ["LOCAL_WORLD_SIZE"]) if dist.is_initialized() else 1)
        ) as pool:
            self.systems = pool.map(construct_dataset, systems)

        self.sampler_list: List[DistributedSampler] = []
        self.index = []
        self.total_batch = 0

        self.dataloaders = []
        for system in self.systems:
            if dist.is_initialized():
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            else:
                system_sampler = None
            if isinstance(batch_size,str):
                if batch_size == "auto":
                    rule = 32
                elif batch_size.startswith("auto:"):
                    rule = int(batch_size.split(":")[1])
                else:
                    rule = None
                    logging.error(f"Unsupported batch size type")
                self.batch_size = rule // system._natoms 
                if self.batch_size * system._natoms < rule:
                    self.batch_size += 1
            else:
                self.batch_size = batch_size
            system_dataloader = DataLoader(
                dataset=system,
                batch_size=self.batch_size,
                num_workers=0,  # Should be 0 to avoid too many threads forked
                sampler=system_sampler,
                collate_fn=collate_batch,
                shuffle=(not dist.is_initialized()) and shuffle,
            )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)
        # Initialize iterator instances for DataLoader
        self.iters = []
        for item in self.dataloaders:
            self.iters.append(iter(item))

    def set_noise(self, noise_settings):
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        for system in self.systems:
            system.set_noise(noise_settings)

    def __len__(self):
        return len(self.dataloaders)

    def __getitem__(self, idx):
        #logging.warning(str(torch.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
        return next(self.iters[idx])


_sentinel = object()
QUEUESIZE = 32


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)
        self._queue = queue
        self._source = source  # Main DL iterator
        self._max_len = max_len  #

    def run(self):
        for item in self._source:
            self._queue.put(item)  # Blocking if the queue is full

        # Signal the consumer we are done.
        self._queue.put(_sentinel)


class BufferedIterator(object):
    def __init__(self, iterable):
        self._queue = queue.Queue(QUEUESIZE)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None
        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(self._queue, self._iterable, self.total)
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()
        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                    self.warning_time is None
                    or time.time() - self.warning_time > 15 * 60
                ):
                    logging.warning(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration
        return item


def collate_tensor_fn(batch):
    elem = batch[0]
    if not isinstance(elem, list):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    else:
        out_hybrid = []
        for ii, hybrid_item in enumerate(elem):
            out = None
            tmp_batch = [x[ii] for x in batch]
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in tmp_batch)
                storage = hybrid_item._typed_storage()._new_shared(numel, device=hybrid_item.device)
                out = hybrid_item.new(storage).resize_(len(tmp_batch), *list(hybrid_item.size()))
            out_hybrid.append(torch.stack(tmp_batch, 0, out=out))
        return out_hybrid


def collate_batch(batch):
    example = batch[0]
    result = example.copy()
    for key in example.keys():
        if key == "shift" or key == "mapping":
            natoms_extended = max([d[key].shape[0] for d in batch])
            n_frames = len(batch)
            list = []
            for x in range(n_frames):
                list.append(batch[x][key])
            if key == "shift":
                result[key] = torch.zeros(
                    (n_frames, natoms_extended, 3),
                    dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                    device=env.PREPROCESS_DEVICE,
                )
            else:
                result[key] = torch.zeros(
                    (n_frames, natoms_extended),
                    dtype=torch.long,
                    device=env.PREPROCESS_DEVICE,
                )
            for i in range(len(batch)):
                natoms_tmp = list[i].shape[0]
                result[key][i, :natoms_tmp] = list[i]
        elif "find_" in key:
            result[key] = batch[0][key]
        else:
            if batch[0][key] is None:
                result[key] = None
            else:
                result[key] = collate_tensor_fn([d[key] for d in batch])
    return result

def get_weighted_sampler(training_data,prob_style,sys_prob=False):
    if sys_prob == False:
        if prob_style == "prob_uniform":
            prob_v = 1.0 / float(training_data.__len__())
            probs = [prob_v for ii in range(training_data.__len__())]
        elif prob_style == "prob_sys_size":
            probs = []
            for ii in range(len(training_data.dataloaders)):
                prob_v = float(training_data.index[ii]) / float(training_data.total_batch)
                probs.append(prob_v)
        else:#prob_sys_size;A:B:p1;C:D:p2
            probs = prob_sys_size_ext(prob_style,len(training_data),training_data.index)
    else:
        probs = process_sys_probs(prob_style,training_data.index)
    logging.info("Generated weighted sampler with prob array: "+str(probs))
    #training_data.total_batch is the size of one epoch, you can increase it to avoid too many  rebuilding of iteraters
    sampler = WeightedRandomSampler(probs,training_data.total_batch, replacement = True)
    return sampler

def prob_sys_size_ext(keywords,nsystems,nbatch):
    block_str = keywords.split(";")[1:]
    block_stt = []
    block_end = []
    block_weights = []
    for ii in block_str:
        stt = int(ii.split(":")[0])
        end = int(ii.split(":")[1])
        weight = float(ii.split(":")[2])
        assert weight >= 0, "the weight of a block should be no less than 0"
        block_stt.append(stt)
        block_end.append(end)
        block_weights.append(weight)
    nblocks = len(block_str)
    block_probs = np.array(block_weights) / np.sum(block_weights)
    sys_probs = np.zeros([nsystems])
    for ii in range(nblocks):
        nbatch_block = nbatch[block_stt[ii] : block_end[ii]]
        tmp_prob = [float(i) for i in nbatch_block] / np.sum(nbatch_block)
        sys_probs[block_stt[ii] : block_end[ii]] = tmp_prob * block_probs[ii]
    return sys_probs
def process_sys_probs(sys_probs,nbatch):
        sys_probs = np.array(sys_probs)
        type_filter = sys_probs >= 0
        assigned_sum_prob = np.sum(type_filter * sys_probs)
        # 1e-8 is to handle floating point error; See #1917
        assert (
            assigned_sum_prob <= 1.0 + 1e-8
        ), "the sum of assigned probability should be less than 1"
        rest_sum_prob = 1.0 - assigned_sum_prob
        if not np.isclose(rest_sum_prob, 0):
            rest_nbatch = (1 - type_filter) * nbatch
            rest_prob = rest_sum_prob * rest_nbatch / np.sum(rest_nbatch)
            ret_prob = rest_prob + type_filter * sys_probs
        else:
            ret_prob = sys_probs
        assert np.isclose(np.sum(ret_prob), 1), "sum of probs should be 1"
        return ret_prob