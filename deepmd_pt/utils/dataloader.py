import logging
import os
import queue
import time
from threading import Thread
from typing import List

import h5py
import torch
import torch.distributed as dist
from deepmd_pt.utils import env
from deepmd_pt.utils.dataset import DeepmdDataSetForLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


class DpLoaderSet(Dataset):
    """A dataset for storing DataLoaders to multiple Systems."""

    def __init__(self, systems, batch_size, model_params):
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        self.systems: List[DeepmdDataSetForLoader] = []
        for system in systems:
            ds = DeepmdDataSetForLoader(
                system=system,
                type_map=model_params["type_map"],
                rcut=model_params["descriptor"]["rcut"],
                sel=model_params["descriptor"]["sel"],
            )
            self.systems.append(ds)
        self.sampler_list: List[DistributedSampler] = []
        self.index = []

        self.dataloaders = []
        for system in self.systems:
            if dist.is_initialized():
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            else:
                system_sampler = None
            system_dataloader = DataLoader(
                dataset=system,
                batch_size=batch_size,
                num_workers=0,  # Should be 0 to avoid too many threads forked
                sampler=system_sampler,
                collate_fn=collate_batch,
                shuffle=(not dist.is_initialized()),
            )
            self.dataloaders.append(system_dataloader)
            for _ in range(len(system_dataloader)):
                self.index.append(len(self.dataloaders) - 1)

        # Initialize iterator instances for DataLoader
        self.iters = []
        for item in self.dataloaders:
            self.iters.append(iter(item))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # logging.warning(str(torch.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
        return next(self.iters[self.index[idx]])


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
            self._queue.put(item) # Blocking if the queue is full

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
                    logging.debug(
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


def collate_batch(batch):
    batch = default_collate(batch)
    shift = batch["shift"]
    mapping = batch["mapping"]
    natoms_extended = max([item.shape[0] for item in shift])
    n_frames = len(shift)
    batch["shift"] = torch.zeros(
        (n_frames, natoms_extended, 3),
        dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        device=env.PREPROCESS_DEVICE,
    )
    batch["mapping"] = torch.zeros(
        (n_frames, natoms_extended), dtype=torch.long, device=env.PREPROCESS_DEVICE
    )
    for i in range(len(shift)):
        natoms_tmp = shift[i].shape[0]
        batch["shift"][i, :natoms_tmp] = shift[i]
        batch["mapping"][i, :natoms_tmp] = mapping[i]
    return batch
