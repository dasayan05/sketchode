import os
from enum import Enum
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningDataModule

from data.batching import BatchedStrokes
from data.synth import SynthChar, SynthCharsetOneSeq, SynthCharsetStrokewise
from data.qd import QDSketchPenState, QDSketchStrokewise


class ReprType(str, Enum):
    ONESEQ = "oneseq"
    STROKEWISE = "strokewise"


class GenericDM(LightningDataModule):

    def __init__(self, split_seed, split_fraction, batch_size, num_worker, repr):
        super().__init__()

        self.split_seed = split_seed
        self.split_fraction = split_fraction
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.repr = repr

        # subclasses need to set this with a 'Dataset' instance
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError(f'Subclass {self.__class__.__name__} is yet to assign a Dataset')
        else:
            return self._dataset

    @dataset.setter
    def dataset(self, d):
        if not isinstance(d, Dataset):
            raise ValueError(f'Expected a Dataset, got {d}')
        else:
            self._dataset = d

    def compute_split_size(self):
        self.train_len = int(len(self.dataset) * self.split_fraction)
        self.valid_len = len(self.dataset) - self.train_len

    def setup(self, stage: str):
        self.train_dataset, self.valid_dataset = \
            random_split(self.dataset, [self.train_len, self.valid_len],
                         torch.Generator().manual_seed(self.split_seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=True,
                          num_workers=self.num_worker, collate_fn=self.dataset.__class__.collate)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=False,
                          num_workers=self.num_worker, collate_fn=self.dataset.__class__.collate)

    def test_dataloader(self):
        return self.val_dataloader()

    def transfer_batch_to_device(self, batch: BatchedStrokes,
                                 device: torch.device, dataloader_idx: int):
        if self.repr == ReprType.ONESEQ:
            batch.to_device(device)
        else:
            lens, batch_list = batch
            lens = lens.to(device)
            for b in batch_list:
                b.to_device(device)
            batch = lens, batch_list

        return batch


class VectorMNISTDM(GenericDM):

    def __init__(self,
                 root_dir: str,
                 samples_each: int = 25,
                 split_fraction: float = 0.8,
                 perlin_noise: float = 0.2,
                 granularity: int = 100,
                 split_seed: int = 1234,
                 batch_size: int = 64,
                 num_workers: int = os.cpu_count() // 2,
                 repr: ReprType = ReprType.ONESEQ
                 ):
        """VectorMNIST Datamodule consisting of synthetic MNIST Digits

        Args:
            root_dir: Root directory of the dataset (containing .pkl files in subfolders)
            samples_each: Number of augmented samples from one real sample
            split_fraction: Train/Validation split fraction
            perlin_noise: Strength of the Perlin noise for augmentation
            granularity: Number of points in each sample
            split_seed: Data splitting seed
            batch_size: Batch size for training
            repr: data representation (oneseq or strokewise)
        """
        self.save_hyperparameters()
        self.hp = self.hparams  # an easier name
        super().__init__(self.hp.split_seed,
                         self.hp.split_fraction,
                         self.hp.batch_size,
                         self.hp.num_workers,
                         self.hp.repr)

        self._construct()

    def _construct(self):
        synthchars = []

        print('Constructing VectorMNIST, please wait ..')
        for root, _, files in os.walk(self.hp.root_dir):
            for file in files:
                if file.endswith('.pkl'):
                    full_path = os.path.join(root, file)
                    synthchars.append(
                        SynthChar(full_path,
                                  total_samples=self.hp.samples_each,
                                  perlin_noise=self.hp.perlin_noise)
                    )

        if self.hp.repr == ReprType.ONESEQ:
            self.dataset = SynthCharsetOneSeq(synthchars,
                                              sketch_granularity=self.hp.granularity, cache=True)
        else:
            self.dataset = SynthCharsetStrokewise(synthchars,
                                                  stroke_granularity=self.hp.granularity, cache=True)

        self.compute_split_size()

        print(f'VectorMNIST constructed with {len(self.dataset)} samples,',
              f'{self.hp.samples_each} samples each',
              f'perline noise {self.hp.perlin_noise}.')


class QuickDrawDM(GenericDM):

    def __init__(self,
                 root_dir: str,
                 category: str,
                 max_sketches: int = 50000,
                 max_strokes: Optional[int] = None,
                 split_fraction: float = 0.85,
                 perlin_noise: float = 0.2,
                 granularity: int = 100,
                 split_seed: int = 4321,
                 batch_size: int = 64,
                 num_workers: int = os.cpu_count() // 2,
                 rdp: Optional[float] = None,
                 repr: ReprType = ReprType.ONESEQ
                 ):
        """QuickDraw Datamodule (OneSeq)

        Args:
            root_dir: Root directory of QD data (unpacked by `unpack_ndjson.py` utility)
            category: QD category name
            max_sketches: Maximum number of sketches to use
            max_strokes: clamp the maximum number of strokes (None for all strokes)
            split_fraction: Train/Validation split fraction
            perlin_noise: Strength of Perlin noise (YET TO BE IMPL)
            granularity: Number of points in each sample
            split_seed: Data splitting seed
            batch_size: Batch size for training
            rdp: RDP algorithm parameter ('None' to ignore RDP entirely)
            repr: data representation (oneseq or strokewise)
        """
        self.save_hyperparameters()
        self.hp = self.hparams  # an easier name
        super().__init__(self.hp.split_seed,
                         self.hp.split_fraction,
                         self.hp.batch_size,
                         self.hp.num_workers,
                         self.hp.repr)

        self._construct()

    def _construct(self):
        if self.hp.repr == ReprType.ONESEQ:
            self.dataset = QDSketchPenState(self.hp.root_dir, self.hp.category,
                                            sketch_granularity=self.hp.granularity,
                                            perlin_noise=self.hp.perlin_noise,
                                            max_sketches=self.hp.max_sketches,
                                            max_strokes=self.hp.max_strokes,
                                            rdp=self.hp.rdp)
        else:
            self.dataset = QDSketchStrokewise(self.hp.root_dir, self.hp.category,
                                              stroke_granularity=self.hp.granularity,
                                              perlin_noise=self.hp.perlin_noise,
                                              max_sketches=self.hp.max_sketches,
                                              max_strokes=self.hp.max_strokes,
                                              rdp=self.hp.rdp)
        self.compute_split_size()
