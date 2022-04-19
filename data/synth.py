import os
import sys
import pickle
import tqdm

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from data.sketch import Sketch
from data.batching import OneSeq, Strokewise
from data.utils import continuous_noise


class SynthCharBase(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.samples = []

    def visualize(self, grid=5):
        # create a 'grid x grid' sized grid
        fig, ax = plt.subplots(grid, grid, figsize=(grid, grid))
        for r in range(grid):
            for c in range(grid):
                sample = self.samples[r*grid+c]
                sample.draw(ax[r, c])
                ax[r, c].axis('off')
        plt.show()


class SynthChar(SynthCharBase):

    def __init__(self, anchor_file, total_samples=32, perlin_noise=0., **kwargs):
        super().__init__(**kwargs)

        self.anchor_file = anchor_file
        self.total_samples = total_samples
        self.perlin_noise = perlin_noise  # None or float

        # load the anchor
        with open(self.anchor_file, 'rb') as f:
            sketch = pickle.load(f)

        for i in range(self.total_samples):
            char = []
            seed = np.random.randint(0, 10000)
            last_stroke_end_time = None
            last_stroke_duration = None
            for j, curve in enumerate(sketch):
                curve = continuous_noise(curve,
                                         seed=seed,
                                         noise_level=self.perlin_noise * (i / self.total_samples))
                # create timestamps according to consecutive distance
                consecutive_dist = np.sqrt(((curve[1:, :]-curve[:-1, :]) ** 2).sum(-1))
                if last_stroke_end_time is not None:
                    penup_duration = last_stroke_duration / 3.
                    current_stroke_start_time = last_stroke_end_time + penup_duration
                    timestamps = [current_stroke_start_time,
                                  *(np.cumsum(consecutive_dist, -1) + current_stroke_start_time)]
                else:
                    timestamps = [0., *np.cumsum(consecutive_dist, -1)]

                last_stroke_end_time = timestamps[-1]
                last_stroke_duration = timestamps[-1] - timestamps[0]
                char.append([*curve.T, timestamps])  # that's how Sketch() expects
            charsketch = Sketch(char)

            charsketch.move()
            charsketch.shift_time(0.)
            charsketch.scale_spatial(1)
            charsketch.scale_time(1)

            self.samples.append(charsketch)

    def __len__(self):
        return len(self.samples)


class SynthCharset(SynthCharBase):

    def __init__(self, chars: list, **kwargs):
        super().__init__()

        for char in chars:
            self.samples.extend(char.samples)

    def __len__(self):
        return len(self.samples)


class SynthCharOneSeq(Dataset, SynthChar, OneSeq):

    def __init__(self, **kwargs):
        Dataset.__init__(self)
        SynthChar.__init__(self, **kwargs)
        OneSeq.__init__(self, granularity=kwargs.get('sketch_granularity', 150),
                        cache=kwargs.get('cache', False))

    def __getitem__(self, i):
        return self.cached_represent(self.samples[i], index=i)


class SynthCharsetOneSeq(Dataset, SynthCharset, OneSeq):

    def __init__(self, chars: list, **kwargs):
        Dataset.__init__(self)
        SynthCharset.__init__(self, chars, **kwargs)
        OneSeq.__init__(self, granularity=kwargs.get('sketch_granularity', 150),
                        cache=kwargs.get('cache', False))

    def __getitem__(self, i):
        return self.cached_represent(self.samples[i], index=i)


class SynthCharStrokewise(Dataset, SynthChar, Strokewise):

    def __init__(self, **kwargs):
        Dataset.__init__(self)
        SynthChar.__init__(self, **kwargs)
        Strokewise.__init__(self, granularity=kwargs.get('stroke_granularity', 30),
                            cache=kwargs.get('cache', False))

    def __getitem__(self, i):
        return self.cached_represent(self.samples[i], index=i)


class SynthCharsetStrokewise(Dataset, SynthCharset, Strokewise):

    def __init__(self, chars: list, **kwargs):
        Dataset.__init__(self)
        SynthCharset.__init__(self, chars, **kwargs)
        Strokewise.__init__(self, granularity=kwargs.get('stroke_granularity', 30),
                            cache=kwargs.get('cache', False))

    def __getitem__(self, i):
        return self.cached_represent(self.samples[i], index=i)


if __name__ == '__main__':
    synthchars = []
    for root, dirs, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith('.pkl'):
                full_path = os.path.join(root, file)
                synthchars.append(
                    SynthChar(full_path, total_samples=25, perlin_noise=0.2)
                )

    vmnist = SynthCharsetStrokewise(synthchars, stroke_granularity=30,
                                    sketch_granularity=100, cache=True)
    vmnistdl = DataLoader(vmnist, batch_size=8, pin_memory=True,
                          shuffle=True, collate_fn=vmnist.__class__.collate, drop_last=True)
    for e in range(5):
        for batch in tqdm.tqdm(vmnistdl):
            pass
