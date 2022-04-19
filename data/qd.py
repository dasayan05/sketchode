import os
import sys
import random
import pickle

from torch.utils.data import Dataset, DataLoader

from data.sketch import Sketch
from data.batching import OneSeq, Strokewise


class QuickDraw(Dataset):

    def __init__(self, data_root, category, train=True, shuffle=True, perlin_noise=0.2, max_sketches=10000, max_strokes=None, rdp=None):
        super().__init__()

        self.data_root = data_root
        self.category = category
        self.max_sketches = max_sketches
        self.max_strokes = max_strokes
        self.perlin_noise = perlin_noise
        self.rdp = rdp
        self.split = 'train' if train else 'test'
        self.data_path = os.path.join(self.data_root, self.category, self.split)

        self.file_list = os.listdir(self.data_path)
        self.max_sketches = min(self.max_sketches, len(self.file_list))
        del self.file_list[self.max_sketches:]

        if shuffle:
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        file_path = os.path.join(self.data_path, self.file_list[i])

        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

            stroke_list = self.data['drawing']

            if self.max_strokes is not None:
                max_strokes = min(self.max_strokes, len(stroke_list))
                stroke_list = stroke_list[:max_strokes]

            sketch = Sketch(stroke_list)
            sketch.move()
            sketch.shift_time(0)
            sketch.scale_spatial(1)
            if self.rdp is not None:
                sketch.rdp(eps=self.rdp)
            sketch.scale_time(1)

            seed = random.randint(0, 10000)
            sketch.jitter(seed=seed, noise_level=self.perlin_noise)

        return sketch


class QDOneStroke(QuickDraw):

    def __init__(self, sketch_no, stroke_no, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sketch_no = sketch_no
        self.stroke_no = stroke_no

    def __len__(self):
        return 1

    def __getitem__(self, i):
        sketch = super().__getitem__(self.sketch_no)
        stroke = sketch[self.stroke_no]

        if self.stroke_granularity > 0:
            stroke.resample(self.stroke_granularity)

        stroke.move(by=-stroke.start)
        stroke.shift_time(to=0.)
        stroke.scale_time()

        poly_stroke, time_range = stroke.tensorize()
        start = poly_stroke[0, :]

        return (start, time_range), poly_stroke


class QDSketchPenState(QuickDraw, OneSeq):

    def __init__(self, *args, sketch_granularity=300, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        OneSeq.__init__(self, granularity=sketch_granularity, cache=False)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i):
        return self.cached_represent(super().__getitem__(i), index=i)


class QDSketchStrokewise(QuickDraw, Strokewise):

    def __init__(self, *args, stroke_granularity, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        Strokewise.__init__(self, granularity=stroke_granularity, cache=False)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i):
        return self.cached_represent(super().__getitem__(i), index=i)


if __name__ == '__main__':
    qd = QDSketchStrokewise(sys.argv[1], sys.argv[2], stroke_granularity=30, max_sketches=1000)
    qdl = DataLoader(qd, batch_size=32, shuffle=True,
                     pin_memory=True, collate_fn=qd.__class__.collate)
    for lens, batch in qdl:
        print(lens)
