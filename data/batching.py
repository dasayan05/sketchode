import torch
import numpy as np
from matplotlib.cm import get_cmap
from torch.nn.utils.rnn import pad_sequence

from data.utils import resample
from data.sketch import Sketch


class SketchRepr(object):

    def __init__(self, stroke_granularity=None, sketch_granularity=None, cache=False):
        super().__init__()

        self.stroke_granularity = stroke_granularity
        self.sketch_granularity = sketch_granularity
        self.cache = cache

        if self.cache:
            self.cached_data = dict()

    def represent(self, sketch):
        raise NotImplementedError('Abstract method not callable')

    def collate(self, batch: list):
        raise NotImplementedError('Abstract method not callable')

    def cached_represent(self, sketch: Sketch, index):
        if not self.cache:
            return self.represent(sketch)
        else:
            # caching enabled
            if index in self.cached_data.keys():
                return self.cached_data[index]
            else:
                repr = self.represent(sketch)
                self.cached_data[index] = repr
                return repr


class OneSeq(SketchRepr):

    def __init__(self, granularity=None, cache=False):
        # Here 'granularity' means sketch-granularity
        super().__init__(stroke_granularity=None, sketch_granularity=granularity, cache=cache)

    def represent(self, sketch: Sketch):
        seq_sketch, seq_timestamps = sketch.tensorize()

        seq_sketch, seq_timestamps = resample(seq_sketch, seq_timestamps, self.sketch_granularity)

        # pen-states messed up due to resampling; simple thresholding would fix it
        soft_zero_idx = seq_sketch[:, -1] < 0.5
        soft_one_idx = seq_sketch[:, -1] >= 0.5
        seq_sketch[:, -1][soft_zero_idx] = 0.
        seq_sketch[:, -1][soft_one_idx] = 1.

        return {
            "start": seq_sketch[0, :],
            "time_range": seq_timestamps,
            "poly_stroke": seq_sketch
        }

    def collate(batch: list):
        return BatchedStrokes(batch)


class Strokewise(SketchRepr):

    def __init__(self, granularity=None, cache=False):
        # Here 'granularity' means stroke-granularity
        super().__init__(stroke_granularity=granularity, sketch_granularity=None, cache=cache)

    def represent(self, sketch: Sketch):
        sk_repr = []

        for stroke in sketch:
            stroke.resample(self.stroke_granularity)
            seq_stroke, _ = stroke.tensorize()
            sk_repr.append({
                'start': seq_stroke[0, :],
                'time_range': torch.linspace(0., 1., self.stroke_granularity, device=seq_stroke.device),
                'poly_stroke': seq_stroke
            })

        return sk_repr

    def collate(batch: list):
        lens = [len(sketch) for sketch in batch]
        maxlen = max(lens)  # maximum number of strokes in a sketch

        # cache a filler stroke with the help of the first stroke
        # well, it could be any stroke TBH
        first_stroke = batch[0][0]
        filler_stroke = {
            'start': torch.zeros_like(first_stroke['start']),
            'time_range': first_stroke['time_range'].clone(),
            'poly_stroke': torch.zeros_like(first_stroke['poly_stroke']),
        }

        batchedstroke_list = []
        for i in range(maxlen):
            stroke_batch = []
            for sketch in batch:
                try:
                    stroke_i = sketch[i]
                except IndexError:
                    stroke_i = filler_stroke
                finally:
                    stroke_batch.append(stroke_i)
            stroke_batch = BatchedStrokes(stroke_batch)
            batchedstroke_list.append(stroke_batch)

        return torch.tensor(lens), batchedstroke_list


class BatchedStrokes(object):

    def __init__(self, batch):
        # constructs a smart object encapsulating naively
        # batched strokes similar to pytorch's 'PackedSequence'
        super().__init__()

        # hold the batch
        self.batch = batch
        self.batch_size = len(self.batch)
        self.x0 = torch.stack([sample['start'] for sample in self.batch], 0)

        self.lens, self.t_union = [], []
        for sample in self.batch:
            time_range = sample['time_range']
            self.lens.append(time_range.shape[0])
            self.t_union.append(time_range)
        self.lens = torch.tensor(self.lens)
        self.t_union = BatchedStrokes.__merge_sort(self.t_union)

        self.t_indecies = []
        for sample in self.batch:
            p = self.t_union.unsqueeze(0) == sample['time_range'].unsqueeze(1)
            # time indecies of this sample wrt the union
            _, pidx = torch.nonzero(p, as_tuple=True)
            self.t_indecies.append(pidx)

        self.t_indecies = pad_sequence(self.t_indecies,
                                       batch_first=False, padding_value=0)
        self.points = pad_sequence([sample['poly_stroke'] for sample in self.batch],
                                   batch_first=False, padding_value=0.)

    @property
    def points_wandb3d(self):
        xy_coords = self.points[..., :-1]
        pens = self.points[..., -1].unsqueeze(-1)
        z_coord = torch.zeros_like(pens)
        return torch.cat([xy_coords, z_coord, pens], -1)

    def __len__(self):
        return self.batch_size

    def __merge_sort(ts):
        ts = torch.cat(ts, 0)
        ts_unique = ts.unique()
        ts_sorted, _ = ts_unique.sort()
        return ts_sorted

    def to_device(self, device):
        self.x0 = self.x0.to(device)
        self.t_indecies = self.t_indecies.to(device)
        self.lens = self.lens.to(device)
        self.t_union = self.t_union.to(device)
        self.points = self.points.to(device)
        return

    def from_prediction(batched_traj_x, batched_traj_p, ts=None):
        # construct a BatchedStrokes() from prediction
        # Note: shape should be (length, batch_size, feature)
        # feature=2 for x and =1 for p
        length, _, _ = batched_traj_x.shape

        # if they are already thresholded, it won't affect anything
        batched_traj_p[batched_traj_p <= .5] = 0.
        batched_traj_p[batched_traj_p > .5] = 1.
        batched_traj_p = batched_traj_p.unsqueeze(-1)

        points = torch.cat([batched_traj_x, batched_traj_p], -1)
        # x0 = points[0, ...]
        t_union = ts if (ts is not None) else torch.linspace(0., 1., length, device=points.device)

        dummy_batch = [
            {
                'start': sample[0, :],
                'poly_stroke': sample,
                'time_range': t_union
            }
            for sample in torch.unbind(points, 1)
        ]
        return BatchedStrokes(dummy_batch)

    def from_prediction_strokewise(batched_strokes_x, batched_stroke_p):
        outx, outp = batched_strokes_x, batched_stroke_p
        outp = torch.stack([p.argmax(1) for p in outp], -1)

        samples = []
        for i_sample, stroke_pen in enumerate(outp):
            sample, pen = [], []
            for i_stroke, p in enumerate(stroke_pen):
                sample.append(outx[i_stroke][:, i_sample, :].unsqueeze(1))
                dummy_pen = torch.zeros(*sample[-1].shape[:-1], dtype=sample[-1].dtype,
                                        device=sample[-1].device)
                dummy_pen[-1, 0] = 1.
                pen.append(dummy_pen)
                if p == 1.:
                    break
            sample = torch.cat(sample, 0)  # along seq length
            pen = torch.cat(pen, 0)
            samples.append(BatchedStrokes.from_prediction(sample, pen))

        return samples

    def __getitem__(self, i):
        return BatchedStrokes([
            {
                'start': self.points[0, i, :],
                'poly_stroke': self.points[:, i, :],
                'time_range': self.t_union
            }
        ])

    def draw(self, axes, lims=None, colormap='viridis', scatter=True):
        # axes must be a grid of 'AxesSubplot's
        # 'lims' is [low, high] limit for both axis; if None,
        # it will wisely make it square
        if not isinstance(axes, np.ndarray):
            # if only one axis, then wrap it like subplot axes
            axes = np.array([axes])

        cm = get_cmap(colormap)

        for g, (ax, data) in enumerate(zip(axes.ravel(), torch.unbind(self.points.cpu(), 1))):
            data = data.numpy()

            if self.points.shape[-1] == 3:  # with pen state (i.e. oneseq)
                curve, pen = data[:, :-1], data[:, -1]
            else:  # (== 2) without pen state (i.e. strokewise)
                curve = data[:, :]
                pen = np.zeros((data.shape[0],), dtype=curve.dtype)

            L = curve.shape[0] - 1
            for i in range(L):
                if pen[i] == 0:
                    ax.plot(curve[i:i+2, 0], curve[i:i+2, 1], color=cm(i/(L-1)))

            if scatter:
                ax.scatter(curve[:, 0], curve[:, 1], s=(1.-pen)*2,  # '0' means touch state
                           c=self.t_union.cpu().numpy(), cmap=cm)  # good looking colormap

            if lims is None:
                (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
                xymin, xymax = min(xmin, ymin), max(xmax, ymax)
            else:
                xymin, xymax = lims

            ax.set_xlim(xymin, xymax)
            ax.set_ylim(xymin, xymax)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_frame_on(False)
