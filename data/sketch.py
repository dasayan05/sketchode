import torch
import numpy as np
import matplotlib.pyplot as plt
from simplification.cutil import simplify_coords_idx

from data.utils import continuous_noise, resample


class Stroke(object):
    def __init__(self, stroke, timestamps=None):
        super().__init__()

        self.type = type(stroke)
        if self.type in [np.ndarray, torch.Tensor]:
            self.stroke = stroke
            assert isinstance(timestamps, self.type), \
                "stroke & timestamps must have same type"
            self.timestamps = timestamps

        if self.timestamps.ndim != 1:
            raise AssertionError('timestamps must be 1D array')

    def draw(self, axis=None):
        raise NotImplementedError('Use one of the subclasses of Stroke')

    def __len__(self):
        return self.stroke.shape[0]

    def tensorize(self):
        if self.type is torch.Tensor:
            return self.stroke, self.timestamps
        else:
            return torch.from_numpy(self.stroke.astype(np.float32)), \
                torch.from_numpy(self.timestamps.astype(np.float32))

    def draw(self, axis=None, cla=False, color='black', direction=True, xlim=None, ylim=None):
        if axis is None:
            fig = plt.figure()
            axis = plt.gca()

        if cla:
            axis.cla()

        stroke = self.stroke.data.cpu().numpy() if (self.type is torch.Tensor) else self.stroke
        axis.plot(stroke[:, 0], stroke[:, 1], color=color)

        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()
        maxlim = max(xmax, ymax)
        minlim = min(xmin, ymin)

        if xlim is not None:
            axis.set_xlim(xlim)
        else:
            axis.set_xlim([minlim, maxlim])
        if ylim is not None:
            axis.set_ylim(ylim)
        else:
            axis.set_ylim([minlim, maxlim])

        if direction:
            # add a little arrow
            m = len(self) // 2
            x, y = self.stroke[m, :]
            dx, dy = self.stroke[m + 1, :] - self.stroke[m, :]
            xmin, xmax = axis.get_xlim()
            ymin, ymax = axis.get_ylim()
            w = min(xmax - xmin, ymax - ymin)
            axis.arrow(x.item(), y.item(), dx.item(), dy.item(),
                       length_includes_head=True, color=color, width=1e-2 * w)

        return axis


class PolylineStroke(Stroke):
    def __init__(self, stroke, timestamps=None):

        stroke = np.array(stroke).T if isinstance(stroke, list) else stroke
        timestamps = np.array(timestamps) if isinstance(timestamps, list) else timestamps
        super().__init__(stroke, timestamps)

    def rdp(self, eps=0.01):
        is_tensor = isinstance(self.stroke, torch.Tensor)
        stroke = self.stroke.data.cpu().numpy() if is_tensor else self.stroke
        stroke = np.ascontiguousarray(stroke)
        simpl_idx = simplify_coords_idx(stroke, eps)

        self.stroke = self.stroke[simpl_idx]
        self.timestamps = self.timestamps[simpl_idx]

    def resample(self, granularity):
        # TODO: Don't convert numpy-to-Tensor and Tensor-to-numpy
        # Try finding something in native numpy for interpolation
        stroke = torch.from_numpy(self.stroke) if (self.type is not torch.Tensor) else self.stroke
        timestamps = torch.from_numpy(self.timestamps) \
            if (self.type is not torch.Tensor) else self.timestamps

        new_stroke, new_t = resample(stroke, timestamps, granularity)

        self.stroke = new_stroke.numpy() if (self.type is not torch.Tensor) else new_stroke
        self.timestamps = new_t.numpy() if (self.type is not torch.Tensor) else new_t

    def jitter(self, seed, noise_level=0.2):
        stroke = self.stroke.numpy() if (self.type is torch.Tensor) else self.stroke
        self.stroke = continuous_noise(stroke, seed=seed, noise_level=noise_level)

    def move(self, by=np.zeros((1, 2))):
        self.stroke = self.stroke + by

    def shift_time(self, to=0.):
        self.timestamps = self.timestamps - self.initial_time + to

    def scale_time(self, factor=1.):
        self.timestamps = (self.timestamps / self.terminal_time) * factor

    @property
    def initial_time(self):
        return self.timestamps[0]

    @property
    def terminal_time(self):
        return self.timestamps[-1]

    @property
    def start(self):
        return self.stroke[0, :]

    @property
    def end(self):
        return self.stroke[-1, :]

    def draw(self, axis=None, cla=False, color='black', direction=False, scatter=True, xlim=None, ylim=None):
        axis = super().draw(axis, cla, direction=direction, xlim=xlim, ylim=ylim, color=color)

        if scatter:
            stroke = self.stroke.data.cpu().numpy() if (self.type is torch.Tensor) else self.stroke
            axis.scatter(stroke[:, 0], stroke[:, 1], color=color, s=5)

    @property
    def enclosing_circle_radius(self):
        norms = np.linalg.norm(self.stroke, 2, -1)
        return norms.max()


class Sketch(object):

    def __init__(self, strokes):
        super().__init__()

        self.strokes = []
        for s in strokes:
            stroke = PolylineStroke(s[:2], s[-1])
            if len(stroke) > 1:
                # one point strokes are not tolerable
                self.strokes.append(stroke)

    @property
    def nstrokes(self):
        return len(self.strokes)

    def __len__(self):
        return self.nstrokes

    def rdp(self, eps=0.01):
        for stroke in self.strokes:
            stroke.rdp(eps)

    def move(self, to=np.zeros((1, 2))):
        move_by = to - self.strokes[0].start
        for stroke in self.strokes:
            stroke.move(move_by)

    def __getitem__(self, i):
        return self.strokes[i]

    def draw(self, axis=None, **kwargs):
        if axis is None:
            fig = plt.figure()
            axis = plt.gca()

        for stroke in self.strokes:
            stroke.draw(axis, **kwargs)

        axis.autoscale()

    @property
    def terminal_time(self):
        return self.strokes[-1].terminal_time

    @property
    def initial_time(self):
        return self.strokes[0].initial_time

    def shift_time(self, to=0.):
        initial_time = self.initial_time
        for stroke in self.strokes:
            stroke.timestamps = stroke.timestamps - initial_time

    def scale_time(self, factor=1.):
        for stroke in self.strokes:
            stroke.timestamps = (stroke.timestamps / self.terminal_time) * factor

    def scale_spatial(self, factor=1.):
        enclosing_circle_radius = max([stroke.enclosing_circle_radius for stroke in self.strokes])
        for stroke in self.strokes:
            stroke.stroke = (stroke.stroke / enclosing_circle_radius) * factor

    def jitter(self, seed, noise_level=0.2):
        for i, stroke in enumerate(self.strokes):
            stroke.jitter(seed + i, noise_level)

    def _fill_penup(start, end, granularity):
        start = start.unsqueeze(0).repeat(granularity, 1)
        end = end.unsqueeze(0).repeat(granularity, 1)
        alpha = torch.linspace(0., 1., granularity).unsqueeze(-1)
        stroke = start * (1. - alpha) + end * alpha
        return stroke

    def _add_pen_state(stroke, fill_value=0.):
        stroke_plus_pen = torch.cat([
            stroke,
            torch.ones(len(stroke), 1, device=stroke.device) * fill_value
        ], dim=-1)
        return stroke_plus_pen

    def tensorize(self, joining_granularity=20):
        seq_strokes, seq_timestamps = [], []

        current_stroke, current_timestamps = self[0].tensorize()
        seq_strokes.append(Sketch._add_pen_state(current_stroke, 0.))
        seq_timestamps.append(current_timestamps)

        for i in range(1, self.nstrokes):
            next_stroke, next_timestamps = self[i].tensorize()
            joining_stroke = Sketch._fill_penup(current_stroke[-1, :], next_stroke[0, :],
                                                granularity=joining_granularity)
            joining_stroke_pen = Sketch._add_pen_state(joining_stroke, 1.)
            joining_timestamps = torch.linspace(current_timestamps[-1], next_timestamps[0], len(joining_stroke_pen),
                                                device=joining_stroke_pen.device)
            # ignore the first and last one to avoid duplication
            seq_strokes.append(joining_stroke_pen[1:-1, ...])
            seq_timestamps.append(joining_timestamps[1:-1])

            next_stroke_pen = Sketch._add_pen_state(next_stroke, 0.)
            seq_strokes.append(next_stroke_pen)
            seq_timestamps.append(next_timestamps)

            current_stroke, current_timestamps = next_stroke, next_timestamps

        return torch.cat(seq_strokes, 0), torch.cat(seq_timestamps, 0)
