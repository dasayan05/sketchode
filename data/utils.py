import torch
import numpy as np
from scipy.special import comb as choose

from noise import pnoise2
from torchcde import natural_cubic_coeffs, CubicSpline


def resample(seq: torch.Tensor, timestamps, granularity):
    # TODO: A bit wierd that passing 't=timestamps' in below function doesn't work properly.
    spline = CubicSpline(coeffs=natural_cubic_coeffs(seq.unsqueeze(0)))
    # resampling will now be uniform in time
    new_t = torch.linspace(spline.interval[0], spline.interval[-1], granularity)
    new_stroke = spline.evaluate(new_t).squeeze()
    return new_stroke, torch.linspace(timestamps[0], timestamps[-1], granularity)


def continuous_noise(stroke: np.ndarray, seed=0, noise_level=0.3):
    '''
    Given stroke is used as seed to generate a continuous noise-stroke
    and added to the original stroke; used as a part of augmentation.
    Implementation uses Perlin noise.
    '''

    if noise_level == 0.:
        return stroke

    noise_on_stroke = np.zeros_like(stroke)
    stroke_ = stroke + seed
    for i in range(len(stroke)):
        n1 = pnoise2(*stroke_[i, ...] + 5)
        n2 = pnoise2(*stroke_[i, ...] - 5)
        noise_on_stroke[i, ...] = [n1, n2]

    noise_on_stroke = noise_on_stroke - noise_on_stroke.mean(0)
    return noise_on_stroke * noise_level + stroke


def discrete_noise(stroke: np.ndarray, seed=0, noise_level=0.3):
    '''Standard random gaussian jittering. Independently applied on each point.'''

    if noise_level == 0.:
        return stroke

    old_state = np.random.get_state()
    np.random.seed(seed)
    stroke = stroke + np.random.rand(*stroke.shape) * noise_level
    np.random.set_state(old_state)
    return stroke


def draw_bezier(ctrlPoints, nPointsCurve=100):
    '''
    Draws a Bezier curve with given control points.

    ctrlPoints: shape (n+1, 2) matrix containing all control points
    nPointsCurve: granularity of the Bezier curve
    '''

    def bezier_matrix(degree):
        m = degree
        Q = np.zeros((degree + 1, degree + 1))
        for i in range(degree + 1):
            for j in range(degree + 1):
                if (0 <= (i+j)) and ((i+j) <= degree):
                    Q[i, j] = choose(m, j) * choose(m-j, m-i-j) * ((-1)**(m-i-j))
        return Q

    def T(ts: np.ndarray, d: int):
        # 'ts' is a vector (np.array) of time points
        ts = ts[..., np.newaxis]
        Q = tuple(ts**n for n in range(d, -1, -1))
        return np.concatenate(Q, 1)

    nCtrlPoints, _ = ctrlPoints.shape

    ts = np.linspace(0., 1., num=nPointsCurve)

    curve = np.matmul(T(ts, nCtrlPoints - 1), bezier_matrix(nCtrlPoints-1) @ ctrlPoints)

    return curve
