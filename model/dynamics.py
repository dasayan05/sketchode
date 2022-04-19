import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cnf import second, logp


class Cosine(nn.Module):
    def __init__(self, omega=3., fixed_freq=True, fn='sin'):
        super().__init__()

        # if false, it is learned and 'omega' becomes maximum freq
        self.fixed_freq = fixed_freq
        self.test_freq = None
        self.fn = fn

        if self.fixed_freq:
            self.freq = omega
        else:
            self.freq = nn.Parameter(torch.tensor(1., requires_grad=True))
            self.max_omega = omega

    @property
    def omega(self):
        if self.fixed_freq:
            if (not self.training) and self.test_freq is not None:
                return self.test_freq
            return self.freq
        else:
            if (not self.training) and self.test_freq is not None:
                freq_scale = self.test_freq
            else:
                freq_scale = 1.

            return torch.sigmoid(self.freq) * self.max_omega * freq_scale

    def forward(self, x):
        if self.fn == 'cos':
            fn = torch.cos
        elif self.fn == 'sin':
            fn = torch.sin
        elif self.fn == 'relu':
            fn = torch.relu
        elif self.fn == 'tanh':
            fn = torch.tanh
        else:
            raise TypeError('Only sin/cos/relu/tanh allowed as activation function')

        if self.fn in ['sin', 'cos']:
            return fn(self.omega * x)
        else:
            return fn(x)


class Dynamics(nn.Module):
    def __init__(self, n_dyn_inp, n_dyn_out, hiddens: list,
                 bias=True, fixed_freq=True, freq=3., time_dependent=False,
                 periodicity='cos'):
        super().__init__()

        self.n_dyn_inp = n_dyn_inp
        self.n_dyn_out = n_dyn_out
        self.bias = bias
        self.fixed_freq, self.freq = fixed_freq, freq
        self.hiddens = hiddens
        # Dynamics dxdt = F(x, t) is time dependent
        self.time_dependent = time_dependent
        self.periodicity = periodicity

        self.net = []
        # 1 extra dimension for time
        last = self.n_dyn_inp + (1 if self.time_dependent else 0)
        for h in self.hiddens:
            self.net.extend([
                nn.Linear(last, h, bias=self.bias),
                Cosine(self.freq, self.fixed_freq, fn=self.periodicity)
            ])
            last = h
        self.net.append(nn.Linear(last, self.n_dyn_out, bias=self.bias))
        self.net = nn.Sequential(*self.net)

    def forward(self, t, x):
        if self.time_dependent:
            x = torch.cat([x, t.repeat(x.shape[0], 1)], dim=-1)
        return self.net(x)


class CDEDynamics(Dynamics):

    def __init__(self, n_data, n_dyn, hiddens: list, bias=True, fixed_freq=False, freq=3., periodicity='cos'):
        super().__init__(n_dyn_inp=n_dyn,
                         n_dyn_out=(n_data * n_dyn),
                         hiddens=hiddens, bias=bias, fixed_freq=fixed_freq, freq=freq,
                         periodicity=periodicity)

        self.n_data, self.n_dyn = n_data, n_dyn

    def forward(self, t, x):
        return super().forward(t, x).view(-1, self.n_dyn, self.n_data)


class ODEDynamics1stOrder(Dynamics):

    def __init__(self, n_dyn, hiddens: list, n_latent=0, bias=True, fixed_freq=True, freq=3., time_dependent=False, periodicity='cos'):
        super().__init__(n_dyn_inp=n_dyn + n_latent,
                         n_dyn_out=n_dyn,
                         hiddens=hiddens, bias=bias, fixed_freq=fixed_freq, freq=freq, time_dependent=time_dependent,
                         periodicity=periodicity)

        self.n_latent = n_latent
        # This is a fixed tensor that represents the derivative of latent vector.
        # It is fixed to zero, so that the latent vector doesn't evolve over time.
        self.dlatent = None

    def forward(self, t, x):
        if self.n_latent != 0:
            x, latent = x
            batch_size = latent.shape[0]
            x = torch.cat([x, latent], dim=-1)

        out = super().forward(t, x)

        if self.n_latent != 0:
            # the latent vector do not 'evolve' when running the solver
            if self.dlatent is None:
                # cache this the first time the control reaches here
                self.dlatent = torch.zeros(batch_size, self.n_latent,
                                           device=x.device, requires_grad=False)
            out = (out, self.dlatent)

        return out


class ODEDynamics2ndOrder(Dynamics):

    def __init__(self, n_dyn, hiddens: list, n_latent=0, bias=True, fixed_freq=False, freq=3., time_dependent=False, periodicity='cos'):
        super().__init__(n_dyn_inp=n_dyn + n_latent,
                         # n_dyn_inp=(2 * n_dyn) + n_latent,
                         n_dyn_out=n_dyn,
                         hiddens=hiddens, bias=bias, fixed_freq=fixed_freq, freq=freq, time_dependent=time_dependent,
                         periodicity=periodicity)

        self.n_latent = n_latent
        # This is a fixed tensor that represents the derivative of latent vector.
        # It is fixed to zero, so that the latent vector doesn't evolve over time.
        self.dlatent = None

        # use this dummy when dlogp calculation not required
        self.dlogp = None

    @second
    @logp
    def forward(self, t, x):
        if self.n_latent != 0:
            x, latent = x
            batch_size = latent.shape[0]
            x = torch.cat([x, latent], dim=-1)

        out = super().forward(t, x)

        if self.n_latent != 0:
            # the latent vector do not 'evolve' when running the solver
            if self.dlatent is None:
                # cache this the first time the control reaches here
                self.dlatent = torch.zeros(batch_size, self.n_latent,
                                           device=x.device, requires_grad=False)
            out = (out, self.dlatent)

        return out
