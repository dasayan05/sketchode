import torch
import torch.nn as nn
import torchcde as cde
import torchdiffeq as tdeq

from model.dynamics import (CDEDynamics, Cosine, ODEDynamics1stOrder,
                            ODEDynamics2ndOrder)


class GRUOneSeqEncoder(nn.Module):

    def __init__(self, n_data, n_hidden, n_latent, _, **kwargs):
        # 'arch' is unused, just to keep compatible with CDE ones
        super().__init__()

        self.n_data = n_data
        self.n_hidden = n_hidden
        self.n_layers = kwargs.get('n_layers', 1)
        self.n_latent = n_latent
        self.bidirectional = kwargs.get('bidirectional', True)
        self.vae = kwargs.get('vae', False)
        self.bounded_latent = kwargs.get('bounded_latent', False)

        self.birnn = nn.LSTM(self.n_data, self.n_hidden, 1,
                             bias=kwargs.get('bias', True), bidirectional=self.bidirectional)

        if self.vae:
            self.enc2latent_mu = torch.nn.Sequential(
                nn.Linear(self.n_hidden * (int(self.bidirectional) + 1) * self.n_layers,
                          self.n_latent)
            )
            self.enc2latent_logvar = torch.nn.Sequential(
                nn.Linear(self.n_hidden * (int(self.bidirectional) + 1) * self.n_layers,
                          self.n_latent)
            )
        else:
            self.enc2latent = torch.nn.Sequential(
                nn.Linear(self.n_hidden * (int(self.bidirectional) + 1) * self.n_layers,
                          self.n_latent),
                nn.Tanh() if self.bounded_latent else nn.Identity()
            )

    def reparameterize(self, mu, logvar):
        # only used if self.vae
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, _):
        # 'x' is expected to be (len x batch x feature)
        _, batch_size, _ = x.shape
        out, _ = self.birnn(x)
        out, _ = torch.max(out, 0)

        if self.vae:
            # set these variables to be used by .loss() call later
            self.latent_mu = self.enc2latent_mu(out)
            self.latent_logvar = self.enc2latent_logvar(out)
            return self.reparameterize(self.latent_mu, self.latent_logvar)
        else:
            return self.enc2latent(out)


class CDEOneSeqEncoder(nn.Module):

    def __init__(self, n_data, n_hidden, n_latent, arch=[32, 64], **kwargs):
        super().__init__()

        self.n_data = n_data
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.arch = arch
        self.ode_method = kwargs.get('ode_method', 'rk4')
        self.vae = kwargs.get('vae', False)
        self.bounded_latent = kwargs.get('bounded_latent', False)
        self.periodicity = kwargs.get('periodicity', 'cos')

        self.dynamics = CDEDynamics(self.n_data + 1,  # add time channel
                                    self.n_hidden, self.arch,
                                    bias=kwargs.get('bias', True),
                                    fixed_freq=kwargs.get('fixed_freq', True),
                                    freq=kwargs.get('freq', 3.),
                                    periodicity=self.periodicity)

        if self.vae:
            self.enc2latent_mu = torch.nn.Sequential(
                nn.Linear(self.n_hidden, self.n_latent)
            )
            self.enc2latent_logvar = torch.nn.Sequential(
                nn.Linear(self.n_hidden, self.n_latent)
            )
        else:
            self.enc2latent = torch.nn.Sequential(
                nn.Linear(self.n_hidden, self.n_latent),
                nn.Tanh() if self.bounded_latent else nn.Identity()
            )

    def reparameterize(self, mu, logvar):
        # only used if self.vae
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_naive(self, x, t, init=None):
        # 'x' is expected to be (len x batch x feature) because this
        # is widely used .. but CDE functions take (batch x len x feature)
        x = x.permute(1, 0, 2)
        batch_size, _, _ = x.shape
        t_ = t.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        x = torch.cat([x, t_], -1)
        spline = cde.CubicSpline(coeffs=cde.natural_cubic_coeffs(x), t=t)
        init = torch.zeros(batch_size, self.n_hidden, device=x.device) if init is None else init

        options = {}
        if self.ode_method in ['rk4', 'euler']:
            step_size = (t[1:] - t[:-1]).min()  # for fixed solvers
            options['step_size'] = step_size

        out = cde.cdeint(spline, self.dynamics, init, t=t, method=self.ode_method,
                         options=options)
        return out

    def forward_vae(self, out):
        if self.vae:
            self.latent_mu = self.enc2latent_mu(out)
            self.latent_logvar = self.enc2latent_logvar(out)
            return self.reparameterize(self.latent_mu, self.latent_logvar)
        else:
            return self.enc2latent(out)

    def forward(self, x, t):
        out = self.forward_naive(x, t)
        out, _ = torch.max(out, 1)
        return self.forward_vae(out)


class CDEStrokewiseEncoder(CDEOneSeqEncoder):

    def __init__(self, n_data, n_hidden, n_latent, arch, **kwargs):
        super().__init__(n_data, n_hidden, n_latent, arch=arch, **kwargs)

        self.update = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU()
        )

    def forward_naive(self, xs, t):
        outs = []
        for i, x in enumerate(xs):
            if i == 0:
                out = super().forward_naive(x.points, t)
            else:
                # end state as init for the next one
                out = super().forward_naive(x.points, t, init=next_init)
            summary_state, _ = torch.max(out, 1, keepdim=False)
            # summary_state = out[:, -1, :]
            next_init = self.update(summary_state)
            outs.append(next_init)

        return torch.stack(outs, 1)


class ODEOneSeqDecoder(nn.Module):

    def __init__(self, n_data, n_hidden, n_latent, arch=[32, 64], **kwargs):
        super().__init__()

        self.n_data = n_data
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.observation_model = kwargs.get('observation_model', True)
        self.n_dyn = self.n_hidden \
            if self.observation_model \
            else (self.n_data + self.n_hidden)  # if a separate observation model is used
        self.arch = arch
        self.ode_method = kwargs.get('ode_method', 'rk4')
        self.order = kwargs.get('order', 2)
        self.append_latent = kwargs.get('append_latent', True)
        self.periodicity = kwargs.get('periodicity', 'cos')

        self.latent2dec = torch.nn.Linear(self.n_latent, self.order * self.n_dyn)

        ODEDynamics = ODEDynamics2ndOrder if self.order == 2 else ODEDynamics1stOrder
        self.dynamics = ODEDynamics(self.n_dyn, self.arch,
                                    n_latent=self.n_latent if self.append_latent else 0,
                                    bias=kwargs.get('bias', True),
                                    fixed_freq=kwargs.get('fixed_freq', True),
                                    freq=kwargs.get('freq', 3.),
                                    time_dependent=kwargs.get('time_dependent', False),
                                    periodicity=self.periodicity)

        if self.observation_model:
            self.observation = nn.Linear(self.n_hidden, self.n_data, bias=True)

    def forward_naive(self, latent, t, init=None, scale_dyn=None):
        options = {}
        if self.ode_method in ['rk4', 'euler']:
            step_size = (t[1:] - t[:-1]).min()  # for fixed solvers
            options['step_size'] = step_size

        dec_init_state = self.latent2dec(latent) if init is None else init
        x = (dec_init_state, latent) if self.append_latent else (dec_init_state,)

        if scale_dyn is not None:
            # this is an experimental thing
            # no effect in original code
            for m in self.dynamics.modules():
                if isinstance(m, Cosine):
                    m.test_freq = scale_dyn

        batch_size, _ = latent.shape
        dummy_logp0 = torch.zeros(batch_size, 1,
                                  dtype=dec_init_state.dtype,
                                  device=dec_init_state.device,
                                  requires_grad=False)
        x = (*x, dummy_logp0)
        seq = tdeq.odeint_adjoint(self.dynamics, y0=x, t=t, method=self.ode_method,
                                  options=options)

        if self.append_latent:
            seq, _, logpt = seq

        return seq, logpt

    def forward_projection(self, seq):
        out_pos = seq[..., :self.n_dyn]
        traj_pos = self.deaugment(out_pos)

        traj_pos_x, traj_pos_p = traj_pos[..., :-1], traj_pos[..., -1]

        return traj_pos_x, torch.sigmoid(traj_pos_p)

    def forward(self, latent, t, _, scale_dyn=None):
        seq, logpt = self.forward_naive(latent, t, scale_dyn=scale_dyn)
        return self.forward_projection(seq), logpt

    def deaugment(self, out):
        if self.observation_model:
            return self.observation(out)
        else:
            return out[..., :self.n_data]


class ODEStrokewiseDecoder(ODEOneSeqDecoder):

    def __init__(self, n_data, n_hidden, n_latent, arch, **kwargs):
        super().__init__(n_data, n_hidden, n_latent, arch=arch, **kwargs)

        self.update = nn.Sequential(
            nn.Linear(self.n_dyn * self.order, self.n_dyn * self.order),
            nn.LeakyReLU()
        )
        self.finish = nn.Linear(self.n_dyn * self.order, 2)

    def forward(self, latent, t, _):
        out_x, out_p = [], []
        # self.max_strokes is kinda sneaking
        # we can update this later to be the maximum number of
        # decoding strokes
        logp = []
        for i in range(self.max_strokes):
            if i == 0:
                out, logpt = super().forward_naive(latent, t)
            else:
                out, logpt = super().forward_naive(latent, t, next_init)
            logp.append(logpt)
            summary_state, _ = torch.max(out, 0, keepdim=False)
            next_init = self.update(summary_state)

            out_p.append(self.finish(summary_state))
            out_x.append(self.forward_projection(out))

        return (out_x, out_p), sum(logp) / len(logp)

    def forward_projection(self, seq):
        out_pos = seq[..., :self.n_dyn]
        return self.deaugment(out_pos)


class GRUOneSeqDecoder(nn.Module):

    def __init__(self, n_data, n_hidden, n_latent, **kwargs):
        super().__init__()

        self.n_data = n_data
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_dyn = self.n_hidden
        self.append_latent = kwargs.get('append_latent', True)

        # *2 is for producing (h0, c0) of LSTM
        self.latent2dec = torch.nn.Linear(self.n_latent, self.n_dyn * 2)

        self.unirnn = nn.LSTM(self.n_data + (self.n_latent if self.append_latent else 0),
                              self.n_dyn, 1, bidirectional=False)  # non-causal, of course.

        self.observation = nn.Linear(self.n_hidden, self.n_data, bias=True)

    def forward(self, latent, t, x):
        training = (x.shape[0] > 1)  # if >1 length sequences, its teacher forcing
        if training:
            return self.forward_train(latent, x)
        else:
            seqlen, = t.shape
            return self.forward_inference(latent, x, seqlen)

    def forward_train(self, latent, x):
        seqlen, _, _ = x.shape
        x0, x = x[0, ...].unsqueeze(0), x[:-1, ...]

        dec_init_state = self.latent2dec(latent)
        dec_init_state = dec_init_state.unsqueeze(0)  # because layer=1 & uni-directional
        dec_init_state = (dec_init_state[..., :self.n_dyn].contiguous(),
                          dec_init_state[..., self.n_dyn:].contiguous())

        latent = latent.unsqueeze(0).repeat(seqlen - 1, 1, 1)  # reshaping for appending later

        inp = torch.cat([x, latent], -1) if self.append_latent else x
        out, _ = self.unirnn(inp, dec_init_state)
        out = self.observation(out)

        out_x, out_p = out[..., :-1], torch.sigmoid(out[..., -1]).unsqueeze(-1)
        out = torch.cat([out_x, out_p], -1)

        traj = torch.cat([x0, out], 0)

        return (traj[..., :-1], traj[..., -1]), \
            torch.tensor([0.], device=traj.device, dtype=traj.dtype)  # dummy for logp_cnf

    def forward_inference(self, latent, x, seqlen):
        dec_init_state = self.latent2dec(latent)
        dec_init_state = dec_init_state.unsqueeze(0)  # because layer=1 & uni-directional
        dec_init_state = (dec_init_state[..., :self.n_dyn].contiguous(),
                          dec_init_state[..., self.n_dyn:].contiguous())

        latent = latent.unsqueeze(0)

        x_s = []
        x_running = x
        state_running = dec_init_state
        for _ in range(seqlen - 1):
            inp = torch.cat([x_running, latent], -1) if self.append_latent else x
            out, state = self.unirnn(inp, state_running)
            out = self.observation(out)
            out_x, out_p = out[..., :-1], torch.sigmoid(out[..., -1]).unsqueeze(-1)
            out = torch.cat([out_x, out_p], -1)
            out_p_binary = out_p
            out_p_binary[out_p_binary <= .5] = 0.
            out_p_binary[out_p_binary > .5] = 1.
            out_binary = torch.cat([out_x, out_p_binary], -1)
            x_s.append(out)
            state_running = state
            x_running = out_binary

        traj = torch.cat([x, torch.cat(x_s, 0)], 0)

        return (traj[..., :-1], traj[..., -1]), \
            torch.tensor([0.], device=traj.device, dtype=traj.dtype)  # dummy for logp_cnf
