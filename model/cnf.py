import functools
import torch


def trace_jac_dynamics(f, z):
    """
    Computes the Tr(J(F)) where F = dz/dt
    Taken from @rtqichen's github repo:
    torchdiffeq/blob/b3df7c9dab5b0dcc2f0187ac0d6e2f8ffd014700/examples/cnf.py#L66
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z,
                                        create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


def logp(forward):

    @functools.wraps(forward)
    def _wrapper(self, t, x):
        # This expects two components of 'x', although
        # the 'logp' state is not required by the impl
        state, latent, logp0 = x
        state.requires_grad_(True)
        with torch.set_grad_enabled(True):
            dstate_dt, dlat_dt = forward(self, t, (state, latent))

            if self.dlogp is None:
                batch_size, _ = state.shape
                self.dlogp = torch.zeros(batch_size, 1,
                                         dtype=state.dtype, device=state.device, requires_grad=False)

            if self.no_dlogp_compute:
                dlogp = self.dlogp
            else:
                dlogp = -trace_jac_dynamics(dstate_dt, state).unsqueeze(-1)
        return dstate_dt, dlat_dt, dlogp

    return _wrapper


def second(forward):

    @functools.wraps(forward)
    def _wrapper(self, t, x):
        drest_dt = None
        if isinstance(x, (list, tuple)):
            x, *rest = x
            _, x_vel = torch.tensor_split(x, 2, dim=-1)
            d2state_dt2, *drest_dt = forward(self, t, (x_vel, *rest))
        else:
            _, x_vel = torch.tensor_split(x, 2, dim=-1)
            d2state_dt2 = forward(self, t, x_vel)

        dim = x.shape[-1]
        _, v = torch.split(x, dim // 2, -1)

        dxdt = torch.cat([v, d2state_dt2], -1)

        if drest_dt is not None:
            return (dxdt, *drest_dt)
        else:
            return dxdt

    return _wrapper
