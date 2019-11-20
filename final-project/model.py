import time
import warnings

import torch
import torch.nn as nn

import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist


def kalman_filter(udim, xdim, zdim, cuda=False, jit=False):
    device = "cuda" if cuda else "cpu"

    def model(u_seq, z_seq, batch_size=None):
        # Move input and output to model device
        u_seq = list(map(lambda x: x.to(device), u_seq))
        z_seq = list(map(lambda x: x.to(device), z_seq))

        # Process sequence lengths
        lengths = torch.tensor(list(map(len, u_seq)), device=device)
        num_sequences, max_length = len(lengths), lengths.max()

        def pad_fn(x): return nn.functional.pad(x, (0, 0, 0, max_length-len(x)))
        u_seq = torch.stack(list(map(pad_fn, u_seq)), dim=0).unsqueeze(-1)
        z_seq = torch.stack(list(map(pad_fn, z_seq)), dim=0).unsqueeze(-1)

        plate_u1 = pyro.plate("plate_u1", udim, dim=-1)
        plate_x1 = pyro.plate("plate_x1", xdim, dim=-1)
        plate_x2 = pyro.plate("plate_x2", xdim, dim=-2)
        plate_z1 = pyro.plate("plate_z1", zdim, dim=-1)
        plate_z2 = pyro.plate("plate_z2", zdim, dim=-2)
        plate_seq = pyro.plate("plate_seq", num_sequences, batch_size, dim=-3)

        # Parameter priors
        # with poutine.mask(mask=True):

        # Noise variance
        Q = 0.1*torch.ones(xdim, device=device).unsqueeze(-1)
        R = 0.1*torch.ones(zdim, device=device).unsqueeze(-1)

        # State transition matrix
        with plate_x1:
            F_diag = pyro.sample("F_diag", dist.Beta(torch.tensor(2.5, device=device),
                                                     torch.tensor(2.5, device=device)))
            with plate_x2:
                F_rest = pyro.sample("F_rest", dist.Normal(torch.tensor(0.0, device=device),
                                                           torch.tensor(0.1, device=device)))
        mask_diag = torch.eye(xdim, device=device)
        F = F_diag*mask_diag + F_rest*(1-mask_diag)

        # Input filter matrix
        with plate_x2, plate_u1:
            G = pyro.sample("G", dist.Normal(torch.tensor(0.0, device=device),
                                             torch.tensor(0.1, device=device)))

        # Measurement matrix
        with plate_z2, plate_x1:
            H = pyro.sample("H", dist.HalfNormal(torch.tensor(1.0, device=device)))

        # Measurement bias
        with plate_z1:
            b = pyro.sample("b", dist.Normal(torch.tensor(0.0, device=device),
                                             torch.tensor(0.2, device=device))).unsqueeze(-1)

        # We subsample batch_size items out of num_sequences items.
        with plate_seq as batch:
            lengths = lengths[batch]

            num_sequences = num_sequences if batch_size is None else batch_size
            x = torch.zeros((num_sequences, xdim, 1), device=device)
            for t in pyro.markov(range(max_length if jit else lengths.max())):
                with pyro.poutine.mask(mask=(t < lengths).unsqueeze(-1).unsqueeze(-1)):
                    x = pyro.sample(f"x_{t}", dist.Normal(F @ x + G @ u_seq[batch, t], Q))
                    with plate_z2:
                        pyro.sample(f"z_{t}", dist.Normal(H @ x + b, R), obs=z_seq[batch, t])

    def expose_fn(msg): return not msg["name"].startswith("x_")
    guide = pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model, expose_fn=expose_fn))

    return model, guide


def inference(svi, U, Z, num_iterations=100, interval=10, batch_size=None, num_particles=1):
    zdim = Z[0].shape[-1]
    num_observations = sum([len(z) for z in Z]) * zdim

    loss_tr = []
    pyro.clear_param_store()
    for j in range(num_iterations):
        t0 = time.clock_gettime(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = svi.step(U, Z, batch_size=batch_size)
        epoch_time = time.clock_gettime(0) - t0
        loss_tr.append(loss/num_observations)

        if j % interval == 0:
            t = time.localtime()
            t = (f'{t.tm_year}/{t.tm_mon:02d}/{t.tm_mday:02d}'
                 + '-{t.tm_hour:02d}:{t.tm_min:02d}')
            print(f'{t} -- [iteration {j+1:04d}] loss: {loss_tr[-1]:.4f}'
                  + f' -- epoch time: {epoch_time:.3f} sec')

    return loss_tr
