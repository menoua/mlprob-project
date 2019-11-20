import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pyro.distributions as dist


def summary_data(stim, data):
    data_len = sum([len(u) for u in stim])/100
    print(f"Data consists of {len(stim)} sequences totalling {data_len} seconds.\n")

    print("5 seconds of sample stimulus of speech:")
    plt.figure(figsize=(8, 3))
    plt.imshow(stim[0][:500, :-5].T, aspect=2, origin='ll')
    plt.xticks([0, 100, 200, 300, 400, 500], [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.yticks([0, 60], [50, 8000])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

    print("5 seconds of sample response to speech:")
    plt.figure(figsize=(8, 3))
    plt.plot(data[0][:500, 67, 0])
    plt.xlim(0, 500)
    plt.xticks([0, 100, 200, 300, 400, 500], [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xlabel("Time (s)")
    plt.ylabel("Neural activity (a.u.)")
    plt.show()


def prep_variables(stim, data, window=10, downsample=3, show_sample=False):
    U = [torch.from_numpy(u)[:, :-5] for u in stim]
    U = [nn.functional.avg_pool1d(u.unsqueeze(0), downsample).squeeze(0) for u in U]
    U = [nn.functional.pad(u, (0, 0, window-1, 0)).T.unsqueeze(0).unsqueeze(-1) for u in U]
    U = [nn.functional.unfold(u, (window, 1)).squeeze(0).squeeze(-1).T for u in U]

    Z = [torch.from_numpy(z.mean(-1)) for z in data]

    if show_sample:
        fbins = U[0].shape[-1]//window
        print(f"Sample windowed input ({window} time steps * {fbins} frequency bins):")
        plt.figure(figsize=(2, 2))
        plt.imshow(U[0][106, :].reshape(fbins, window), aspect=1, origin='ll')
        plt.show()

    return U, Z


def filter_electrodes(Z, locz):
    ind_hg = locz["hg"].flatten().astype("bool")
    ind_pt = locz["pt"].flatten().astype("bool")
    ind_stg = locz["stg"].flatten().astype("bool")

    Z = [torch.cat((z[:, ind_hg], z[:, ind_pt], z[:, ind_stg]), 1) for z in Z]
    print(f"electrodes: {Z[0].shape[-1]} (HG: {ind_hg.sum()}, "
          + f"PT: {ind_pt.sum()}, STG: {ind_stg.sum()})")

    ind_hg = torch.arange(ind_hg.sum(), dtype=torch.long)
    ind_pt = torch.arange(ind_pt.sum(), dtype=torch.long) + ind_hg.max() + 1
    ind_stg = torch.arange(ind_stg.sum(), dtype=torch.long) + ind_pt.max() + 1

    return Z, ind_hg, ind_pt, ind_stg


def summary_prior():
    plt.figure(figsize=(12, 2.5))

    plt.subplot(131)
    D = dist.Beta(2.5, 2.5)    # F_diag
    plt.hist([D.sample() for _ in range(50_000)], range=(-1, 2), bins=200, density=True, alpha=0.7)
    D = dist.Normal(0.0, 0.1)  # F_rest
    plt.hist([D.sample() for _ in range(50_000)], range=(-1, 2), bins=200, density=True, alpha=0.7)
    plt.legend(['F_diag', 'F_rest'])
    plt.xlim(-1, 2)

    plt.subplot(132)
    D = dist.Normal(0.0, 0.1)  # G
    plt.hist([D.sample() for _ in range(50_000)], range=(-0.75, 0.75), bins=200, density=True)
    plt.legend(['G'])
    plt.xlim(-0.75, 0.75)

    plt.subplot(133)
    D = dist.HalfNormal(1.0)   # H
    plt.hist([D.sample() for _ in range(50_000)], range=(-1, 4), bins=200, density=True, alpha=0.7)
    D = dist.Normal(0.0, 0.2)  # b
    plt.hist([D.sample() for _ in range(50_000)], range=(-1, 4), bins=200, density=True, alpha=0.7)
    plt.legend(['H', 'b'])
    plt.xlim(-1, 4)

    plt.show()


def gaussian_smooth(input, kernel_size=5, sigma=1):
    channels = input.shape[0]
    input = input.unsqueeze(0)
    kernel_size = [kernel_size] * 2
    sigma = [sigma] * 2

    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    kernel = kernel / torch.sum(kernel)

    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    input = nn.functional.pad(input, [k//2 for k in kernel_size]*2, "constant")
    output = nn.functional.conv2d(input, kernel, groups=channels).squeeze(0)
    return output


def summary_state(x):
    plt.figure(figsize=(10, 8))

    plt.subplot(211)
    plt.plot(x)
    plt.xlim(0, 500)
    plt.title("Latent state variables")
    plt.legend(("state 1", "state 2", "state 3"))
    plt.xlabel("Time (s)")
    plt.ylabel("Latent state (a.u.)")
    plt.xticks((0, 100, 200, 300, 400, 500), (0, 0.1, 0.2, 0.3, 0.4, 0.5))

    plt.subplot(223)
    plt.plot(x[:, 1], x[:, 0])
    plt.xlabel("Latent state 2")
    plt.ylabel("Latent state 1")

    plt.subplot(224)
    plt.plot(x[:, 2], x[:, 0])
    plt.xlabel("Latent state 3")

    plt.show()


def summary_accuracy(Q):
    plt.figure(figsize=(10, 2))

    plt.plot(Q)
    plt.plot([64.5, 64.5], [0, 1], "r")  # Mark start of PT electrodes
    plt.plot([89.5, 89.5], [0, 1], "r")  # Mark start of STG electrodes
    plt.ylim(0.0, 0.8)
    plt.title("Prediction accuracy")
    plt.xlabel("Electrode index")
    plt.ylabel("Pearson correlation")

    plt.show()


def summary_G(G, window, smooth=False, kernel_size=3, sigma=1):
    xdim, udim = G.shape[:2]

    G = G.reshape(xdim, udim//window, window)
    if smooth:
        G = gaussian_smooth(G, kernel_size, sigma)

    plt.figure(figsize=(xdim*2.5, 2))
    for l in range(xdim):
        plt.subplot(1, xdim, l+1)
        absmax = abs(G[l]).max() * 0.8
        plt.imshow(G[l], aspect=1, origin='ll', vmin=-absmax, vmax=absmax)
        plt.title(f"Latent state {l+1}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
    plt.show()


def summary_H(H):
    xdim = H.shape[1]
    plt.figure(figsize=(12, 2))

    plt.imshow(H.T, aspect=4, origin='ll')
    plt.plot([64.5, 64.5], [-0.5, xdim-0.5], "r", linewidth=2)  # Mark start of PT electrodes
    plt.plot([89.5, 89.5], [-0.5, xdim-0.5], "r", linewidth=2)  # Mark start of STG electrodes
    plt.title("State-measurement transition matrix")
    plt.xlabel("Electrode index")
    plt.ylabel("Latent state")

    plt.show()


def summary_b(b):
    zdim = b.shape[0]
    plt.figure(figsize=(12, 2))

    plt.plot(b)
    plt.xlim(0, zdim)
    plt.plot([64.5, 64.5], [-0.1, 0.1], "r")  # Mark start of PT electrodes
    plt.plot([89.5, 89.5], [-0.1, 0.1], "r")  # Mark start of STG electrodes
    plt.title("Measurement bias")
    plt.xlabel("Electrode index")
    plt.ylabel("Bias value (a.u.)")

    plt.show()


def read_params(params, xdim, zdim):
    F_diag = params["auto_F_diag"].detach().cpu()
    F_rest = params["auto_F_rest"].detach().cpu()
    F = F_diag*torch.eye(xdim) + F_rest*(1-torch.eye(xdim))
    G = params["auto_G"].detach().cpu()
    H = params["auto_H"].detach()[:zdim].cpu()  # The saved model includes 64 more electrodes
    b = params["auto_b"].detach()[:zdim].cpu()  # Same for bias

    return F, G, H, b
