from functools import partial
from pathlib import Path

import futhark_data
import torch
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = Path(__file__).parent / 'data'


def cost(points, centers):
    dists = ((points[None, ...] - centers[:, None, ...]) ** 2).sum(-1)
    min_dist = torch.take_along_dim(dists, torch.argmin(dists, 0)[None, :], dim=0)
    return min_dist.sum()


if __name__ == '__main__':
    with (data_dir / 'kdd_cup.in').open('rb') as f:
        tolerance, k, max_iter, features = map(partial(torch.tensor, device=device, dtype=torch.float32),
                                               futhark_data.load(f))
    clusters = torch.flip(features[-int(k):], (0,))

    t = 0
    converged = False
    while not converged and t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.))
        _, hes = vhp(partial(cost, features), clusters, v=torch.ones_like(clusters))

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1

    print(clusters.tolist())
