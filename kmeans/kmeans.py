import argparse
from functools import partial
from pathlib import Path

import futhark_data
import torch
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp

data_dir = Path(__file__).parent / 'data'


def cost(points, centers):
    dists = ((points[None, ...] - centers[:, None, ...]) ** 2).sum(-1)
    min_dist = torch.take_along_dim(dists, torch.argmin(dists, 0)[None, :], dim=0)
    return min_dist.sum()


def kmeans(tolerance, k, max_iter, features):
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
    return clusters


def bench(kmeans_args, times=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((times,), dtype=float)
    for i in range(times):
        start.record()
        kmeans(*kmeans_args)
        torch.cuda.synchronize()
        end.record()
        timings[i] = start.elapsed_time(end) * 1000  # micro seconds

    return float(timings[1:].mean()), float(timings[1:].std())


def data_gen(name):
    # run futhark dataget kmeans.fut '0' > data/random.in
    data_file = data_dir / f'{name}.in'
    assert data_file.exists()

    with data_file.open('rb') as f:
        kmeans_args = tuple(futhark_data.load(f))
    return tuple(map(partial(torch.tensor, device=args.device, dtype=torch.float32), kmeans_args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Newtonian KMeans"
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--datasets", nargs="+", default=["random", "kdd_cup", ])

    args = parser.parse_args()
    if args.device == 'cuda':
        assert torch.cuda.is_available(), "Cuda not available"

    for name in args.datasets:
        print(name, '±'.join(map(str, bench(data_gen(name))), 'micro seconds'))
