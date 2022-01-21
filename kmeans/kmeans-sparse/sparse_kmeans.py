import gzip
from functools import partial
from pathlib import Path

import futhark_data
import torch
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp

data_dir = Path(__file__).parent / "data"

VERBOSE = False


def all_pairs_norm(a, b):
    a_sqr = torch.sparse.sum(a ** 2, 1).to_dense()[None, :]
    b_sqr = torch.sum(b ** 2, 1)[:, None]
    diff = torch.sparse.mm(a, b.T).T
    if VERBOSE:
        print('entries', diff.numel(), 'zeros', diff.numel() - torch.count_nonzero(diff))
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def kmeans(max_iter, clusters, features, tolerance=1):
    t = 0
    converged = False
    hes_v = torch.ones_like(clusters)
    while not converged and t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.0))
        _, hes = vhp(partial(cost, features), clusters, v=hes_v)

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1
    return clusters


def data_gen(name):
    """Dataformat CSR  (https://en.wikipedia.org/wiki/Sparse_matrix)"""
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()

    with gzip.open(data_file.resolve(), ("rb")) as f:
        values, indices, pointers = tuple(futhark_data.load(f))

    return (
        torch.tensor(pointers, dtype=torch.int32),
        torch.tensor(indices, dtype=torch.int32),
        values,
    )


def bench_gpu(dataset, k=10, max_iter=1000, times=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((times,), dtype=float)
    sp_data = data_gen(dataset)
    sp_features = (
        torch.sparse_csr_tensor(*sp_data, dtype=torch.float32).to_dense().to_sparse()
    )
    sp_clusters = get_clusters(k, *sp_data, sp_features.size()[1])
    for i in range(times):
        torch.cuda.synchronize()
        start.record()
        kmeans(max_iter, sp_clusters, sp_features, threshold)
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        timings[i] = start.elapsed_time(end) * 1000  # micro seconds

    return float(timings[1:].mean()), float(timings[1:].std())


def get_clusters(k, pointers, indices, values, num_col):
    end = pointers[k]
    sp_clusters = (
        torch.sparse_csr_tensor(
            pointers[: (k + 1)],
            indices[:end],
            values[:end],
            requires_grad=True,
            dtype=torch.float32,
            size=(k, num_col),
        ).to_dense()
    )
    return sp_clusters


if __name__ == "__main__":
    k = 10
    max_iter = 1000
    threshold = 5e-3

    for dataset in ['movielens', 'nytimes', 'scrna']:
        print(dataset, f'{bench_gpu(dataset)} micro seconds')
