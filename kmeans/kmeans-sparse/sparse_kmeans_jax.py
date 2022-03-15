from functools import partial

from jax import numpy as jnp, jacrev, jacfwd
from jax.experimental import sparse
from jax.experimental.sparse import sparsify
from jax.lax import while_loop
from jax.random import PRNGKey, split, normal, bernoulli


def cost(points, centers):
    a_sqr = jnp.sum(points * points, 1)[None, :]
    b_sqr = jnp.sum(centers * centers, 1)[:, None]
    diff = jnp.matmul(points, centers.T).T
    dists = sparse.todense(a_sqr) + sparse.todense(b_sqr) - 2 * sparse.todense(diff)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


def kmeans(max_iter, clusters, features, tolerance=1):
    cost_sp = sparsify(cost)

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        jac_fn = jacrev(partial(cost_sp, features))
        hes_fn = jacfwd(jac_fn)

        new_cluster = clusters - jac_fn(clusters) / hes_fn(clusters).sum((0, 1))
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    t, rmse, clusters = while_loop(cond, sparsify(body), (0, float("inf"), clusters))
    return clusters


if __name__ == '__main__':
    data_key, sparse_key = split(PRNGKey(8))
    num_datapoints = 100
    num_features = 7
    sparsity = .5
    num_clusters = 5
    max_iter = 10
    features = normal(data_key, (num_datapoints, num_features))
    features = features * bernoulli(sparse_key, sparsity, (num_datapoints, num_features))
    clusters = features[:num_clusters]
    features = sparse.BCOO.fromdense(features)
    new_cluster = kmeans(max_iter, clusters, features)
    print(new_cluster)
