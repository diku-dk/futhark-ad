import "lib/github.com/diku-dk/cpprandom/random"

module type optimizable = {
  module param: real
  module loss: real
  type data
  val eval_loss: param.t -> data -> loss.t
}

module type grad_optimizable = {
  include optimizable
  val grad: param.t -> data -> param.t
}

module type bound_optimizable = {
  include optimizable
  val param_lower: param.t
  val param_upper: param.t
}

module type optimizer = {
  module param: real
  module loss: real
  type data
  type rng

  type options
  val default_options: options

  val run: options -> rng -> param.t -> data -> i32 -> (rng, []loss.t, param.t)
  val run': rng -> param.t -> data -> i32 -> (rng, []loss.t, param.t)
}

module stochastic_gradient_descent (optable: grad_optimizable) (E : rng_engine):
  optimizer with data = optable.data
            with rng = E.rng
            with param.t = optable.param.t
            with loss.t = optable.loss.t
            with options = {learning_rate: f32} = {
  module param = optable.param
  module loss = optable.loss
  type data = optable.data
  type rng = E.rng

  type options = {learning_rate: f32}
  let default_options: options = {learning_rate=0.1}

  let run ({learning_rate}: options) (rng: rng) (p: param.t) (xs: data) (n_iters: i32) =
  loop (rng, losses, p) = (rng, [optable.eval_loss p xs], p) for _i < n_iters do
    let p = p param.- param.f32 learning_rate param.* optable.grad p xs
    let losses = losses ++ [optable.eval_loss p xs]
    in (rng, losses, p)

  let run' = run default_options
}

module particle_swarm (optable: bound_optimizable) (E: rng_engine):
  optimizer with data = optable.data
            with rng = E.rng
            with param.t = optable.param.t
            with loss.t = optable.loss.t
            with options = {swarm_size: i32, acceleration_rate: f32,
                            local_learning_rate: f32, global_learning_rate: f32} =
{
  module param = optable.param
  module loss = optable.loss
  module unif_dist = uniform_real_distribution param E

  type data = optable.data
  type rng = E.rng

  type options = {swarm_size: i32, acceleration_rate: f32, local_learning_rate: f32, global_learning_rate: f32}
  let default_options: options = {swarm_size=100,acceleration_rate=(-0.5),local_learning_rate=0.5,global_learning_rate=3.0}

  let sample (rng: rng) (lower: param.t) (upper: param.t) (n: i32): (rng, []param.t) =
    let (rngs, vals) = E.split_rng n rng 
                     |> map (unif_dist.rand (lower, upper))
                     |> unzip
    in (E.join_rng rngs, vals)

  let argmin (lf: param.t -> loss.t) (ps: []param.t) =
    let argmin_op p p' = if lf p loss.< lf p' then p else p'
    in reduce argmin_op ps[0] ps

  let run ({swarm_size,acceleration_rate,local_learning_rate,global_learning_rate}:options) (rng: rng) (p: param.t) (xs: data) (n_iters: i32) =
    let (rng, ps) = sample rng optable.param_lower optable.param_upper swarm_size
    let param_range = optable.param_upper param.- optable.param_lower
    let (rng, vs) = sample rng (param.negate param_range) param_range swarm_size
    let gl = p
    let (rng, losses, _, _, gl, _) =
      loop (rng, losses, ps, bps, gl, vs) = (rng, [], ps, ps, gl, vs) for _i < n_iters do
      let sample_vs (rng, p, bp, v) =
        let (rng, rp) = unif_dist.rand (param.f32 0.0, param.f32 1.0) rng
        let (rng, rg) = unif_dist.rand (param.f32 0.0, param.f32 1.0) rng
        let new_v = param.f32 acceleration_rate param.* v param.+
                    param.f32 local_learning_rate param.* rp param.* (bp param.- p) param.+
                    param.f32 global_learning_rate param.* rg param.* (gl param.- p)
        in (rng, new_v)
      let (rngs, vs) = map sample_vs (zip4 (E.split_rng swarm_size rng) ps bps vs) |> unzip
      let rng = E.join_rng rngs
      let ps = map2 (\p v -> p param.+ v) ps vs
      let bps = map2 (\p bp -> if optable.eval_loss p xs loss.< optable.eval_loss bp xs then p else bp) ps bps
      let gl = argmin (\bp -> optable.eval_loss bp xs) bps
      in (rng, concat losses [optable.eval_loss gl xs], ps, bps, gl, vs)
    in (rng, losses, gl)
  let run' = run default_options
}
