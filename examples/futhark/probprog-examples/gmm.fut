import "../ad"

module mk_gmm (P: real) = {
  let logsumexp xs =
    let mxs = P.maximum xs
    in (map (\x -> P.(exp (x - mxs))) xs |> P.sum |> P.log) P.+ mxs

  let log_normalize x = P.(map (\y -> y-logsumexp x) x)

  let dotprod xs ys = map2 (P.*) xs ys |> P.sum

  let logpdf x mu sigma = P.(negate ((log sigma) + f32 0.5 * (log (f32 2.0) + log pi)) +
                             negate ((x-mu)**(f32 2.0))/(f32 2.0 * sigma**f32 2.0))

  let log_likelihood (ws: []P.t) (mus: []P.t) (sigmas: []P.t) (xs: []P.t) =
    let cluster_lls =
      map3 (\w mu sigma ->
              map (\x -> P.(w + logpdf x mu sigma)) xs)
           ws mus sigmas
    in map logsumexp (transpose cluster_lls) |> P.sum
}

module real = f32
type real = real.t
module gmm = mk_gmm real
module dual_real = mk_dual real
module gmm_dual = mk_gmm dual_real

let diff (ws: [3]real) (mus: [3]real) (sigmas: [3]real) (xs: []real): ([]real, []real, []real) =
  let params = ws ++ mus ++ sigmas
  let derivatives =
    tabulate (length params)
             (\i ->
                let params' = tabulate (length params) (\j -> real.bool (j == i))
                let ws' = map2 dual_real.make_dual ws params'[0:3]
                let mus' = map2 dual_real.make_dual mus params'[3:6]
                let sigmas' = map2 dual_real.make_dual sigmas params'[6:9]
                in gmm_dual.log_likelihood ws' mus' sigmas' (map dual_real.inject xs))
    |> map dual_real.get_deriv
  in (derivatives[0:3], derivatives[3:6], derivatives[6:9])

import "optim"
import "util"

module nine = { let len = 9i32 }

module gmm_grad : grad_optimizable with param.t = [nine.len]real
                                   with loss.t = real
                                   with data = []real = {
  module param = arr real nine
  module loss = real
  type data = []real

  let eval_loss (param: param.t) (xs: data): loss.t =
    -gmm.log_likelihood param[0:3] param[3:6] param[6:9] xs

  let grad (param: param.t) (xs: data): param.t =
    let (ws, mus, sigmas) = diff param[0:3] param[3:6] param[6:9] xs
    in map real.negate (ws ++ mus ++ sigmas)
}

module sgd = stochastic_gradient_descent gmm_grad minstd_rand

-- ==
-- random input { 100 [1000000]f32 }

let main (n: i32) (xs: []f32) =
  let xs = map real.f32 xs
  let ws = [0.1,0.5,0.4]
  let mus = [0.1,0.2,0.3]
  let sigmas = [0.4,0.5,0.6]
  let (_, losses, params') = sgd.run {learning_rate=0.01} (minstd_rand.rng_from_seed [1337])
                                     (ws++mus++sigmas)
                                     xs
                                     n
  let ws' = params'[0:3]
  let mus' = params'[3:6]
  let sigmas' = params'[6:9]
  in (losses, params', gmm.log_likelihood ws' mus' sigmas' xs)

