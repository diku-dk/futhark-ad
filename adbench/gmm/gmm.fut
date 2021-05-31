-- ==
-- entry: calculate_objective
-- input @ gmm_d10_K5.in

-- ==
-- entry: calculate_jacobian
-- input @ gmm_d10_K5.in

import "lib/github.com/diku-dk/linalg/linalg"

module linalg_f64 = mk_linalg f64

let fst (x,_) = x
let snd (_,y) = y

let sumBy 'a (f : a -> f64)  (xs : []a) : f64 = map f xs |> f64.sum

let l2normSq (v : []f64) = map (** 2) v |> f64.sum

let logsumexp = sumBy (f64.exp) >-> f64.log

let vMinus [m] (xs : [m]f64) (ys : [m]f64) : [m]f64 = zip xs ys |> map (\(x, y) -> x - y)

let frobeniusNormSq (mat : [][]f64) = flatten mat |> map (**2) |> f64.sum

let unpackQ [k] (logdiag: [k]f64) (lt: []f64) : [k][k]f64  =
  let d = length logdiag
  in (tabulate_2d d d (\i j ->
		        if i < j then 0
		        else if i == j then f64.exp logdiag[i]
		        else lt[d * j + i - j - 1 - j * (j + 1) / 2])) :> [k][k]f64

let logGammaDistrib (a : f64) (p : i64) =
  0.25 * f64.i64 p * f64.i64 (p - 1) * f64.log f64.pi +
  ((1...p) |> sumBy (\j -> f64.lgamma (a + 0.5 * f64.i64 (1 - j))))

let logsumexp_DArray (arr : []f64) =
    let mx = f64.maximum arr
    let sumShiftedExp = arr |> sumBy (\x -> f64.exp (x - mx))
    in f64.log sumShiftedExp + mx

let logWishartPrior (qsAndSums: []([][]f64, f64)) wishartGamma wishartM p =
    let k = length qsAndSums
    let n = p + wishartM + 1
    let c = f64.i64 (n * p) * (f64.log wishartGamma - 0.5 * f64.log 2) - (logGammaDistrib (0.5 * f64.i64 n) p)
    let frobenius = qsAndSums |> sumBy (fst >-> frobeniusNormSq)
    let sumQs = qsAndSums |> sumBy snd
    in 0.5 * wishartGamma * wishartGamma * frobenius - f64.i64 wishartM * sumQs - f64.i64 k * c

let gmmObjective [D][k] (alphas: [k]f64) (means: [k][D]f64) (icf: [k][]f64) (x: [][D]f64) (wishartGamma: f64) (wishartM: i64) =
    let d = length (transpose x)
    let n = length x
    let constant = -(f64.i64 n * f64.i64 d * 0.5 * f64.log (2 * f64.pi))
    let alphasAndMeans = zip alphas means
    let qsAndSums = icf |> map (\v ->
                                let logdiag = v[0:d]
                                let lt = v[d:]
                                in ((unpackQ logdiag lt) :> [D][D]f64, f64.sum logdiag))
    let slse = x |> sumBy (\xi ->
                    logsumexp_DArray <| map2
                        (\qAndSum alphaAndMeans ->
                            let (q, sumQ) = qAndSum
                            let (alpha, meansk) = alphaAndMeans
			    let qximeansk = flatten (transpose (linalg_f64.matmul q (transpose [vMinus xi meansk])))
                            in -0.5 * l2normSq qximeansk + alpha + sumQ
                        ) qsAndSums (alphasAndMeans :> [k](f64, [D]f64)))
    in constant + slse  - f64.i64 n * logsumexp alphas + logWishartPrior qsAndSums wishartGamma wishartM d

let grad f x = vjp f x 1f64

let fwd_grad [D][k][j] f (a : [k]f64, m :[k][D]f64, i : [k][j]f64 ) : []f64 =
  let a_zero = replicate k 0.0
  let m_zero = replicate k (replicate D 0.0)
  let i_zero = replicate k (replicate j 0.0)
  let one_hot_1d n idx = tabulate n (\i -> if idx == i then 1.0 else 0.0)
  let one_hot_1ds n = map (one_hot_1d n) (iota n)
  let one_hot_2d n m (idx, jdx) = tabulate_2d n m (\i j -> if (idx, jdx) == (i,j) then 1.0 else 0.0)
  let one_hot_2ds n m = flatten <| map (\i -> map (\j -> one_hot_2d n m (i, j)) (iota m)) (iota n)
  let a_grad = map (\a_d -> jvp f (a, m, i) (a_d, m_zero, i_zero)) (one_hot_1ds k)
  let m_grad = map (\m_d -> jvp f (a, m, i) (a_zero, m_d, i_zero)) (one_hot_2ds k D)
  let i_grad = map (\i_d -> jvp f (a, m, i) (a_zero, m_zero, i_d)) (one_hot_2ds k j)
  in a_grad ++ m_grad ++ i_grad

entry calculate_objective [D][k]
			  (alphas: [k]f64)
			  (means: [k][D]f64)
			  (icf: [k][]f64)
			  (x: [][D]f64)
			  (w_gamma: f64) (w_m: i64) =
    gmmObjective alphas means icf x w_gamma w_m

entry calculate_jacobian [D] [k]
			 (alphas: [k]f64)
			 (means: [k][D]f64)
			 (icf: [k][]f64)
			 (x: [][D]f64)
			 (w_gamma: f64) (w_m: i64) =
  grad (\(a, m, i) -> gmmObjective a m i x w_gamma w_m) (alphas, means, icf)

entry calculate_jacobian_fwd [D] [k]
			 (alphas: [k]f64)
			 (means: [k][D]f64)
			 (icf: [k][]f64)
			 (x: [][D]f64)
			 (w_gamma: f64) (w_m: i64) =
  fwd_grad (\(a, m, i) -> gmmObjective a m i x w_gamma w_m) (alphas, means, icf)
