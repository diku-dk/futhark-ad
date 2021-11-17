let fst (x,_) = x

let snd (_,y) = y

let sumBy 'a (f : a -> f64)  (xs : []a) : f64 = map f xs |> f64.sum

let l2normSq (v : []f64) = sumBy (\x -> x * x) v

let logsumexp = sumBy f64.exp >-> f64.log

let frobeniusNormSq (mat : [][]f64) = sumBy (\x -> x * x) (flatten mat)

let unpackQ [d] (logdiag: [d]f64) (lt: []f64) : [d][d]f64  =
  tabulate_2d d d (\j i ->
                    if i < j then 0
                    else if i == j then f64.exp logdiag[i]
                    else lt[d * j + i - j - 1 - j * (j + 1) / 2])

let logGammaDistrib (a : f64) (p : i64) =
  0.25 * f64.i64 p * f64.i64 (p - 1) * f64.log f64.pi +
  ((1...p) |> sumBy (\j -> f64.lgamma (a + 0.5 * f64.i64 (1 - j))))

let logsumexp_DArray (arr : []f64) =
    let mx = f64.maximum arr
    let sumShiftedExp = arr |> sumBy (\x -> f64.exp (x - mx))
    in f64.log sumShiftedExp + mx

let logWishartPrior [k] (qs: [k][][]f64) (sums: [k]f64) wishartGamma wishartM p =
    let n = p + wishartM + 1
    let c = f64.i64 (n * p) * (f64.log wishartGamma - 0.5 * f64.log 2) -
            (logGammaDistrib (0.5 * f64.i64 n) p)
    let frobenius = sumBy frobeniusNormSq qs
    let sumQs = f64.sum sums
    in 0.5 * wishartGamma * wishartGamma * frobenius -
       f64.i64 wishartM * sumQs - f64.i64 k * c

let matmultr [d][n] (a: [d][n]f64) (b: [d][d]f64) : [n][d]f64 =
    map (\a_col ->
            map (\b_col ->
                    map2 (*) a_col b_col  |> f64.sum
                ) (transpose b)
        ) (transpose a)

let gmmObjective [d][k][n]
                 (alphas: [k]f64)
                 (means:  [k][d]f64)
                 (icf:    [k][] f64)
                 (xtr:    [d][n]f64)
                 (wishartGamma: f64)
                 (wishartM: i64) =
    let constant = -(f64.i64 n * f64.i64 d * 0.5 * f64.log (2 * f64.pi))
    let logdiags = icf[:,:d]
    let lts = icf[:,d:]
    let qs = map2 unpackQ logdiags lts
    let diffs = tabulate_3d k d n
            (\ b q a -> xtr[q,a] - means[b, q]) |> opaque
    let qximeans_mats = map2 matmultr diffs qs |> opaque  -- : [k][n][d]f64
    let tmp1 = map (map l2normSq) qximeans_mats |> opaque -- tmp1 : [k][n]f64

    let sumQs= map f64.sum logdiags -- sumQs : [k]f64
    let tmp2 = map3(\ (row: [n]f64) alpha sumQ ->
                        map (\ el ->
                                -0.5 * el + alpha + sumQ
                            ) row
                   ) tmp1 alphas sumQs
        -- tmp2 : [k][n]f64
    let tmp3  = map logsumexp_DArray (transpose tmp2)
    let slse  = f64.sum tmp3

--    let onX xi =
--      map4 (\q sumQ alpha meansk ->
--              let qximeansk = linalg_f64.matvecmul_row q (map2 (-) xi meansk)
--              in -0.5 * l2normSq qximeansk + alpha + sumQ)
--           qs sumQs alphas means
--      |> logsumexp_DArray
--    let slse = sumBy onX x
    in constant + slse  - f64.i64 n * logsumexp alphas +
       logWishartPrior qs sumQs wishartGamma wishartM d

let grad f x = vjp f x 1f64

entry calculate_objective [d][k][n]
                          (alphas: [k]f64)
                          (means: [k][d]f64)
                          (icf: [k][]f64)
                          (x:   [n][d]f64)
                          (w_gamma: f64) (w_m: i64) =
  gmmObjective alphas means icf (transpose x) w_gamma w_m

entry calculate_jacobian [d][k][n]
                         (alphas: [k]f64)
                         (means: [k][d]f64)
                         (icf: [k][]f64)
                         (x:   [n][d]f64)
                         (w_gamma: f64) (w_m: i64) =
  let (alphas_, means_, icf_) =
    grad (\(a, m, i) -> gmmObjective a m i (transpose x) w_gamma w_m) (alphas, means, icf)
  in (alphas_, means_, icf_)

-- ==
-- entry: calculate_objective
-- compiled input @ data/1k/gmm_d2_K10.in.gz output @ data/1k/gmm_d2_K10.F
-- compiled input @ data/1k/gmm_d64_K100.in.gz
-- compiled input @ data/1k/gmm_d2_K200.in.gz
-- compiled input @ data/1k/gmm_d10_K10.in.gz
-- compiled input @ data/1k/gmm_d20_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K5.in.gz
-- compiled input @ data/1k/gmm_d2_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K50.in.gz
-- compiled input @ data/1k/gmm_d20_K200.in.gz
-- compiled input @ data/1k/gmm_d64_K10.in.gz
-- compiled input @ data/1k/gmm_d10_K50.in.gz
-- compiled input @ data/1k/gmm_d128_K50.in.gz
-- compiled input @ data/1k/gmm_d2_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K25.in.gz
-- compiled input @ data/1k/gmm_d32_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K200.in.gz
-- compiled input @ data/1k/gmm_d20_K25.in.gz
-- compiled input @ data/1k/gmm_d128_K25.in.gz
-- compiled input @ data/1k/gmm_d128_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K200.in.gz
-- compiled input @ data/1k/gmm_d128_K200.in.gz
-- compiled input @ data/1k/gmm_d10_K100.in.gz
-- compiled input @ data/1k/gmm_d128_K10.in.gz
-- compiled input @ data/1k/gmm_d10_K5.in.gz
-- compiled input @ data/1k/gmm_d20_K50.in.gz
-- compiled input @ data/1k/gmm_d128_K5.in.gz
-- compiled input @ data/1k/gmm_d10_K200.in.gz
-- compiled input @ data/1k/gmm_d2_K25.in.gz
-- compiled input @ data/1k/gmm_d20_K10.in.gz
-- compiled input @ data/1k/gmm_d32_K10.in.gz
-- compiled input @ data/1k/gmm_d20_K100.in.gz
-- compiled input @ data/1k/gmm_d2_K50.in.gz
-- compiled input @ data/1k/gmm_d32_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K25.in.gz
-- compiled input @ data/1k/gmm_d10_K25.in.gz
-- compiled input @ data/1k/gmm_d64_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K200.in.gz
-- compiled input @ data/10k/gmm_d10_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K5.in.gz
-- compiled input @ data/10k/gmm_d2_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K200.in.gz
-- compiled input @ data/10k/gmm_d64_K10.in.gz
-- compiled input @ data/10k/gmm_d10_K50.in.gz
-- compiled input @ data/10k/gmm_d128_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K25.in.gz
-- compiled input @ data/10k/gmm_d32_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K200.in.gz
-- compiled input @ data/10k/gmm_d20_K25.in.gz
-- compiled input @ data/10k/gmm_d128_K25.in.gz
-- compiled input @ data/10k/gmm_d128_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K200.in.gz
-- compiled input @ data/10k/gmm_d128_K200.in.gz
-- compiled input @ data/10k/gmm_d10_K100.in.gz
-- compiled input @ data/10k/gmm_d128_K10.in.gz
-- compiled input @ data/10k/gmm_d10_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K100.in.gz
-- compiled input @ data/10k/gmm_d20_K50.in.gz
-- compiled input @ data/10k/gmm_d128_K5.in.gz
-- compiled input @ data/10k/gmm_d10_K200.in.gz
-- compiled input @ data/10k/gmm_d2_K25.in.gz
-- compiled input @ data/10k/gmm_d20_K10.in.gz
-- compiled input @ data/10k/gmm_d32_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K100.in.gz
-- compiled input @ data/10k/gmm_d2_K50.in.gz
-- compiled input @ data/10k/gmm_d32_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K25.in.gz
-- compiled input @ data/10k/gmm_d10_K25.in.gz
-- compiled input @ data/10k/gmm_d64_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K200.in.gz
-- compiled input @ data/2.5M/gmm_d10_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K5.in.gz
-- compiled input @ data/2.5M/gmm_d2_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K200.in.gz
-- compiled input @ data/2.5M/gmm_d64_K10.in.gz
-- compiled input @ data/2.5M/gmm_d10_K50.in.gz
-- compiled input @ data/2.5M/gmm_d128_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K25.in.gz
-- compiled input @ data/2.5M/gmm_d32_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K200.in.gz
-- compiled input @ data/2.5M/gmm_d20_K25.in.gz
-- compiled input @ data/2.5M/gmm_d128_K25.in.gz
-- compiled input @ data/2.5M/gmm_d128_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K200.in.gz
-- compiled input @ data/2.5M/gmm_d128_K200.in.gz
-- compiled input @ data/2.5M/gmm_d10_K100.in.gz
-- compiled input @ data/2.5M/gmm_d128_K10.in.gz
-- compiled input @ data/2.5M/gmm_d10_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K100.in.gz
-- compiled input @ data/2.5M/gmm_d20_K50.in.gz
-- compiled input @ data/2.5M/gmm_d128_K5.in.gz
-- compiled input @ data/2.5M/gmm_d10_K200.in.gz
-- compiled input @ data/2.5M/gmm_d2_K25.in.gz
-- compiled input @ data/2.5M/gmm_d20_K10.in.gz
-- compiled input @ data/2.5M/gmm_d32_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K100.in.gz
-- compiled input @ data/2.5M/gmm_d2_K50.in.gz
-- compiled input @ data/2.5M/gmm_d32_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K25.in.gz
-- compiled input @ data/2.5M/gmm_d10_K25.in.gz
-- compiled input @ data/2.5M/gmm_d64_K50.in.gz

-- ==
-- entry: calculate_jacobian
-- compiled input @ data/1k/gmm_d2_K10.in.gz output @ data/1k/gmm_d2_K10.J
-- compiled input @ data/1k/gmm_d64_K100.in.gz
-- compiled input @ data/1k/gmm_d2_K200.in.gz
-- compiled input @ data/1k/gmm_d10_K10.in.gz
-- compiled input @ data/1k/gmm_d20_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K5.in.gz
-- compiled input @ data/1k/gmm_d2_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K50.in.gz
-- compiled input @ data/1k/gmm_d20_K200.in.gz
-- compiled input @ data/1k/gmm_d64_K10.in.gz
-- compiled input @ data/1k/gmm_d10_K50.in.gz
-- compiled input @ data/1k/gmm_d128_K50.in.gz
-- compiled input @ data/1k/gmm_d2_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K25.in.gz
-- compiled input @ data/1k/gmm_d32_K5.in.gz
-- compiled input @ data/1k/gmm_d64_K200.in.gz
-- compiled input @ data/1k/gmm_d20_K25.in.gz
-- compiled input @ data/1k/gmm_d128_K25.in.gz
-- compiled input @ data/1k/gmm_d128_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K200.in.gz
-- compiled input @ data/1k/gmm_d128_K200.in.gz
-- compiled input @ data/1k/gmm_d10_K100.in.gz
-- compiled input @ data/1k/gmm_d128_K10.in.gz
-- compiled input @ data/1k/gmm_d10_K5.in.gz
-- compiled input @ data/1k/gmm_d20_K50.in.gz
-- compiled input @ data/1k/gmm_d128_K5.in.gz
-- compiled input @ data/1k/gmm_d10_K200.in.gz
-- compiled input @ data/1k/gmm_d2_K25.in.gz
-- compiled input @ data/1k/gmm_d20_K10.in.gz
-- compiled input @ data/1k/gmm_d32_K10.in.gz
-- compiled input @ data/1k/gmm_d20_K100.in.gz
-- compiled input @ data/1k/gmm_d2_K50.in.gz
-- compiled input @ data/1k/gmm_d32_K100.in.gz
-- compiled input @ data/1k/gmm_d32_K25.in.gz
-- compiled input @ data/1k/gmm_d10_K25.in.gz
-- compiled input @ data/1k/gmm_d64_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K200.in.gz
-- compiled input @ data/10k/gmm_d10_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K5.in.gz
-- compiled input @ data/10k/gmm_d2_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K200.in.gz
-- compiled input @ data/10k/gmm_d64_K10.in.gz
-- compiled input @ data/10k/gmm_d10_K50.in.gz
-- compiled input @ data/10k/gmm_d128_K50.in.gz
-- compiled input @ data/10k/gmm_d2_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K25.in.gz
-- compiled input @ data/10k/gmm_d32_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K200.in.gz
-- compiled input @ data/10k/gmm_d20_K25.in.gz
-- compiled input @ data/10k/gmm_d128_K25.in.gz
-- compiled input @ data/10k/gmm_d128_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K200.in.gz
-- compiled input @ data/10k/gmm_d128_K200.in.gz
-- compiled input @ data/10k/gmm_d10_K100.in.gz
-- compiled input @ data/10k/gmm_d128_K10.in.gz
-- compiled input @ data/10k/gmm_d10_K5.in.gz
-- compiled input @ data/10k/gmm_d64_K100.in.gz
-- compiled input @ data/10k/gmm_d20_K50.in.gz
-- compiled input @ data/10k/gmm_d128_K5.in.gz
-- compiled input @ data/10k/gmm_d10_K200.in.gz
-- compiled input @ data/10k/gmm_d2_K25.in.gz
-- compiled input @ data/10k/gmm_d20_K10.in.gz
-- compiled input @ data/10k/gmm_d32_K10.in.gz
-- compiled input @ data/10k/gmm_d20_K100.in.gz
-- compiled input @ data/10k/gmm_d2_K50.in.gz
-- compiled input @ data/10k/gmm_d32_K100.in.gz
-- compiled input @ data/10k/gmm_d32_K25.in.gz
-- compiled input @ data/10k/gmm_d10_K25.in.gz
-- compiled input @ data/10k/gmm_d64_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K200.in.gz
-- compiled input @ data/2.5M/gmm_d10_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K5.in.gz
-- compiled input @ data/2.5M/gmm_d2_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K200.in.gz
-- compiled input @ data/2.5M/gmm_d64_K10.in.gz
-- compiled input @ data/2.5M/gmm_d10_K50.in.gz
-- compiled input @ data/2.5M/gmm_d128_K50.in.gz
-- compiled input @ data/2.5M/gmm_d2_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K25.in.gz
-- compiled input @ data/2.5M/gmm_d32_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K200.in.gz
-- compiled input @ data/2.5M/gmm_d20_K25.in.gz
-- compiled input @ data/2.5M/gmm_d128_K25.in.gz
-- compiled input @ data/2.5M/gmm_d128_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K200.in.gz
-- compiled input @ data/2.5M/gmm_d128_K200.in.gz
-- compiled input @ data/2.5M/gmm_d10_K100.in.gz
-- compiled input @ data/2.5M/gmm_d128_K10.in.gz
-- compiled input @ data/2.5M/gmm_d10_K5.in.gz
-- compiled input @ data/2.5M/gmm_d64_K100.in.gz
-- compiled input @ data/2.5M/gmm_d20_K50.in.gz
-- compiled input @ data/2.5M/gmm_d128_K5.in.gz
-- compiled input @ data/2.5M/gmm_d10_K200.in.gz
-- compiled input @ data/2.5M/gmm_d2_K25.in.gz
-- compiled input @ data/2.5M/gmm_d20_K10.in.gz
-- compiled input @ data/2.5M/gmm_d32_K10.in.gz
-- compiled input @ data/2.5M/gmm_d20_K100.in.gz
-- compiled input @ data/2.5M/gmm_d2_K50.in.gz
-- compiled input @ data/2.5M/gmm_d32_K100.in.gz
-- compiled input @ data/2.5M/gmm_d32_K25.in.gz
-- compiled input @ data/2.5M/gmm_d10_K25.in.gz
-- compiled input @ data/2.5M/gmm_d64_K50.in.gz
