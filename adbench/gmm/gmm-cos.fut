type real = f32
let log = f32.log
let exp = f32.exp
let sum = f32.sum
let toReal = f32.i64
let pi = f32.pi
let lgamma = f32.lgamma
let maximum = f32.maximum
let zero = 0.0f32
let one  = 1.0f32 

let fst (x,_) = x

let snd (_,y) = y

let sumBy 'a (f : a -> real)  (xs : []a) : real = map f xs |> sum

let l2normSq (v : []real) = sumBy (\x -> x * x) v

let logsumexp = sumBy exp >-> log

let frobeniusNormSq (mat : [][]real) = sumBy (\x -> x * x) (flatten mat)

let unpackQ [d] (logdiag: [d]real) (lt: []real) : [d][d]real  =
  tabulate_2d d d (\j i ->
                    if i < j then 0
                    else if i == j then exp logdiag[i]
                    else lt[d * j + i - j - 1 - j * (j + 1) / 2])

let logGammaDistrib (a : real) (p : i64) =
  0.25 * toReal p * toReal (p - 1) * log pi +
  ((1...p) |> sumBy (\j -> lgamma (a + 0.5 * toReal (1 - j))))

let logsumexp_DArray (arr : []real) =
    let mx = maximum arr
    let sumShiftedExp = arr |> sumBy (\x -> exp (x - mx))
    in log sumShiftedExp + mx

let logWishartPrior [k] (qs: [k][][]real) (sums: [k]real) wishartGamma wishartM p =
    let n = p + wishartM + 1
    let c = toReal (n * p) * (log wishartGamma - 0.5 * log 2) -
            (logGammaDistrib (0.5 * toReal n) p)
    let frobenius = sumBy frobeniusNormSq qs
    let sumQs = sum sums
    in 0.5 * wishartGamma * wishartGamma * frobenius - 
       toReal wishartM * sumQs - toReal k * c

let matmultr [d][n] (a: [d][n]real) (b: [d][d]real) : [n][d]real =
    map (\a_col -> 
            map (\b_col ->
                    map2 (*) a_col b_col  |> sum
                ) (transpose b)
        ) (transpose a)

let gmmObjective [d][k][n]
                 (alphas: [k]real)
                 (means:  [k][d]real)
                 (icf:    [k][] real)
                 (xtr:    [d][n]real)
                 (wishartGamma: real)
                 (wishartM: i64) =
    let constant = -(toReal n * toReal d * 0.5 * log (2 * pi))
    let logdiags = icf[:,:d]
    let lts = icf[:,d:]
    let qs = map2 unpackQ logdiags lts
    let diffs = tabulate_3d k d n
            (\ b q a -> xtr[q,a] - means[b, q]) |> opaque
    let qximeans_mats = map2 matmultr diffs qs |> opaque  -- : [k][n][d]real
    let tmp1 = map (map l2normSq) qximeans_mats |> opaque -- tmp1 : [k][n]real

    let sumQs= map sum logdiags -- sumQs : [k]real
    let tmp2 = map3(\ (row: [n]real) alpha sumQ ->
                        map (\ el -> 
                                -0.5 * el + alpha + sumQ
                            ) row
                   ) tmp1 alphas sumQs
        -- tmp2 : [k][n]real
    let tmp3  = map logsumexp_DArray (transpose tmp2)
    let slse  = sum tmp3
 
--    let onX xi =
--      map4 (\q sumQ alpha meansk ->
--              let qximeansk = linalg_f64.matvecmul_row q (map2 (-) xi meansk)
--              in -0.5 * l2normSq qximeansk + alpha + sumQ)
--           qs sumQs alphas means
--      |> logsumexp_DArray
--    let slse = sumBy onX x
    in constant + slse  - toReal n * logsumexp alphas +
       logWishartPrior qs sumQs wishartGamma wishartM d

let grad f x = vjp f x one

entry calculate_objective [d][k][n]
                          (alphas: [k]real)
                          (means: [k][d]real)
                          (icf: [k][]real)
                          (x:   [n][d]real)
                          (w_gamma: real) (w_m: i64) =
  gmmObjective alphas means icf (transpose x) w_gamma w_m

entry calculate_jacobian [d][k][n]
                         (alphas: [k]real)
                         (means: [k][d]real)
                         (icf: [k][]real)
                         (x:   [n][d]real)
                         (w_gamma: real) (w_m: i64) =
  let (alphas_, means_, icf_) =
    grad (\(a, m, i) -> gmmObjective a m i (transpose x) w_gamma w_m) (alphas, means, icf)
  in (alphas_, means_, icf_)

--output @ data/1k/gmm_d2_K10.F

-- ==
-- entry: calculate_objective
-- compiled random input { [200]f32 [200][128]f32 [200][8256]f32 [1024][128]f32 1.0f32 3i64 }

-- compiled random input { [300][1024][80]f32 [256][1024]f32 [256][1024]f32 [1024][80]f32 [1024][256]f32 [1024]f32 [1024]f32 [256][80]f32 [80]f32 f32 }

-- compiled input @ data/1k/gmm_d2_K10.in.gz
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


--output @ data/1k/gmm_d2_K10.J

-- ==
-- entry: calculate_jacobian
-- compiled random input { [200]f32 [200][128]f32 [200][8256]f32 [1024][128]f32 1.0f32 3i64 }

-- compiled input @ data/1k/gmm_d2_K10.in.gz
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
