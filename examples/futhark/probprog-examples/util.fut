import "lib/github.com/diku-dk/cpprandom/random"

module gamma_distribution (R: real) (E: rng_engine):
  rng_distribution with num.t = R.t
                   with engine.rng = E.rng
                   with distribution = {concentration: R.t, rate: R.t} = {
  module engine = E
  module num = R
  module norm_dist = normal_distribution R E
  module unif_dist = uniform_real_distribution R E

  type distribution = {concentration: R.t, rate: R.t}

  let rand ({concentration, rate}: distribution) (rng: E.rng) =
    assert (concentration R.> R.f32 0.0 && rate R.> R.f32 0.0)
           (let lc = concentration R.>= R.f32 1.0
            let conc = if lc then concentration else concentration R.+ R.f32 1.0
            let (rng, g, _) = loop (rng, _, acc) = (rng, R.f32 0.0, false)
                              while acc do
                              let d = conc R.- (R.f32 1.0 R./ R.f32 3.0)
                              let c = R.f32 1.0 R./ R.sqrt (R.f32 9.0 R.* d)
                              let (rng, z) = norm_dist.rand {mean=R.f32 0.0, stddev=R.f32 1.0} rng
                              let (rng, u) = unif_dist.rand (R.f32 0.0, R.f32 1.0) rng
                              let v = (R.f32 1.0 R.+ c R.* z) R.** R.f32 3.0
                              let acc = z R.> R.f32 (- 1.0) R./ c && R.log u R.< R.f32 0.5 R.* z R.** R.f32 2.0 R.+ d R.- d R.* v R.+ d R.* R.log v
                              in (rng, d R.* v, acc)
            let (rng, g) =
              if lc
              then (rng, g)
              else let (rng, u) = unif_dist.rand (R.f32 0.0, R.f32 1.0) rng
                   in (rng, g R.* (u R.** (R.f32 1.0 R./ concentration)))
            in (rng, g R./ rate))
}

module dirichlet_distribution (R: real) (E: rng_engine) = {
  -- Should be generalized to allow for more than one output, as in case of dirichlet
  --rng_distribution with num.t = R.t
  --                 with engine.rng = E.rng
  --                 with distribution = {concentrations: []R.t} =
  module engine = E
  module num = R
  module gamma_dist = gamma_distribution R E

  type distribution = {concentrations: []R.t}

  let rand ({concentrations}: distribution) (rng: E.rng): (E.rng, []R.t) =
    assert (length concentrations > 0 && all (R.> R.f32 0.0) concentrations)
           (let sample_conc ((rng, samples): (E.rng, []R.t)) (conc: R.t) =
              let (rng, a) = gamma_dist.rand {concentration=conc, rate=R.f32 1.0} rng
              in (rng, concat samples [a])
            in foldl sample_conc (rng, []) concentrations)
}

module beta_distribution (R: real) (E: rng_engine):
  rng_distribution with num.t = R.t
                   with engine.rng = E.rng
                   with distribution = {concentration0: R.t, concentration1: R.t} = {
  module engine = E
  module num = R
  module dirichlet_dist = dirichlet_distribution R E

  type distribution = {concentration0: R.t, concentration1: R.t}

  let rand ({concentration0, concentration1}: distribution) (rng: E.rng): (E.rng, R.t) =
    let (rng, a) = dirichlet_dist.rand {concentrations=[concentration1, concentration0]} rng
    in (rng, a[0])
}

module tensor_ops (R: real): {
    val sigmoid: R.t -> R.t
    val dotv [n][m]: [m][n]R.t -> [n]R.t -> [m]R.t
    val dotm [n][m][o]: [n][m]R.t -> [m][o]R.t -> [n][o]R.t
    val logsumexp: []R.t -> R.t
    val log_fact: R.t -> R.t
    val log_binomial: R.t -> R.t -> R.t
    val log_gammafn: R.t -> R.t
    val log_betafn: R.t -> R.t -> R.t
} = {
    let sigmoid (r: R.t): R.t = R.f32 1.0 R./ (R.f32 1.0 R.+ R.exp (R.negate r))

    let dotv [n][m] (yss: [m][n]R.t) (xs: [n]R.t): [m]R.t =
      map (\ys -> reduce (R.+) (R.f32 0.0) (map2 (R.*) xs ys)) yss

    let dotm [n][m][o] (xss: [n][m]R.t) (yss: [m][o]R.t): [n][o]R.t =
      map (\xs -> map (\ys -> reduce (R.+) (R.f32 0.0) (map2 (R.*) xs ys)) (transpose yss)) xss

    let max (x: R.t) (y: R.t) = if x R.> y then x else y

    let logsumexp (xs: []R.t): R.t =
      let xm = reduce max (R.f32 0.0) xs
      let expmax (xm: R.t) (x: R.t) = R.exp (x R.- xm)
      in R.log (reduce (R.+) (R.f32 0.0) (map (expmax xm) xs)) R.+ xm

    let log_fact (r: R.t): R.t =
      assert (r R.> R.f32 0.0 && R.floor r R.== r)
             (if r R.< R.f32 10.0 then
                R.log (loop n = R.f32 1.0 for i < R.to_i32 r do
                         (R.i32 i R.+ R.f32 1.0) R.* n)
              else r R.* R.log r R.- r R.+ R.f32 0.5 R.* R.log r R.+ R.f32 0.5 R.* R.log (R.f32 2.0 R.* R.pi) )

    let log_binomial (n: R.t) (k: R.t): R.t =
      log_fact n R.- (log_fact k R.+ log_fact (n R.- k))

    let log_gammafn (z: R.t) =
      let log_gammafn' (z: R.t) =
        let g = R.f32 5.0
        let z = z R.- R.f32 1.0
        let base = z R.+ g R.+ R.f32 0.5
        let lct = map R.f32 [-0.5395239384953e-5, 0.1208650973866179e-2, -1.231739572450155,
                             24.01409824083091, -86.50532032941677, 76.18009172947146, 1.000000000190015]

        let sum =
          loop sum = R.f32 0.0 for i < length lct do
            let j = (length lct - 1) - i
            in sum R.+ lct[j] R./ (z R.+ R.i32 j)
        in R.f32 0.5 R.* R.log (R.f32 2.0 R.* R.pi) R.+ R.log sum R.- base R.+ (R.log base R.* (z R.+ R.f32 0.5))
      in assert (z R.> R.f32 0) (if z R.< R.f32 0.5
                                 then
                                 let r = log_gammafn' (R.f32 1.0 R.- z)
                                 in R.log (R.pi R./ R.sin (R.pi R.* z)) R.- r
                                 else log_gammafn' z)

    let log_betafn (z: R.t) (w: R.t) = log_gammafn z R.+ log_gammafn w R.- log_gammafn (z R.+ w)
}

module type sized = {
  val len: i32
}

module arr (R: real) (S: sized): real with t = [S.len]R.t = {
  type t = [S.len]R.t
  let i8 (v: i8): t = replicate S.len (R.i8 v)
  let i16 (v: i16): t = replicate S.len (R.i16 v)
  let i32 (v: i32): t = replicate S.len (R.i32 v)
  let i64 (v: i64): t = replicate S.len (R.i64 v)
  let u8 (v: u8): t = replicate S.len (R.u8 v)
  let u16 (v: u16): t = replicate S.len (R.u16 v)
  let u32 (v: u32): t = replicate S.len (R.u32 v)
  let u64 (v: u64): t = replicate S.len (R.u64 v)
  let f32 (v: f32): t = replicate S.len (R.f32 v)
  let f64 (v: f64): t = replicate S.len (R.f64 v)
  let bool (v: bool): t = replicate S.len (R.bool v)
  let (+) = map2 (R.+)
  let (-) = map2 (R.-)
  let (*) = map2 (R.*)
  let (/) = map2 (R./)
  let (**) = map2 (R.**)
  let to_i32 (_xs: t): i32 = assert false (- 1)
  let to_i64 (_xs: t): i64 = assert false (- 1)
  let to_f64 (_xs: t): f64 = assert false 0.0
  let (==) xs ys = map2 (R.==) xs ys |> reduce (&&) true
  let (<) xs ys = map2 (R.<) xs ys |> reduce (&&) true
  let (>) xs ys = map2 (R.>) xs ys |> reduce (&&) true
  let (<=) xs ys = map2 (R.<=) xs ys |> reduce (&&) true
  let (>=) xs ys = map2 (R.>=) xs ys |> reduce (&&) true
  let (!=) xs ys = map2 (R.!=) xs ys |> reduce (||) false
  let negate = map R.negate
  let max = map2 R.max
  let min = map2 R.min
  let abs = map R.abs
  let sgn = map R.sgn
  let highest = replicate S.len R.highest
  let lowest = replicate S.len R.lowest
  let sum = map R.sum
  let product = map R.product
  let maximum = map R.maximum
  let minimum = map R.minimum
  let from_fraction n d = replicate S.len (R.from_fraction n d)
  let sqrt = map R.sqrt
  let exp = map R.exp
  let cos = map R.cos
  let sin = map R.sin
  let tan = map R.tan
  let asin = map R.asin
  let acos = map R.acos
  let atan = map R.atan
  let atan2 = map2 R.atan2
  let log = map R.log
  let log2 = map R.log2
  let log10 = map R.log10
  let ceil = map R.ceil
  let floor = map R.floor
  let trunc = map R.trunc
  let round = map R.round
  let isinf (xs: t) = map R.isinf xs |> reduce (||) false
  let isnan (xs: t) = map R.isnan xs |> reduce (||) false
  let inf = replicate S.len R.inf
  let nan = replicate S.len R.nan
  let pi = replicate S.len R.pi
  let e = replicate S.len R.e
}
