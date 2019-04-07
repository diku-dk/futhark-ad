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
    val logsumexp: []R.t -> R.t
    val log_fact: R.t -> R.t
    val log_binomial: R.t -> R.t -> R.t
    val log_gammafn: R.t -> R.t
    val log_betafn: R.t -> R.t -> R.t
} = {
    let sigmoid (r: R.t): R.t = R.f32 1.0 R./ (R.f32 1.0 R.+ R.exp (R.f32 0.0 R.- r))

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

module T = tensor_ops (f32)
let main = map T.sigmoid [1.0, 2.0, 100.0]
