import "lib/github.com/diku-dk/cpprandom/random"

module gamma_distribution (R: real) (E: rng_engine):
  rng_distribution with num.t = R.t
                   with engine.rng = E.rng
                   with distribution = {concentration: R.t, rate: R.t} = {
  let to_R (x: E.int.t) =
    R.u64 (u64.i64 (E.int.to_i64 x))

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
