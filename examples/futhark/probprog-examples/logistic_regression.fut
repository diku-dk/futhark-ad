import "util"
import "optim"

module T = tensor_ops f64

module logistic_regression (ws_size: sized) : {
  val data_len: i64
  include grad_optimizable with param.t = [ws_size.len]f64
                           with loss.t = f64
                           with data = ([][data_len]f64, []f64)
  val logreg [m]: [ws_size.len]f64 -> [m][data_len]f64 -> [m]f64
} = {
  let data_len: i64 = ws_size.len - 1

  type~ data = ([][data_len]f64, []f64)
  module param = arr f64 ws_size
  module loss = f64

  let logreg [m] (ws: [ws_size.len]f64) (xss: [m][data_len]f64): [m]f64 =
    assert (ws_size.len > 1)
           (let xss = map (concat_to ws_size.len [1.0]) xss
            in map T.sigmoid (T.dotv xss ws))

  let cross_entropy [m] (ysp: [m]f64) (ysc: [m]f64): f64 =
    let eps: f64 = 1e-5
    in reduce (+) 0.0 (map2 (\yp yc -> 1 / f64.i64 m * (- ((yc + eps) * f64.log (yp + eps))) + (- (1 - yc + eps) * f64.log (1 - yp + eps))) ysc ysp)

  let eval_loss [m] (ws: [ws_size.len]f64) ((xss, ysc): ([m][data_len]f64, [m]f64)): f64 =
    cross_entropy (logreg ws xss) ysc

  let grad [m] (ws: [ws_size.len]f64) ((xss, ysc): ([m][data_len]f64, [m]f64)): [ws_size.len]f64 =
    let xss' = map (concat_to ws_size.len [1.0]) xss
    let lossdif: [m]f64 = (map2 (-) (logreg ws xss) ysc)
    in T.dotv (transpose xss') lossdif
}

module logreg_m = logistic_regression { let len:i32 = 3 }
module sgd = stochastic_gradient_descent logreg_m minstd_rand

let main =
  let test_data: [5][2]f64 = [[13.0, 15.0],
                              [23.0, 12.0],
                              [15.0, 0.0],
                              [17.0, -14.0],
                              [12.0, -5.0]]
  let test_ys: [5]f64 = [1.0, 1.0, 0.0, 0.0, 0.0]
  let init_ws: [3]f64 = [-5.0, -5.0, 5.0]
  let (_, losses, ws) = sgd.run {learning_rate=0.01} (minstd_rand.rng_from_seed [1337]) init_ws (test_data, test_ys) 10
  in (losses, ws, logreg_m.logreg ws test_data)
