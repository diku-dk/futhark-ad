import "util"
import "optim"

module T = tensor_ops f32

module logistic_regression (ws_size: sized) : {
  val data_len: i32
  include grad_optimizable with param.t = [ws_size.len]f32
                           with loss.t = f32
                           with data = ([][data_len]f32, []f32)
  val logreg [m]: [ws_size.len]f32 -> [m][data_len]f32 -> [m]f32
} = {
  let data_len: i32 = ws_size.len - 1

  type data = ([][data_len]f32, []f32)
  module param = arr f32 ws_size
  module loss = f32

  let logreg [m] (ws: [ws_size.len]f32) (xss: [m][data_len]f32): [m]f32 =
    assert (ws_size.len > 1)
           (let xss = concat [replicate m 1.0] xss
            in map T.sigmoid (T.dotv xss ws))

  let cross_entropy [m] (ysp: [m]f32) (ysc: [m]f32): f32 =
    reduce (+) 0.0 (map2 (\yp yc -> 1 / f32.i32 m * (- (yc * f32.log yp)) + (- (1 - yc) * f32.log (1 - yp))) ysc ysp)

  let eval_loss [m] (ws: [ws_size.len]f32) ((xss, ysc): ([m][data_len]f32, [m]f32)): f32 =
    cross_entropy (logreg ws xss) ysc

  let grad [n'][n][m] (ws: [n']f32) ((xss, ysc): ([m][n]f32, [m]f32)): [n']f32 =
    assert (n' == n + 1 && n > 0)
          (let xss' = concat [replicate m 1.0] xss
            in T.dotv xss' (map2 (-) (logreg ws xss) ysc))
}

module logreg_m = logistic_regression { let len:i32 = 3 }
module sgd = stochastic_gradient_descent logreg_m minstd_rand

let main =
  let test_data: [5][2]f32 = [[13.0, 15.0],
                              [23.0, 12.0],
                              [15.0, 0.0],
                              [17.0, -14.0],
                              [12.0, -5.0]]
  let test_ys: [5]f32 = [1.0, 1.0, 0.0, 0.0, 0.0]
  let init_ws: [3]f32 = [-5.0, -5.0, 5.0]
  let (_, losses, ws) = sgd.run {learning_rate=1.0} (minstd_rand.rng_from_seed [1337]) init_ws (test_data, test_ys) 10
  in (losses, ws)
