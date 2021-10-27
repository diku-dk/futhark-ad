type real = f32
let  sum = f32.sum
let  exp = f32.exp
let  to_real = f32.i64
let  sin = f32.sin

let matmul [n][m][p] (A: [n][m]real) (B: [m][p]real) : [n][p]real =
  map (\A_row ->
         map (\B_col ->
                reduce (+) 0 (map2 (*) A_row B_col)
             ) (transpose B)
      ) A

let matvec_mul [n][m] (A: [m][n]real) (B: [n]real) : [m]real =
  map (\A_row -> reduce (+) 0 (map2 (*) A_row B)) A

let vector_add [n] (x: [n]real) (y: [n]real) : [n]real =
  map2 (+) x y

let mat_add [m][n]
            (x: [m][n]real) (y: [m][n]real)
            : [m][n]real =
  map2 (\a b -> map2 (+) a b) x y

let vector_mul [n] (x: [n]real) (y: [n]real) : [n]real =
  map2 (*) x y

let recurrent_layer_activation (x: real) : real =
  (exp x - exp (-1.0 * x)) / (exp x + exp (-1.0 * x))

let elman_layer [d]
          (x: [d]real) (last_h: [d]real) (wh: [d][d]real) (u: [d][d]real) (bh: [d]real)
          : ([d]real, [d]real) =
  let wx  = matvec_mul wh x
  let ulh = matvec_mul u last_h
  let h   = map (recurrent_layer_activation) (vector_add (vector_add wx ulh) bh)
  in (h, h)

let elman_predict [d][layers]
          (input: [d]real) (last_hs: [layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real) (bh: [layers][d]real)
          : ([d]real, [layers][d]real) =
  let x0 = input
  let state_ini = replicate layers <| replicate d 0
  let (state', x') =
    loop (#[true_dep]s, x) = (state_ini, x0)
    for i < layers do
      let (h, h') = elman_layer x last_hs[i] wh[i] u[i] bh[i]
      let s[i] = h'
      in  (s, h)
  in (x', state')

let elman_rnn [n][d][layers]
          (inputs: [n][d]real) (first_hs: [layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real) (bh: [layers][d]real)
          : [d]real =
  let h'' =
    #[stripmine(2)]
    loop (h) = (first_hs)
    for i < n-1 do
      let (_, h') = elman_predict inputs[i] h wh u bh
      in  h'
  let (pred', _) = elman_predict inputs[n-1] h'' wh u bh
  in pred'

let mse_loss_function [n] (a: [n]real) (b: [n]real) : real =
  (reduce (+) 0 (map2 (\a b -> (a - b) ** 2.0) a b)) / f32.i64(n)

let objFun [n][d][layers]
          (inputs: [n][d]real) (first_hs: [layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real)
          (bh: [layers][d]real)    (true_values: [n][d]real) : real =
  let last_pred = elman_rnn inputs first_hs wh u bh
  in  mse_loss_function last_pred (last true_values)


let linspace (start: f32) (end: f32) (steps: i64) =
  tabulate steps (\step -> start + f32.i64 step * (end - start)/(f32.i64 steps - 1.0))


-- trivial: n=1, l=1, d=3
let main (n: i64) (l: i64) (d: i64) =

  let wh = unflatten_3d l d d (linspace 0.1 0.2 (l*d*d))
  let u  = unflatten_3d l d d (linspace 0.1 0.2 (l*d*d))
  let bh = unflatten    l d   (linspace 0.1 0.2 (l*d))

  let inputs  = unflatten n d (linspace 0.1 0.2 (n*d))
  let first_h = unflatten l d (linspace 0.1 0.2 (l*d))
  let true_values = unflatten n d (linspace 100.0 300.0 (n*d))

  let objFun' x y z (a, b, c) = objFun x y a b c z
  let result = vjp (objFun' inputs first_h true_values) (wh, u, bh) 1.0

  in result

------------------------
------------------------
let elman_rnn_gen [n][d][layers]
          (inputs: [n][d]real) (first_hs: [layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real) (bh: [layers][d]real)
          : ([n][d]real, [][layers][d]real, [layers][d]real) =
  let predictions = replicate n <| replicate d 0
  let state_ini = replicate (n + 1) <| replicate layers <| replicate d 0
  let state_ini[0] = first_hs
  let (preds', states', final_hs') =
    loop (p, s, hs) = (predictions, state_ini, first_hs)
    for i < n do
      let (pred, h') = elman_predict inputs[i] hs wh u bh
      let p[i] = pred
      let s[i + 1] = h'
      in  (p, s, h')
  in (preds', states', final_hs')

let objFun_gen [n][d][layers]
          (inputs: [n][d]real) (first_hs: [layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real)
          (bh: [layers][d]real)    (true_values: [n][d]real) : real =
  let (preds, _, _) = elman_rnn_gen inputs first_hs wh u bh
  in mse_loss_function (last preds) (last true_values)
