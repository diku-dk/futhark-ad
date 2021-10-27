type real = f32
let  sum = f32.sum
let  exp = f32.exp
let  to_real = f32.i64
let  sin = f32.sin

let scan_exc 't [n] (op: t->t->t) (ne: t) (arr: [n]t) : [n]t =
  map (\i -> if i==0 then ne else arr[i-1]) (iota n)
  |> scan op ne

let myfilter 't [n] (dummy: t) (pred: t -> bool) (arr: [n]t) : []t =
  let indsT = map (\el -> if pred el then 1i64 else 0) arr
            |> opaque
  let indssc = scan (+) 0 indsT
            |> opaque
  let len = indssc[n-1]
  let inds = map2 (\ii v -> if v == 1 then ii-1 else -1) indssc indsT |> opaque
  in  scatter (replicate len dummy) inds arr

let scatter3D [q][q'][layers][d]
              (valid_sgm_inds: [q']i64)
              (hs:      *[layers][q ][d]real)
              (hs_updt:  [layers][q'][d]real) :
              *[layers][q][d]real =
  let qd' = q'*d
  let qd  = q *d
  let indvals = 
    tabulate (layers*qd')
      (\ii -> let k1  = ii / qd'
              let k_r = ii - k1*qd'
              let k2_0= k_r / d 
              let k2  = valid_sgm_inds[k2_0]
              let k3  = k_r - k2_0 * d
              in  (k1*qd + k2*d + k3, hs_updt[k1,k2_0,k3])
      )
  let (inds, vals) = unzip indvals |> opaque
  let hs_flat = flatten (flatten hs)
  let hs_flat' = scatter hs_flat inds vals
  in  unflatten layers q (unflatten (layers*q) d hs_flat')

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

let elman_layer_batch [q][d]
          (xs: [q][d]real) (last_hs: [q][d]real) (wh: [d][d]real) (u: [d][d]real) (bh: [d]real)
          : ([q][d]real, [q][d]real) =
  let wxs = map (\x -> matvec_mul wh x) xs --|> opaque
  let ulhs= map (\last_h -> matvec_mul u last_h) last_hs --|> opaque
  let h   = map2 (\wx ulh -> map recurrent_layer_activation (vector_add (vector_add wx ulh) bh)) wxs ulhs
  in  (h, h)

let elman_predict_batch [q][d][layers]
          (input: [q][d]real) (last_hs: [layers][q][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real) (bh: [layers][d]real)
          : ([q][d]real, [layers][q][d]real) =
  let x0 = input
  let state_ini = replicate layers <| replicate q <| replicate d 0
  let (state', x') =
    loop (s, x) = (state_ini, x0)
    for i < layers do
      let (hs, hs') = elman_layer_batch x last_hs[i] wh[i] u[i] bh[i]
      let s[i] = hs'
      in  (s, hs)
  in (x', state')

let elman_rnn_batch [q][shpsum][d][layers]
          (shape: [q]i64) (beg_inds: [q]i64) (flat_inputs: [shpsum][d]real) -- (inputs: [q][n][d]real) 
          (first_hs: [q][layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real) (bh: [layers][d]real)
          : [q][d]real =
  let first_hs_tr = copy (transpose first_hs)

  -- let valid_inps = map (\j -> flat_inputs[beg_inds[j]+i]) valid_sgm_inds
  let getFlatInps i =
        map(\j -> tabulate d (\k -> flat_inputs[beg_inds[j]+i, k]) )

  let max_n = reduce i64.max 0 shape
  let hs'' =
    loop (hs) = (first_hs_tr)
    for i < max_n-1 do
      let valid_sgm_inds = myfilter 0i64 (\j -> i < shape[j]-1) (indices shape) -- length q or less

      let valid_inps = getFlatInps i valid_sgm_inds
      let valid_hs   = tabulate layers 
                        (\k1 -> map (\j -> tabulate d (\k2 -> hs[k1,j,k2])) valid_sgm_inds)
                      --map (\j -> tabulate2d layers d (\k1 k2 -> hs[k1,j,k2])) valid_sgm_inds

      let (_, valid_hs') = elman_predict_batch valid_inps valid_hs wh u bh

      let hs' = scatter3D valid_sgm_inds hs valid_hs'
      in  hs'
  -- COSMIN IS HERE: FIX THIS!!!
  let inps_last = map2 (\beg n -> tabulate d (\k -> flat_inputs[beg+n-1, k])) beg_inds shape
  let (pred', _) = elman_predict_batch inps_last hs'' wh u bh
  in pred'

let mse_loss_function [n] (a: [n]real) (b: [n]real) : real =
  (reduce (+) 0 (map2 (\a b -> (a - b) ** 2.0) a b)) / f32.i64(n)

-- batch size is q!
let objFunBatch [q][shpsum][d][layers]
          (shape: [q]i64) (flat_inputs: [shpsum][d]real) --(inputs: [q][n][d]real)
          (first_hs: [q][layers][d]real)
          (wh: [layers][d][d]real) (u: [layers][d][d]real)
          (bh: [layers][d]real)
          (true_vals: [shpsum][d]real) : real =
  let beg_inds = scan_exc (+) 0 shape
  let last_preds = elman_rnn_batch shape beg_inds flat_inputs first_hs wh u bh
  let last_true_vals = map2 (\beg n -> tabulate d (\k -> true_vals[beg+n-1,k])) beg_inds shape
  let losses = map2 mse_loss_function last_preds last_true_vals
  in  reduce (+) 0 losses -- ToDo: is it correct???


--------------------
--- Main
--------------------
let linspace (start: f32) (end: f32) (steps: i64) =
  tabulate steps (\step -> start + f32.i64 step * (end - start)/(f32.i64 steps - 1.0))

-- trivial: n=1, l=1, d=3
let main (q: i64) (n: i64) (l: i64) (d: i64) =

  let wh = unflatten_3d l d d (linspace 0.1 0.2 (l*d*d))
  let u  = unflatten_3d l d d (linspace 0.1 0.2 (l*d*d))
  let bh = unflatten    l d   (linspace 0.1 0.2 (l*d))

  let qr = to_real q
  let shape = replicate q n
  let inputs  = unflatten (q*n) d (linspace 0.1 (0.2*qr) (q*n*d))
  let first_h = unflatten_3d q l d (linspace 0.1 (0.2*qr) (q*l*d))
  let true_values = unflatten (q*n) d (linspace 1.0 (3.0*qr) (q*n*d))

  let objFunBatch' s x y z (a, b, c) = objFunBatch s x y a b c z
  let result = vjp (objFunBatch' shape inputs first_h true_values) (wh, u, bh) 1.0

  in result
