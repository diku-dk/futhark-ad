type real= f64
let zero = 0f64
let sum  = f64.sum
let log  = f64.log
let tanh = f64.tanh
let exp  = f64.exp 

let dotproduct [n] (a: [n]real) (b: [n]real) : real =
    sum <| map2 (*) a b

let matvec [m][n] (mat: [m][n]real) (vec: [n]real) =
    map (dotproduct vec) mat

let sigmoid (x: real) : real =
    1.0 / (1.0 + exp(-x))

let lstmModel [hx4][h][d]
              (input_el: [d]real)
              (wght_ih: [hx4][d]real)
              (wght_hh: [hx4][h]real)
              (bias:    [hx4]real)
              (hidn_st: [h]real, cell_st: [h]real) -- size h or d ???
                : ([h]f64, [h]f64) =
  let gates = map2 (+) (matvec wght_hh hidn_st) <|
              map2 (+) bias <| matvec wght_ih input_el
  let gates'= unflatten 4 h gates
  let ingate     = map sigmoid (gates'[0])
  let forgetgate = map sigmoid (gates'[1])
  let cellgate   = map tanh    (gates'[2])
  let outgate    = map sigmoid (gates'[3])

  let cell_st' = map2 (+) (map2 (*) forgetgate cell_st)
                   <| map2 (*) ingate cellgate
  let hidn_st' = map2 (*) outgate <| map tanh cell_st'
  in  (hidn_st', cell_st')

--let lstmModel [d] (weight: [4][d]f64) (bias: [4][d]f64)
--                  (hidden: [d]f64) (cell: [d]f64) (input: [d]f64)
--                : ([d]f64, [d]f64) =

let lstmPredict [h] [hx4] [d]
                (input_el:[d]real)
                (wght_ih: [hx4][d]real)
                (wght_hh: [hx4][h]real)
                (bias:    [hx4]real)
                --(mainParams: [slen][2][4][d]real)
                (extraParams: [3][d]real)
                (hidn_st: [h][d]real, cell_st: [h][d]real)
                --(state: [slen][2][d]real)
               : ([d]f64, ([h][d]real, [h][d]real)) = -- innermost size: h or d?
    let x0 = map2 (*) input_el extraParams[0]
    let hidn_st_0 = replicate h <| replicate h zero
    let cell_st_0 = replicate h <| replicate h zero
    let state_0 = (hidn_st_0, cell_st_0)

    -- CONFLICT ON THE SIZE OF x: should it be [d] or [h]???
    let (state'', x') =
        loop ((hidn_st', cell_st'), x) = (state_0, x0)
        for i < h do
            let (h, c) = lstmModel x wght_ih wght_hh bias (hidn_st[i], cell_st[i])
            let hidn_st'[i] = h
            let cell_st'[i] = c
            in  ((hidn_st', cell_st'), h)

    let x'' = map2 (*) x' extraParams[1] |>
              map2 (+) extraParams[2]

    in  (x'', state'')

-- `n` is the length of a time series;
-- `d` is the dimensionality of a point of a time series;
-- `h` is the length of the hidden layer
-- `layers` is the number of layers 
let lstmObj [n][d][h]
            (input: [n][d]real)
            (wght_ih: [n][4][h][d]real)
            (wght_hh: [h][4][h][d]real)
            (bias: [4][h]real)
            (extraParams: [3][d]real)
            (hidn_st0: [h][d]real)
            (cell_st0: [h][d]real) =
  let (_, total) =
    loop(state, total) = ((hidn_st0, cell_st0), zero)
    for i < n - 1 do
        let (y_pred, new_state) = lstmPredict input[i] wght_ih wght_hh bias extraParams state
        let tmp_sum = sum <| map exp y_pred
        let tmp_log = - (log (tmp_sum + 2.0))
        let ynorm   = map (+tmp_log) y_pred
        let new_total = total + (dotproduct input[i+1] ynorm)
        in  (new_state, new_total)
  let count = d * (n - 1)
  in  - (total / f64.i64(count))

