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

let step [hx4] [h] [d]
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (bias:    [hx4]real)
         (inp_el:  [d]real)
         (hidn_st: [h]real, cell_st: [h]real)
         : ([h]f64, [h]f64) =
  let gates =  matvec wght_ih inp_el
            |> map2 (+) bias 
            |> map2 (+) (matvec wght_hh hidn_st)
              
  let gates'     = assert (4*h == 4xh)
                          (unflatten 4 h gates)
  let ingate     = map sigmoid (gates'[0])
  let forgetgate = map sigmoid (gates'[1])
  let cellgate   = map tanh    (gates'[2])
  let outgate    = map sigmoid (gates'[3])

  let cell_st' =  map2 (*) ingate cellgate
               |> map2 (+) (map2 (*) forgetgate cell_st)
  let hidn_st' = map tanh cell_st' |> map2 (*) outgate

  in  (hidn_st', cell_st')

-- `n` is the length of a time series;
-- `d` is the dimensionality of a point of a time series;
-- `h` is the length of the hidden layer
-- `layers` is the number of layers (simplified, not used)
let lstmObj [n][d][h]
            (input: [n][d]real)
            (wght_ih: [4][h][d]real)
            (wght_hh: [4][h][d]real)
            (wght_y :    [h][d]real)
            (bias_ih: [4][h]real)
            (bias_hh: [4][h]real)
            (bias_y :    [d]real)
            (extraParams: [3][d]real)
            (hidn_st0: [h]real)
            (cell_st0: [h]real) =
  -- rnn component
  let hidn_stack0 = replicate n (replicate h zero)
  let (hidn_stack, (hidn_st, cell_st) =
    loop (hidn_stack, (hidn_st, cell_st)) = (hidn_stack0, (hidn_st0, cell_st0))
    for i < n do
        let (hidn_st', cell_st') = step wght_ih wght_hh bias input[i] (hidn_st, cell_st)
        let hidn_stack[i] = hidn_st'
        in  (hidn_stack, (hidn_st', cell_st'))
  -- fully connected output
  let y_hat = matmul hidn_stack wght_y |> map (map2 (+) bias_y)
  in  (y_hat, hidn_stack, cell_st)
  -- hidden_states[:, h-1] instead of hidn_stack in the return?

----------------------------------------------------
--- We simplify a bit and support only one layer ---
--- if multiple layers are to be supported, the  ---
--- first wght_ih0 : [hx4][d]real, and the others---
--- wght_ihs : [layers-1][hx4][h]real.           ---
--- The wght_hhs : [layers][hx4][d]real.         ---
--- Hence the first invocation of `lstmModel`    ---
---   should be treated separately outside the   ---
---   loop because the last loop-param `x` ion   ---
---   changes dimension from [d] to [h] after the---
---   first iteration.
----------------------------------------------------

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
            let (h_st, c_st) = lstmModel x wght_ih wght_hh bias (hidn_st[i], cell_st[i])
            let hidn_st'[i] = h_st
            let cell_st'[i] = c_st
            in  ((hidn_st', cell_st'), h_st)

    let x'' = map2 (*) x' extraParams[1] |>
              map2 (+) extraParams[2]

    in  (x'', state'')
