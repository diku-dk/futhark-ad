type real= f64
let zero = 0f64
let sum  = f64.sum
let log  = f64.log
let tanh = f64.tanh
let exp  = f64.exp
let fromi64 = f64.i64

let dotproduct [n] (a: [n]real) (b: [n]real) : real =
    map2 (*) a b |> sum

let matvec [m][n] (mat: [m][n]real) (vec: [n]real) =
    map (dotproduct vec) mat

let matmul [m][n][q] (ass: [m][q]real) (bss: [q][n]real) : [m][n]real =
    map (matvec (transpose bss)) ass

let sigmoid (x: real) : real =
    1.0 / (1.0 + exp(-x))

let meanSqr [n]
            (y_y_hat : [n](real, real)) =
  let s  = map (\(a, b) -> (a - b) * (a - b)) y_y_hat
        |> sum
  in  s / (fromi64 n)

let step [bs] [hx4] [h] [d]
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (bias:    [hx4]real)
         (inp_els: [bs][d]real)
         (hidn_st: [h][bs]real, cell_st: [h][bs]real)
         : ([h][bs]real, [h][bs]real) =
  let mm_ih = map (matvec inp_els) wght_ih |> opaque
    -- map (matvec wght_ih) inp_els |> opaque

  let mm_hh = matmul wght_hh hidn_st
    -- map (matvec (transpose hidn_st)) wght_hh |> opaque
    -- map (matvec wght_hh) hidn_st |> opaque

  let gates = map2 (map2 (+)) mm_ih mm_hh
           |> map2 (\b row -> map (+b) row) bias
              
  let gates'     = assert (4*h == hx4)
                          (unflatten 4 h gates)
  let ingate     = map (map sigmoid) (gates'[0])
  let forgetgate = map (map sigmoid) (gates'[1])
  let cellgate   = map (map tanh   ) (gates'[2])
  let outgate    = map (map sigmoid) (gates'[3])

  let cell_st' =  map2 (map2 (*)) ingate cellgate
               |> map2 (map2 (+))
                       (map2 (map2 (*)) forgetgate cell_st)
  let hidn_st' = map (map tanh) cell_st' |> map2 (map2 (*)) outgate

  in  (hidn_st', cell_st')

entry lstmPrd [bs][n][d][h][hx4]
            (input: [n][bs][d]real)
            (hidn_st0: [h][bs]real)
            (cell_st0: [h][bs]real)
            (wght_ih: [hx4][d]real)
            (wght_hh: [hx4][h]real)
            (bias_ih:    [hx4]real)
            (bias_hh:    [hx4]real)
            (wght_y:    [h][d]real)
            (bias_y:       [d]real)
          : ([n][bs][d]real, [n][bs][h]real, [h][bs]real) =
  -- rnn component
  let hidn_stack0  = replicate bs zero
                  |> replicate h
                  |> replicate n
  let bias = map2 (+) bias_ih bias_hh

  -- hidn_stack0 :: [n][bs][h]
  let (hidn_stack, (_, cell_st)) =
    loop (hidn_stack, (hidn_st, cell_st)) = (hidn_stack0, (hidn_st0, cell_st0))
    for i < n do
        let (hidn_st', cell_st') = step wght_ih wght_hh bias input[i] (hidn_st, cell_st)
        let hidn_stack[i] = hidn_st'
        in  (hidn_stack, (hidn_st', cell_st'))

  -- fully connected output
  let hidn_stack'  = hidn_stack
                  |> map transpose

  let y_hat  = matmul (flatten hidn_stack') wght_y
            |> map (map2 (+) bias_y)
            |> unflatten n bs

  in  (y_hat, hidn_stack', cell_st)
  -- hidden_states[:, h-1] instead of hidn_stack in the return?

----------------------------------------------------------------
-- `bs`  is the batch size (for the moment `bs = 1`)          --
-- `n`   is the length of a time series;                      --
-- `d`   is the dimensionality of a point of a time series;   --
-- `h`   is the length of the hidden layer                    --
-- `hx4` is `4 x h`                                           --
-- `layers`: not used, i.e., we assume num layers is 1        --
----------------------------------------------------------------
let lstmObj [bs][n][d][h][hx4]
            (input: [n][bs][d]real)
            (hidn_st0: [h][bs]real)
            (cell_st0: [h][bs]real)
            ( wght_ih: [hx4][d]real
            , wght_hh: [hx4][h]real
            , bias_ih:    [hx4]real
            , bias_hh:    [hx4]real
            , wght_y:    [h][d]real
            , bias_y:       [d]real
	    )
          : real =
  let (input_hat, _, _) =
        lstmPrd input hidn_st0 cell_st0 wght_ih wght_hh bias_ih bias_hh wght_y bias_y
  let y_y_hat  = map2 (map2 zip) input_hat input
              |> flatten
              |> flatten
  let loss = meanSqr y_y_hat
  in  loss

let main [bs][n][d][h][hx4]
         (input: [n][bs][d]real)
         (hidn_st0: [h][bs]real)
         (cell_st0: [h][bs]real)
         --- to-diff params
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (bias_ih:    [hx4]real)
         (bias_hh:    [hx4]real)
         (wght_y:    [h][d]real)
         (bias_y:       [d]real)
         --- adjoints ---
         (loss_adj : real) :
         ( [hx4][d]real
         , [hx4][h]real
         , [hx4]real
         , [hx4]real
         , [h][d]real
         , [d]real
         ) =
  vjp (lstmObj input hidn_st0 cell_st0)
      (wght_ih, wght_hh, bias_ih, bias_hh, wght_y, bias_y) loss_adj
