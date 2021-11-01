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

let meanSqr [n][d]
            (ys_hat : [n][d]real) 
            (ys :     [n][d]real) : real =
  let s  = map2 zip ys_hat ys
        |> flatten
        |> map (\(a, b) -> (a - b) * (a - b))
        |> sum
  in  s / (fromi64 (n*d))

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
              
  let gates'     = assert (4*h == hx4)
                          (unflatten 4 h gates)
  let ingate     = map sigmoid (gates'[0])
  let forgetgate = map sigmoid (gates'[1])
  let cellgate   = map tanh    (gates'[2])
  let outgate    = map sigmoid (gates'[3])

  let cell_st' =  map2 (*) ingate cellgate
               |> map2 (+) (map2 (*) forgetgate cell_st)
  let hidn_st' = map tanh cell_st' |> map2 (*) outgate

  in  (hidn_st', cell_st')

let lstmPrd [n][d][h][hx4]
            (input:     [n][d]real)
            (wght_ih: [hx4][d]real)
            (wght_hh: [hx4][h]real)
            (wght_y :    [h][d]real)
            (bias:        [hx4]real)
            (bias_y :       [d]real)
            (hidn_st0:      [h]real)
            (cell_st0:      [h]real)
          : ([n][d]real, [n][h]real, [h]real) =
  -- rnn component
  let hidn_stack0 = replicate n (replicate h zero)
  let (hidn_stack, (_, cell_st)) =
    loop (hidn_stack, (hidn_st, cell_st)) = (hidn_stack0, (hidn_st0, cell_st0))
    for i < n do
        let (hidn_st', cell_st') = step wght_ih wght_hh bias input[i] (hidn_st, cell_st)
        let hidn_stack[i] = hidn_st'
        in  (hidn_stack, (hidn_st', cell_st'))
  -- fully connected output
  let y_hat = matmul hidn_stack wght_y |> map (map2 (+) bias_y)
  in  (y_hat, hidn_stack, cell_st)
  -- hidden_states[:, h-1] instead of hidn_stack in the return?

----------------------------------------------------------------
-- `bs`  is the batch size (for the moment `bs = 1`)          --
-- `n`   is the length of a time series;                      --
-- `d`   is the dimensionality of a point of a time series;   --
-- `h`   is the length of the hidden layer                    --
-- `hx4` is `4 x h`                                           --
-- `layers`: not used, i.e., we assume num layers is 1        --
----------------------------------------------------------------
let lstmObj [n][d][h][hx4]
            (input:     [n][d]real)
            (wght_ih: [hx4][d]real)
            (wght_hh: [hx4][h]real)
            (wght_y:    [h][d]real)
            (bias:       [hx4]real)
            (bias_y:       [d]real)
            (hidn_st0:     [h]real)
            (cell_st0:     [h]real)
          : real =
  let (input_hat, _, _) = lstmPrd input wght_ih wght_hh wght_y bias bias_y hidn_st0 cell_st0
  let loss = meanSqr input_hat input
  in  loss