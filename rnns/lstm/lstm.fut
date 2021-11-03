type real= f32
let zero = 0f32
let sum  = f32.sum
let log  = f32.log
let tanh = f32.tanh
let exp  = f32.exp
let fromi64 = f32.i64

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

let mkGate [hx4][bs]
       (h: i64) (beg: i64)
       (mm_ih: [hx4][bs]real)
       (mm_hh: [hx4][bs]real)
       (bias:  [hx4]real) : [h][bs]real =
  let all = zip3 mm_ih mm_hh bias
  in  iota h |>
      map (\ i ->
            let (row_ih, row_hh, b) = all[beg+i]
            in  map2 (+) row_ih row_hh |> map (+b) 
          ) 

let step [bs] [hx4] [h] [d]
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (bias:    [hx4]real)
         (inp_els: [bs][d]real)
         (hidn_st: [h][bs]real, cell_st: [h][bs]real)
         : ([h][bs]real, [h][bs]real) =
  let mm_ih = map (matvec inp_els) wght_ih |> opaque
    -- map (matvec wght_ih) inp_els |> opaque

  let mm_hh = matmul wght_hh hidn_st |> opaque
    -- map (matvec (transpose hidn_st)) wght_hh |> opaque
    -- map (matvec wght_hh) hidn_st |> opaque

  let ingate0     = mkGate h 0     mm_ih mm_hh bias
  let forgetgate0 = mkGate h h     mm_ih mm_hh bias
  let cellgate0   = mkGate h (2*h) mm_ih mm_hh bias
  let outgate0    = mkGate h (3*h) mm_ih mm_hh bias

  let ingate     = map (map sigmoid) ingate0
  let forgetgate = map (map sigmoid) forgetgate0
  let cellgate   = map (map tanh   ) cellgate0
  let outgate    = map (map sigmoid) forgetgate0


--  let gates = map2 (map2 (+)) mm_ih mm_hh
--           |> map2 (\b row -> map (+b) row) bias
--              
--  let gates'     = assert (4*h == hx4)
--                          (unflatten 4 h gates)
--  let ingate     = map (map sigmoid) (gates'[0])
--  let forgetgate = map (map sigmoid) (gates'[1])
--  let cellgate   = map (map tanh   ) (gates'[2])
--  let outgate    = map (map sigmoid) (gates'[3])

  let cell_st' =  map2 (map2 (*)) ingate cellgate
               |> map2 (map2 (+))
                       (map2 (map2 (*)) forgetgate cell_st)
  let hidn_st' = map (map tanh) cell_st' |> map2 (map2 (*)) outgate

  in  (hidn_st', cell_st')

let lstmPrd [bs][n][d][h][hx4]
            (input: [n][bs][d]real)
            (wght_ih: [hx4][d]real)
            (wght_hh: [hx4][h]real)
            (wght_y:    [h][d]real)
            (bias:       [hx4]real)
            (bias_y:       [d]real)
            (hidn_st0: [h][bs]real)
            (cell_st0: [h][bs]real)
          : ([][d]real, [][h]real, [h][bs]real) = --([n][bs][d]real, [][h]real, [h][bs]real) =
  -- rnn component
  let hidn_stack0  = replicate bs zero
                  |> replicate h
                  |> replicate n

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
                  |> flatten
                  |> map (map (+(opaque 0)))

  let y_hat  = matmul hidn_stack' wght_y
            |> opaque
            |> map (map2 (+) bias_y)

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
            , wght_y:    [h][d]real
            , bias:       [hx4]real
            , bias_y:       [d]real
            )
          : real =
  let (input_hat, _, _) =
        lstmPrd input wght_ih wght_hh wght_y bias bias_y hidn_st0 cell_st0
  let bsd = bs * d
  let tot_loss =
    tabulate (n*bs*d)
        (\ind -> let i = ind / (bsd)
                 let r = ind - i *(bsd)
                 let j = r / d
                 let k = r - j*d
                 let ii= ind / d
                 let y_el     = input[i,j,k]
                 let y_hat_el = input_hat[ii,k] + bias_y[k]
                 in  (y_el - y_hat_el) * (y_el - y_hat_el)
        )
    |> sum
  let loss = tot_loss / (fromi64 (n*bs*d))
  in  loss
--  let y_y_hat  = map2 (map2 zip) input_hat input
--              |> flatten
--              |> flatten
--  let loss = meanSqr y_y_hat
--  in  loss


-- ==
-- compiled random input { [300][1024][80]f32 [256][1024]f32 [256][1024]f32 [1024][80]f32 [1024][256]f32 [256][80]f32 [1024]f32 [80]f32 f32 } auto output

-- Trying with: bs = 1024, d = 80, h = 256, n = 300
let main [bs][n][d][h][hx4]
         (input: [n][bs][d]real)
         (hidn_st0: [h][bs]real)
         (cell_st0: [h][bs]real)
         --- to-diff params
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (wght_y:    [h][d]real)
         (bias_h:    [hx4]real)
         (bias_y:       [d]real)
         --- adjoints ---
         (loss_adj : real) :
         ( [hx4][d]real
         , [hx4][h]real
         , [h][d]real
         , [hx4]real
         , [d]real
         ) =
  vjp (lstmObj input hidn_st0 cell_st0)
      (wght_ih, wght_hh, wght_y, bias_h, bias_y) loss_adj
