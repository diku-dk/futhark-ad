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
         (inp_els: [bs][d]real)
         (hidn_st: [h][bs]real, cell_st: [h][bs]real)
         -- weights
         ( wght_ih: [hx4][d]real
         , wght_hh: [hx4][h]real
         , bias:    [hx4]real
         )
         : ([h][bs]real, [h][bs]real) =
  let mm_ih = map (matvec inp_els) wght_ih |> opaque

  let mm_hh = matmul wght_hh hidn_st |> opaque

  let ingate0     = mkGate h 0     mm_ih mm_hh bias
  let forgetgate0 = mkGate h h     mm_ih mm_hh bias
  let cellgate0   = mkGate h (2*h) mm_ih mm_hh bias
  let outgate0    = mkGate h (3*h) mm_ih mm_hh bias

  let ingate     = map (map sigmoid) ingate0
  let forgetgate = map (map sigmoid) forgetgate0
  let cellgate   = map (map tanh   ) cellgate0
  let outgate    = map (map sigmoid) forgetgate0

  let cell_st' =  map2 (map2 (*)) ingate cellgate
               |> map2 (map2 (+))
                       (map2 (map2 (*)) forgetgate cell_st)
  let hidn_st' = map (map tanh) cell_st' |> map2 (map2 (*)) outgate

  in  (hidn_st', cell_st')

-- ==
-- compiled random input { [1024][80]f32 [256][1024]f32 [256][1024]f32 [1024][80]f32 [1024][256]f32 [1024]f32 [256][1024]f32 [256][1024]f32 } auto output

let main [bs] [hx4] [h] [d]
         (inp_els: [bs][d]real)
         (hidn_st: [h][bs]real)
         (cell_st: [h][bs]real)
         -- weights
         (wght_ih: [hx4][d]real)
         (wght_hh: [hx4][h]real)
         (bias:    [hx4]real)
         -- adjoints
         (hidn_st_adj: [h][bs]real)
         (cell_st_adj: [h][bs]real) =
  vjp (step inp_els (hidn_st, cell_st))
      (wght_ih, wght_hh, bias)
      (hidn_st_adj, cell_st_adj)

