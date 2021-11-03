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

let outputLayer [n] [bs] [h] [d]
         (y:           [n][bs][d]real)
         -- to diff
         ( hidn_stack: [n][h][bs]real
         , wght_y:         [h][d]real
         , bias_y:            [d]real
         )
         : real =
  let hidn_stack'  = hidn_stack
                  |> map transpose
                  |> flatten
                  |> map (map (+(opaque 0)))

  let y_hat  = matmul hidn_stack' wght_y
            |> opaque
              --|> map (map2 (+) bias_y)

  let bsd = bs * d
  let tot_loss =
    tabulate (n*bs*d)
        (\ind -> let i = ind / (bsd)
                 let r = ind - i *(bsd)
                 let j = r / d
                 let k = r - j*d
                 let ii= ind / d
                 let y_el     = y[i,j,k]
                 let y_hat_el = y_hat[ii,k] + bias_y[k]
                 in  (y_el - y_hat_el) * (y_el - y_hat_el)
        )
    |> sum
  let loss = tot_loss / (fromi64 (n*bs*d))
  in  loss

-- ==
-- compiled random input { [300][1024][80]f32 [300][256][1024]f32 [256][80]f32 [80]f32 f32 } auto output

let main [n] [bs] [h] [d]
         (y:           [n][bs][d]real)
         -- to diff
         (hidn_stack: [n][h][bs]real)
         (wght_y:         [h][d]real)
         (bias_y:            [d]real)
         -- adjoints
         (loss_adj: real) =
  vjp (outputLayer y) (hidn_stack, wght_y, bias_y) loss_adj

