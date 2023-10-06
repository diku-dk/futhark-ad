type real = f32
let sum = f32.sum

let matmultr [d][n] (a: [d][n]real) (b: [d][d]real) : [n][d]real =
    map (\a_col ->
            map (\b_col ->
                    map2 (*) a_col b_col  |> sum
                ) (transpose b)
        ) (transpose a)

let objfun [k][d][n] (ass: [k][d][n]real, bss: [k][d][d]real) : [k][n][d]real =
    map2 matmultr ass bss

-- ==
-- random input { [200][128][5120]f32 [200][128][128]f32 [200][5120][128]f32 }

-- random input { [200][128][5120]f64 [200][128][128]f64 [200][5120][128]f64 }

entry main [d][k][n]
           (ass: [k][d][n]real)
           (bss: [k][d][d]real)
           (radj:[k][n][d]real) =
  objfun (ass, bss)
  --vjp objfun (ass, bss) radj

