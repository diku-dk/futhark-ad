

let matmultr [d][n] (a: [d][n]f64) (b: [d][d]f64) : [n][d]f64 =
    map (\a_col ->
            map (\b_col ->
                    map2 (*) a_col b_col  |> f64.sum
                ) (transpose b)
        ) (transpose a)

let objfun [k][d][n] (ass: [k][d][n]f64, bss: [k][d][d]f64) : [k][n][d]f64 =
    map2 matmultr ass bss

-- ==
-- random input { [200][128][10240]f64 [200][128][128]f64 [200][10240][128]f64 }

entry main [d][k][n]
           (ass: [k][d][n]f64)
           (bss: [k][d][d]f64)
           (radj:[k][n][d]f64) =
  vjp objfun (ass, bss) radj

