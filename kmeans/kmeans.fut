-- Playing around with defining kmeans clustering with AD.
-- ==
-- compiled nobench input @ data/trivial.in
-- output @ data/trivial.out
-- compiled nobench input @ data/100.in
-- output @ data/100.out
-- compiled input @ data/204800.in.gz
-- output @ data/204800.out
-- compiled input @ data/kdd_cup.in.gz
-- output @ data/kdd_cup.out

let euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32): f32 =
  f32.sum (map (\x->x*x) (map2 (-) pt1 pt2))

let cost [n][k][d] (points: [n][d]f32) (centres: [k][d]f32) =
  points
  |> map (\p -> map (euclid_dist_2 p) centres)
  |> map f32.minimum
  |> f32.sum

let grad f x = vjp f x 1f32

let tolerance = 1 : f32

let main [n][d]
        (_threshold: i32) (k: i32) (max_iterations: i32)
        (points: [n][d]f32) =
  let k = i64.i32 k
  -- Assign arbitrary initial cluster centres.
  let cluster_centres = take k points
  let i = 0
  let (cluster_centres,_i) =
    loop (cluster_centres : [k][d]f32, i)
    while i < max_iterations do
    let cost' = grad (cost points) cluster_centres
    let cost'' = jvp (grad (cost points)) cluster_centres
                     (replicate k (replicate d 1))
    let x = map2 (map2 (/)) cost' cost''
    let new_centres = map2 (map2 (-)) cluster_centres x
    in if (map2 euclid_dist_2 new_centres cluster_centres |> f32.sum) < tolerance
       then (new_centres, max_iterations)
       else (new_centres, i+1)
  in cluster_centres
