-- Playing around with defining kmeans clustering with AD.
-- ==
-- compiled nobench input @ data/trivial.in
-- output @ data/trivial.out

let euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32): f32 =
  f32.sum (map (\x->x*x) (map2 (-) pt1 pt2))

let cost [k][n][d] (points: [n][d]f32) (centres: [k][d]f32) =
  map (\c -> map (euclid_dist_2 c) points |> f32.sum) centres |> f32.sum

let grad f x = vjp f x 1f32

let learning_rate = 0.01 : f32

let main [n][d]
        (threshold: i32) (k: i32) (max_iterations: i32)
        (points: [n][d]f32) =
  let k = i64.i32 k
  -- Assign arbitrary initial cluster centres.
  let cluster_centres = take k points
  let i = 0
  let (cluster_centres,_i) =
    loop (cluster_centres, i)
    while i < max_iterations do
    let new_centres = map2 (map2 (-))
                           cluster_centres
                           (map (map (*learning_rate)) (#[trace(grad)] grad (cost points) cluster_centres))
    let score = #[trace(score)] cost points new_centres
    in if score == 0
       then (new_centres, max_iterations)
       else (new_centres, i+1)
  in (cluster_centres, cost points cluster_centres)
