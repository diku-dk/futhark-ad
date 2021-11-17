-- ==
-- input @ data/kdd_cup.in.gz
-- output @ data/kdd_cup.out

let euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32): f32 =
  f32.sum (map (\x->x*x) (map2 (-) pt1 pt2))

let cost [n][k][d] (points: [n][d]f32) (centres: [k][d]f32) =
  points
  |> map (\p -> map (euclid_dist_2 p) centres)
  |> map f32.minimum
  |> f32.sum

let tolerance = 1 : f32

let main [n][d]
        (_threshold: i32) (k: i64) (max_iterations: i32)
        (points: [n][d]f32) =
  -- Assign arbitrary initial cluster centres.
  let cluster_centres = take k (reverse points)
  let i = 0
  let stop = false
  let (cluster_centres,i,_stop) =
    loop (cluster_centres : [k][d]f32, i, stop)
    while i < max_iterations && !stop do
    let (cost', cost'') =
      jvp2 (\x -> vjp (cost points) x 1) cluster_centres
           (replicate k (replicate d 1))
    let x = map2 (map2 (/)) cost' cost''
    let new_centres = map2 (map2 (-)) cluster_centres x
    let stop =
      (map2 euclid_dist_2 new_centres cluster_centres |> f32.sum)
      < tolerance
    in (new_centres, i+1, stop)
  in cluster_centres
