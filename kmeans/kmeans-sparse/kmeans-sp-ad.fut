import "utils"

let euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32): f32 =
  f32.sum (map (\x->x*x) (map2 (-) pt1 pt2))

let initCenters [nnz][np1]
                (k: i64) (columns: i64)
                (pointers: [np1]i64) (row_indices: [nnz]i64)
                (values: [nnz]f32) (indices_data: [nnz]i64) :
                [k][columns]f32 =

  -- initialize the cluster centers (in dense rep) with the first k (sparse) elements
  let first_k_total_nz = pointers[k+1] -- reduce (+) 0 (take k vector_nnz)
  let dense_k_inds = map2 (\row col -> row*columns + col)
                          (take first_k_total_nz row_indices)
                          (take first_k_total_nz indices_data)

  let cluster_centers_flat =
        scatter (replicate (k*columns) 0)
                dense_k_inds
                (take first_k_total_nz values)

  let cluster_centers =
        unflatten k columns cluster_centers_flat

  in  cluster_centers

let costSparse [nnz][np1][cols][k]
      (values:  [nnz]f32)
      (colidxs: [nnz]i64)
      (begrows: [np1]i64)
      (cluster_centers: [k][cols]f32) : f32 =
  let n = np1 - 1
  let cluster_squared_sum = map (\c -> f32.sum (map (\x -> x*x) c)) cluster_centers

  -- partial_distss : [n][k]f32
  let partial_distss =
    map (\row -> 
          map (\cl_ind ->
                let index_start = begrows[row]
                let nnz = begrows[row+1] - index_start
                in  loop (correction) = (0) for j < nnz do
                      let element_value = values[index_start+j]
                      let column = colidxs[index_start+j]
                      let cluster_value = cluster_centers[cl_ind, column]
                      let value = (element_value - 2 * cluster_value)*element_value
                      in  correction+value
              ) (iota k)
        ) (iota n)
    |> opaque

  let min_dists =
    map (\dists ->
            map2 (+) dists cluster_squared_sum |>
            reduce f32.min f32.inf
        ) partial_distss
    |> opaque

  let cost = reduce (+) 0.0f32 min_dists

  in  cost

let tolerance = 1 : f32

let kmeansSpAD [nnz][np1]
        (k: i64)
        (_threshold: f32)
        (num_iterations: i32)
        (_fix_iter: bool)
        (values: [nnz]f32)
        (indices_data: [nnz]i64) 
        (pointers: [np1]i64) =

  let n = np1 - 1
  let columns = 1 + reduce (i64.max) (-1) indices_data
    
  let shape = map (\i -> pointers[i+1] - pointers[i]) (iota n) -- this is shape
  let flags = mkFlagArray shape false (replicate n true)  :> [nnz]bool
  let row_indices =
         map2 (\f i -> if f && (i!=0) then 1i64 else 0i64) flags (indices flags)
      |> scan (+) 0i64

  -- Assign arbitrary initial cluster centres.
  let cluster_centers =
      initCenters k columns pointers row_indices values indices_data

  let i = 0
  let stop = false
  let (cluster_centers, i, _stop) =
    -- ALWAYS EXECUTE num_iterations
    loop (cluster_centers : [k][columns]f32, i, stop)
      while i < num_iterations && !stop do
        let (cost', cost'') =
          jvp2  (\x -> vjp (costSparse values indices_data pointers) x 1)
                cluster_centers
                (replicate k (replicate columns 1))
        -- let x = map2 (map2 (/)) cost' cost''
        let x = map2 (map2 (\ c' c'' -> 1.0 * c' / c'' )) cost' cost''
        let new_centers = map2 (map2 (-)) cluster_centers x
--        let stop =
--          (map2 euclid_dist_2 new_centers cluster_centers |> f32.sum)
--          < tolerance
        in (new_centers, i+1, stop)
  in (cluster_centers, i)
