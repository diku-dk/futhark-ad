-- Manual version of sparse-kmeans for k=10
-- ==
-- compiled input @ data/movielens.in.gz
-- compiled input @ data/nytimes.in.gz
-- compiled input @ data/scrna.in.gz

-- compiled nobench input @ data/scrna.in.gz
-- output @ data/movielens.out

import "kmeans-sp-ad"


let main [nnz][np1] 
         (values: [nnz]f32)
         (indices_data: [nnz]i64) 
         (pointers: [np1]i64) =
  let fix_iter = false
  let threshold = 0.005f32
  let num_iterations = 10i32 --250i32
  let k = 10i64

  let (cluster_centers, num_its) =
      kmeansSpAD k threshold num_iterations fix_iter
                 values
                 indices_data
                 pointers
  in  (cluster_centers[0,:33], num_its)

