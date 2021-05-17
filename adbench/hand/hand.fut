import "lib/github.com/athas/vector/vspace"

module v3d = mk_vspace_3d f64
type point_3d = v3d.vector
let point_3d a: point_3d = {x=a[0], y=a[1], z=a[2]}

import "lib/github.com/athas/vector/vector"

module vec1 = vector_1
module vec2 = cat_vector vec1 vec1
module vec3 = cat_vector vec2 vec1
type vec3 'a = vec3.vector a

type mat3x3 = vec3 point_3d

let mat3x3_row i = vec3.get i

let mat3x3_col (i: i32) (A: mat3x3) =
  match i
  case 0 -> point_3d [(vec3.get 0 A).x, (vec3.get 1 A).x, (vec3.get 2 A).x]
  case 1 -> point_3d [(vec3.get 0 A).y, (vec3.get 1 A).y, (vec3.get 2 A).y]
  case _ -> point_3d [(vec3.get 0 A).z, (vec3.get 1 A).z, (vec3.get 2 A).z]

let mat3x3 (A: [3][3]f64) : *mat3x3 =
  vec3.from_array
  (take vec3.length [point_3d A[0], point_3d A[1], point_3d A[2]])

let mul3x3 (A: mat3x3) (B: mat3x3) : *mat3x3 =
  mat3x3 [[v3d.dot (mat3x3_row 0 A) (mat3x3_col 0 B),
           v3d.dot (mat3x3_row 0 A) (mat3x3_col 1 B),
           v3d.dot (mat3x3_row 0 A) (mat3x3_col 2 B)],
          [v3d.dot (mat3x3_row 1 A) (mat3x3_col 0 B),
           v3d.dot (mat3x3_row 1 A) (mat3x3_col 1 B),
           v3d.dot (mat3x3_row 1 A) (mat3x3_col 2 B)],
          [v3d.dot (mat3x3_row 2 A) (mat3x3_col 0 B),
           v3d.dot (mat3x3_row 2 A) (mat3x3_col 1 B),
           v3d.dot (mat3x3_row 2 A) (mat3x3_col 2 B)]]

let add3x3 (A: mat3x3) (B: mat3x3) : mat3x3 =
  vec3.map2 (v3d.+) A B

let identity3x3 : mat3x3 =
  mat3x3 [[1,0,0], [0,1,0], [0,0,1]]

let angle_axis_to_rotation_matrix (angle_axis: point_3d) : mat3x3 =
  let n = v3d.norm angle_axis
  in if n < 0.0001 then identity3x3 else
     let x = angle_axis.x / n
     let y = angle_axis.y / n
     let z = angle_axis.z / n
     let s = f64.sin n
     let c = f64.cos n
     in mat3x3 [[x * x + (1 - x * x) * c,
                 x * y * (1 - c) - z * s,
                 x * z * (1 - c) + y * s],
                [x * y * (1 - c) + z * s,
                 y * y + (1 - y * y) * c,
                 y * z * (1 - c) - x * s],
                [x * z * (1 - c) - y * s,
                 z * y * (1 - c) + x * s,
                 z * z + (1 - z * z) * c]]

let apply_global_transform (pose_params: []point_3d) (positions: mat3x3) =
  let R = angle_axis_to_rotation_matrix pose_params[0]
          |> vec3.map (v3d.* pose_params[1])
  in (R `mul3x3` positions) `add3x3`
     mat3x3 (transpose (replicate 3 [pose_params[2].x, pose_params[2].y, pose_params[2].z]))

let relatives_to_absolutes [n] (relatives: []mat3x3) (parents: [n]i64) : [n]mat3x3 =
  -- Initial value does not matter (I think).
  loop absolutes = replicate n identity3x3 for i < n do
    if parents[i] == -1
    then absolutes with [i] = relatives[i]
    else absolutes with [i] = (absolutes[parents[i]] `mul3x3` relatives[i])
