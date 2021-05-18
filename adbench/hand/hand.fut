import "lib/github.com/diku-dk/linalg/linalg"

module linalg_f64 = mk_linalg f64

let matmul = linalg_f64.matmul
let matadd = map2 (map2 (f64.+))

let identity n = tabulate_2d n n (\i j -> f64.bool(i == j))

let angle_axis_to_rotation_matrix (angle_axis: [3]f64) : [3][3]f64 =
  let n = f64.sqrt (angle_axis[0]**2+angle_axis[1]**2+angle_axis[2]**2)
  in if n < 0.0001 then identity 3 else
     let x = angle_axis[0] / n
     let y = angle_axis[1] / n
     let z = angle_axis[2] / n
     let s = f64.sin n
     let c = f64.cos n
     in [[x * x + (1 - x * x) * c,
          x * y * (1 - c) - z * s,
          x * z * (1 - c) + y * s],
         [x * y * (1 - c) + z * s,
          y * y + (1 - y * y) * c,
          y * z * (1 - c) - x * s],
         [x * z * (1 - c) - y * s,
          z * y * (1 - c) + x * s,
          z * z + (1 - z * z) * c]]

let apply_global_transform [m] (pose_params: [][3]f64) (positions: [3][m]f64) =
  let R = angle_axis_to_rotation_matrix pose_params[0]
          |> map (map2 (*) pose_params[1])
  in (R `matmul` positions) `matadd`
     transpose (replicate m [pose_params[2,0], pose_params[2,1], pose_params[2,2]])

let relatives_to_absolutes [n] (relatives: [][4][4]f64) (parents: [n]i64) : [n][4][4]f64 =
  -- Initial value does not matter (I think).
  loop absolutes = replicate n (identity 4)
  for (relative, parent, i) in zip3 relatives parents (iota n) do
    if parent == -1
    then absolutes with [i] = relative
    else absolutes with [i] = copy (absolutes[parent] `matmul` relative)

let euler_angles_to_rotation_matrix (xzy: [3]f64) : [4][4]f64 =
  let tx = xzy[0]
  let ty = xzy[2]
  let tz = xzy[1]
  let costx = f64.cos(tx)
  let sintx = f64.sin(tx)
  let costy = f64.cos(ty)
  let sinty = f64.sin(ty)
  let costz = f64.cos(tz)
  let sintz = f64.sin(tz)
  in [[costy * costz,
       -costx * sintz + sintx * sinty * costz,
       sintx * sintz + costx * sinty * costz,
       0],
      [costy * sintz,
       costx * costz + sintx * sinty * sintz,
       -sintx * costz + costx * sinty * sintz,
       0],
      [-sinty,
       sintx * costy,
       costx * costy,
       0],
      [0,
       0,
       0,
       1]]

type~ hand_model [num_bones][n][m] =
  { parents: []i64,
    base_relatives: [][][]f64,
    inverse_base_absolutes: [][][]f64,
    base_positions: [n][m]f64,
    weights: [num_bones][]f64,
    triangles: [][]i64,
    is_mirrored: bool
  }

let get_posed_relatives [num_bones] (model: hand_model [num_bones][][]) (pose_params: [][3]f64) =
  let offset = 3
  let f i =
    matmul model.base_relatives[i]
           (euler_angles_to_rotation_matrix pose_params[i+offset])
  in tabulate num_bones f

let get_skinned_vertex_positions [num_bones][n][m]
                                 (model: hand_model [num_bones][n][m])
                                 (pose_params: [][3]f64)
                                 (apply_global: bool) =
  let relatives = get_posed_relatives model pose_params
  let absolutes = relatives_to_absolutes relatives model.parents
  let transforms = map2 matmul absolutes model.inverse_base_absolutes
  let base_positions = model.base_positions
  let positions =
    loop pos = tabulate_2d 3 m (\_ _ -> 0)
    for (transform, weights) in zip transforms model.weights
    do map2 (map2 (+))
            pos
            (transform[0:3] `matmul` base_positions |> map (map2 (*) weights))

  let positions = if model.is_mirrored
                  then positions with [0] = map f64.neg positions[0]
                  else positions

  in if apply_global
     then apply_global_transform pose_params positions
     else positions

let to_pose_params (theta: []f64) (num_bones: i64) : [][]f64 =
  let n = 3 + num_bones
  let num_fingers = 5
  let cols = 5 + num_fingers * 4
  in tabulate n (\i -> match i
                       case 0 -> take 3 theta[0:]
                       case 1 -> [1,1,1]
                       case 2 -> take 3 theta[3:]
                       case j ->
                         if j >= cols || j == 3 || j % 4 == 0 then [0,0,0]
                         else if j % 4 == 1 then [theta[j + 1], theta[j + 2], 0]
                         else [theta[j + 2], 0, 0])

entry calculate_objective [num_bones] (model: hand_model [num_bones][][]) (correspondences: []i64) (points: [][]f64) (theta: []f64) : []f64 =
    let pose_params = to_pose_params theta num_bones
    let vertex_positions = get_skinned_vertex_positions model pose_params true
    in map2 (\point correspondence ->
               map2 (-) point vertex_positions[:, correspondence])
            points correspondences |> flatten
