-- Bundle Adjustment from https://github.com/awf/autodiff

import "ad"
import "lib/github.com/athas/vector/vspace"

let N_CAM_PARAMS: i32 = 11
let ROT_IDX: i32 = 0
let CENTER_IDX: i32 = 3
let FOCAL_IDX: i32 = 6
let X0_IDX: i32 = 7
let RAD_IDX: i32 = 9

module fs (M: real) = {
  module v3d = mk_vspace_3d M
  type point_3d = v3d.vector
  let point_3d a: point_3d = {x=a[0], y=a[1], z=a[2]}

  module v2d = mk_vspace_2d M
  type point_2d = v2d.vector
  let point_2d a: point_2d = {x=a[0], y=a[1]}

  let rodrigues_rotate_point (rot: point_3d) (X: point_3d) =
      let sqtheta = v3d.quadrance rot in
      if M.(sqtheta != i32 0) then
          let theta = M.sqrt sqtheta
          let costheta = M.cos theta
          let sintheta = M.sin theta
          let theta_inv = M.(i32 1 / theta)

          let w = v3d.scale theta_inv rot
          let w_cross_X = v3d.cross w X
          let tmp = M.(v3d.dot w X * (i32 1 - costheta))

          in v3d.(scale costheta X +
                  scale sintheta w_cross_X +
                  scale tmp w)
      else v3d.(X + cross rot X)

  let radial_distort (rad_params: point_2d) (proj: point_2d) =
    let rsq = v2d.quadrance proj
    let L = M.(i32 1 + rad_params.x * rsq + rad_params.y * rsq * rsq)
    in v2d.scale L proj

  let project cam X =
    let Xcam = rodrigues_rotate_point
               (point_3d cam[ROT_IDX:ROT_IDX+3])
               (X v3d.- (point_3d cam[CENTER_IDX:CENTER_IDX+3]))
    let distorted = radial_distort (point_2d cam[RAD_IDX:RAD_IDX+2])
                                   (v2d.scale M.(i32 1/Xcam.z) {x=Xcam.x, y=Xcam.y})
    in point_2d cam[X0_IDX:X0_IDX+2] v2d.+
       v2d.scale cam[FOCAL_IDX] distorted

  let compute_reproj_err cam X w feat =
    v2d.scale w (project cam X v2d.- feat)

  let compute_zach_weight_error w =
    M.(i32 1 - w*w)

  let ba_objective cams X w (obs:[][]i32) (feat:[]point_2d) =
    let p = length w
    let reproj_err =
      tabulate p (\i -> compute_reproj_err cams[obs[i,0]]
                                           X[obs[i,1]]
                                           w[i]
                                           feat[i])
    let w_err = map compute_zach_weight_error w
    in (reproj_err, w_err)
}

module f32_dual = mk_dual f32

module fs_f32_dual = fs f32_dual

entry diff [n][m][p] (cams: [n][]f32)
                     (X: [m][3]f32)
                     (w: [p]f32)
                     (obs: [p][2]i32)
                     (feats: [p][2]f32) =
  let compute_reproj_err_J_block (cam: []f32) (X: [3]f32) (w: f32) feat =
    let compute_reproj_err_wrapper (parameters: []f32_dual.t) =
      let (cam, parameters) = split N_CAM_PARAMS parameters
      let X = {x=parameters[0], y=parameters[1], z=parameters[2]}
      let w = parameters[3]
      let feat = fs_f32_dual.point_2d (map f32_dual.inject feat)
      in fs_f32_dual.compute_reproj_err cam X w feat
    in tabulate (N_CAM_PARAMS+4)
                (\i -> let parameters =
                         map2 f32_dual.make_dual
                              (cam++[X[0], X[1], X[2], w])
                              (tabulate (N_CAM_PARAMS+4) (\j -> f32.bool(i == j)))
                         in compute_reproj_err_wrapper parameters)
  in map4 compute_reproj_err_J_block
          (map (\i -> unsafe cams[i]) obs[:,0])
          (map (\i -> unsafe X[i]) obs[:,1])
          w feats
