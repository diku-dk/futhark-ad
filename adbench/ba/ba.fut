-- ==
-- entry: calculate_objective
-- input @ data/ba1_n49_m7776_p31843.in output @ data/ba1_n49_m7776_p31843.F
-- input @ data/ba2_n21_m11315_p36455.in
-- input @ data/ba3_n161_m48126_p182072.in
-- input @ data/ba4_n372_m47423_p204472.in
-- input @ data/ba5_n257_m65132_p225911.in
-- input @ data/ba6_n539_m65220_p277273.in
-- input @ data/ba7_n93_m61203_p287451.in
-- input @ data/ba8_n88_m64298_p383937.in
-- input @ data/ba9_n810_m88814_p393775.in
-- input @ data/ba10_n1197_m126327_p563734.in
-- input @ data/ba11_n1723_m156502_p678718.in
-- input @ data/ba12_n253_m163691_p899155.in
-- input @ data/ba13_n245_m198739_p1091386.in
-- input @ data/ba14_n356_m226730_p1255268.in
-- input @ data/ba15_n1102_m780462_p4052340.in
-- input @ data/ba16_n1544_m942409_p4750193.in
-- input @ data/ba17_n1778_m993923_p5001946.in
-- input @ data/ba18_n1936_m649673_p5213733.in
-- input @ data/ba19_n4585_m1324582_p9125125.in
-- input @ data/ba20_n13682_m4456117_p2987644.in

-- ==
-- entry: calculate_jacobian
-- input @ data/ba1_n49_m7776_p31843.in output @ data/ba1_n49_m7776_p31843.J
-- input @ data/ba2_n21_m11315_p36455.in
-- input @ data/ba3_n161_m48126_p182072.in
-- input @ data/ba4_n372_m47423_p204472.in
-- input @ data/ba5_n257_m65132_p225911.in
-- input @ data/ba6_n539_m65220_p277273.in
-- input @ data/ba7_n93_m61203_p287451.in
-- input @ data/ba8_n88_m64298_p383937.in
-- input @ data/ba9_n810_m88814_p393775.in
-- input @ data/ba10_n1197_m126327_p563734.in
-- input @ data/ba11_n1723_m156502_p678718.in
-- input @ data/ba12_n253_m163691_p899155.in
-- input @ data/ba13_n245_m198739_p1091386.in
-- input @ data/ba14_n356_m226730_p1255268.in
-- input @ data/ba15_n1102_m780462_p4052340.in
-- input @ data/ba16_n1544_m942409_p4750193.in
-- input @ data/ba17_n1778_m993923_p5001946.in
-- input @ data/ba18_n1936_m649673_p5213733.in
-- input @ data/ba19_n4585_m1324582_p9125125.in
-- input @ data/ba20_n13682_m4456117_p2987644.in

import "lib/github.com/athas/vector/vspace"

let N_CAM_PARAMS: i64 = 11
let ROT_IDX: i64 = 0
let CENTER_IDX: i64 = 3
let FOCAL_IDX: i64 = 6
let X0_IDX: i64 = 7
let RAD_IDX: i64 = 9

module v3d = mk_vspace_3d f64
type point_3d = v3d.vector
let point_3d a: point_3d = {x=a[0], y=a[1], z=a[2]}

module v2d = mk_vspace_2d f64
type point_2d = v2d.vector
let point_2d a: point_2d = {x=a[0], y=a[1]}

let rodrigues_rotate_point (rot: point_3d) (X: point_3d) =
    let sqtheta = v3d.quadrance rot in
    if sqtheta != 0 then
        let theta = f64.sqrt sqtheta
        let costheta = f64.cos theta
        let sintheta = f64.sin theta
        let theta_inv = 1 / theta

        let w = v3d.scale theta_inv rot
        let w_cross_X = v3d.cross w X
        let tmp = v3d.dot w X * (1 - costheta)

        in v3d.(scale costheta X +
                scale sintheta w_cross_X +
                scale tmp w)
    else v3d.(X + cross rot X)

let radial_distort (rad_params: point_2d) (proj: point_2d) =
  let rsq = v2d.quadrance proj
  let L = 1 + rad_params.x * rsq + rad_params.y * rsq * rsq
  in v2d.scale L proj

let project cam X =
  let Xcam = rodrigues_rotate_point
             (point_3d cam[ROT_IDX:ROT_IDX+3])
             (X v3d.- (point_3d cam[CENTER_IDX:CENTER_IDX+3]))
  let distorted = radial_distort (point_2d cam[RAD_IDX:RAD_IDX+2])
                                 (v2d.scale (1/Xcam.z) {x=Xcam.x, y=Xcam.y})
  in point_2d cam[X0_IDX:X0_IDX+2] v2d.+
     v2d.scale cam[FOCAL_IDX] distorted

let compute_reproj_err cam X w feat : point_2d =
  v2d.scale w (project cam X v2d.- feat)

let compute_zach_weight_error w : f64 =
  1 - w*w

let ba_objective [n][m][p] (cams: [n][11]f64) (X: [m]point_3d) (w: [p]f64) (obs:[p][2]i32) (feat:[]point_2d) =
  let p = length w
  let reproj_err =
    tabulate p (\i -> compute_reproj_err cams[obs[i,0]]
                                         X[obs[i,1]]
                                         w[i]
                                         feat[i])
  let w_err = map compute_zach_weight_error w
  in (reproj_err, w_err)

let grad f x = vjp f x 1f64

let ba_diff [n][m][p] (cams: [n][11]f64)
                      (X: [m]point_3d)
                      (w: [p]f64)
                      (obs: [p][2]i32)
                      (feats: [p]point_2d) =
  let compute_reproj_err_J_block cam X w feat =
    let wrt_x (cam, X, w) = (compute_reproj_err cam X w feat).x
    let wrt_y (cam, X, w) = (compute_reproj_err cam X w feat).y
    let (dcam1, dX1, dw1) = grad wrt_x (cam, X, w)
    let (dcam2, dX2, dw2) = grad wrt_y (cam, X, w)
    in ((dcam1,dcam2), (dX1,dX2), (dw1, dw2))
  in map4 compute_reproj_err_J_block
          (map (\i -> cams[i]) obs[:,0])
          (map (\i -> X[i]) obs[:,1])
          w feats

entry calculate_objective [n][m][p] (cams: [n][11]f64) (X: [m][3]f64) (w: [p]f64) (obs:[p][2]i32) (feat:[][2]f64) =
  let X = map (\p -> {x=p[0],y=p[1],z=p[2]}) X
  let feat = map (\p -> {x=p[0],y=p[1]}) feat
  let (a,b) = ba_objective cams X w obs feat
  in (map (\{x,y} -> [x,y]) a, b)

-- The packing code is derived from
-- https://github.com/tomsmeding/ADBench/blob/157260330293a46068593357cebc0f71f203750b/tools/Accelerate/src/BA.hs#L85-L138
-- by Tom Smeding.
entry calculate_jacobian [n][m][p] (cams: [n][11]f64) (X: [m][3]f64) (w: [p]f64) (obs:[p][2]i32) (feat:[][2]f64) =
  let X = map (\p -> {x=p[0],y=p[1],z=p[2]}) X
  let feat = map (\p -> {x=p[0],y=p[1]}) feat

  -- Compute derivatives.
  let grads = #[noinline] ba_diff cams X w obs feat

  -- Convert to matrix format.
  let dcams =
    tabulate_2d (2*p) 11
                (\i j -> let ((dcam1, dcam2), _, _) = grads[i/2]
                         in if (i % 2 == 0) then dcam1[j] else dcam2[j])

  let dX =
    tabulate (2*p)
             (\i -> let (_, (dX1, dX2), _) = grads[i/2]
                    in if (i % 2 == 0)
                       then [dX1.x,dX1.y,dX1.z]
                       else [dX2.x,dX2.y,dX2.z])
  let drdw =
    tabulate (2*p)
             (\i -> let (_, _, (dw1, dw2)) = grads[i/2]
                    in if (i % 2 == 0) then dw1 else dw2)

  let dwdw = map (grad compute_zach_weight_error) w

  -- Pack sparse Jacobian.
  let rows = map (*15) (iota (2*p)) ++ map (+(2*p*15)) (iota (p+1))
  let colsvals =
    tabulate (2*p*15)
             (\i -> let obsi = i / (2*15)
                    let valouti = i / 15
                    let (ci, pi) = (i64.i32 obs[obsi,0], i64.i32 obs[obsi,1])
                    let j = i % 15
                    in if j < 11
                       then (11*ci+j, dcams[valouti,j])
                       else if j < 14
                       then (11*n+3*pi+j-11, dX[valouti,j-11])
                       else (11*n+3*m+obsi, drdw[valouti]))
             ++ tabulate p (\i -> (11*n+3*m+i, dwdw[i]))

  in (rows, map (.0) colsvals, map (.1) colsvals)
