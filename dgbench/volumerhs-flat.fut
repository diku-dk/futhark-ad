-- type real "parametrization"
type real = f32
let zero = 0f32
let sum  = f32.sum
let grav = 9.81f32 -- is BigFloat double?
let gdm1 = 0.4f32  -- is BigFloat double?

-- helpers
let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap3 as bs cs f = map3 f as bs cs
let imap4 as bs cs ds f = map4 f as bs cs ds

let unzip16 as =
  let (xs, ys, zs, ts) =
    unzip4 <|
    map (\ ( a1, a2, a3, a4, a5, a6, a7, a8
           , b1, b2, b3, b4, b5, b6, b7, b8
           ) ->
           ( (a1, a2, a3, a4), (a5, a6, a7, a8)
           , (b1, b2, b3, b4), (b5, b6, b7, b8)
           )
      ) as
  let ( (a1, a2, a3, a4), (a5, a6, a7, a8)
      , (b1, b2, b3, b4), (b5, b6, b7, b8)
      ) =
      ( unzip4 xs, unzip4 ys, unzip4 zs, unzip4 ts)
  in  ( a1, a2, a3, a4, a5, a6, a7, a8
      , b1, b2, b3, b4, b5, b6, b7, b8
      )


-- assumed compile-time constants: N, nmoist, ntrace
-- these are their default values,
-- unsure if this is what gets run
let N = 4i32
let Nq= 8i64
let nmoist = 0i64
let ntrace = 0i64

-- these are known constants
let nstate = 5i64
let nvar   = nstate + nmoist + ntrace
let nvgeo  = 14i64

-- vgeo dims
let d_eps_x = 0i64
let d_nta_x = 1i64
let d_tau_x = 2i64
let d_eps_y = 3i64
let d_nta_y = 4i64
let d_tau_y = 5i64
let d_eps_z = 6i64
let d_nta_z = 7i64
let d_tau_z = 8i64
let d_MJ    = 9i64
let d_MJI   = 10i64
let d_x     = 11i64
let d_y     = 12i64
let d_z     = 13i64

-- Q dims
let d_p = 0i64
let d_U = 1i64
let d_V = 2i64
let d_W = 3i64
let d_E = 4i64

let get_ji (ind_b: i64) : (i64,i64) =
  let ind = i16.i64 ind_b
  let Nq' = i16.i64 Nq
  let j = ind / Nq'
  let i = ind - j*Nq'
  in  (i64.i16 j, i64.i16 i)

-- nelem is the number of Cuda blocks
-- I assume gravity is a constant (?)
let volumerhs [Nq][nelem]
        ( gravity: real )
        ( rhs:  [nelem][nvar] [Nq][Nq][Nq]real )
        ( Q:    [nelem][nvar] [Nq][Nq][Nq]real 
        , vgeo: [nelem][nvgeo][Nq][Nq][Nq]real 
        , D:    [Nq][Nq]real
        ) : [nelem][5][Nq][Nq][Nq]real =
  -- s_D : [Nq][Nq]real, s_F : [Nq][Nq][nstate]real, s_G : [Nq][Nq][nstate]real
  -- r_rhsp, r_rhsU, r_rhsV, r_rhsW, r_rhsE : [Nq]real
  -- e = blockIdx().x, j = threadIdx().y, i = threadIdx().x

  let rhs' =
    iota nelem |>
    #[incremental_flattening(only_intra)]
    map (\ e -> -- Cuda block level
           let Nq2 = Nq*Nq
           let (r_rhsps, r_rhsUs, r_rhsVs, r_rhsWs, r_rhsEs) =
            imap (iota Nq2)
              (\_ind ->
                ( -- this should be sequentialized
                  #[sequential] replicate Nq (f32.i64 e * opaque zero)
                , #[sequential] replicate Nq (f32.i64 e * opaque zero)
                , #[sequential] replicate Nq (f32.i64 e * opaque zero)
                , #[sequential] replicate Nq (f32.i64 e * opaque zero)
                , #[sequential] replicate Nq (f32.i64 e * opaque zero)
                )
              ) |> unzip5

           let e' = i32.i64 e
           let one = f32.i32 (#[unsafe] (2*e' + 2)/(e' + 1) - 1) 
           -- this also seems transposed; are they talking column major???
           let s_D = imap (iota Nq2)
                          (\ind -> let (j,i) = get_ji ind in one * #[unsafe] D[j,i])

           -- julia for k in 0:n includes n
           let (r_rhsps, r_rhsUs, r_rhsVs, r_rhsWs, r_rhsEs) =
            loop (r_rhsps, r_rhsUs, r_rhsVs, r_rhsWs, r_rhsEs)
             for k < Nq do
              let ( s_F_ps, s_F_Us, s_F_Vs, s_F_Ws, s_F_Es
                  , s_G_ps, s_G_Us, s_G_Vs, s_G_Ws, s_G_Es
                  , r_Hps,  r_HUs,  r_HVs,  r_HWs,  r_HEs
                  , MJ_p_gravs
                  ) =
               unzip16 <|
               imap (iota Nq2)
                (\flat_tid ->
                  -- j = threadIdx.y; i = threadIdx.x
                  let (j,i) = get_ji flat_tid
                  let MJ = #[unsafe] vgeo[e, d_MJ, k, j, i]
                  let (eps_x, eps_y, eps_z) = #[unsafe]
                    ( vgeo[e, d_eps_x, k, j, i]
                    , vgeo[e, d_eps_y, k, j, i]
                    , vgeo[e, d_eps_z, k, j, i]
                    )
                  let (nta_x, nta_y, nta_z) = #[unsafe]
                    ( vgeo[e, d_nta_x, k, j, i]
                    , vgeo[e, d_nta_y, k, j, i]
                    , vgeo[e, d_nta_z, k, j, i]
                    )
                  let (tau_x, tau_y, tau_z) = #[unsafe]
                    ( vgeo[e, d_tau_x, k, j, i]
                    , vgeo[e, d_tau_y, k, j, i]
                    , vgeo[e, d_tau_z, k, j, i]
                    )
                  let z = #[unsafe] vgeo[e, d_z, k, j, i]
                  let (U, V, W) = #[unsafe]
                    ( Q[e, d_U, k, j, i]
                    , Q[e, d_V, k, j, i]
                    , Q[e, d_W, k, j, i]
                    )
                  let (p, E) = #[unsafe]
                    ( Q[e, d_p, k, j, i]
                    , Q[e, d_E, k, j, i]
                    )
                  let mj_p_grav = MJ * p * gravity
                  let pinv  = 1/p -- rcp(p)
                  let p2inv = 0.5 * pinv
                  let P = gdm1 * (E - (U*U + V*V + W*W) * p2inv - p*gravity*z)
                  
                  let fluxp_x = U
                  let fluxU_x = pinv * U * U + P
                  let fluxV_x = pinv * U * V
                  let fluxW_x = pinv * U * W
                  let fluxE_x = pinv * U * (E + P)

                  let fluxp_y = V
                  let fluxU_y = pinv * V * U
                  let fluxV_y = pinv * V * V + P
                  let fluxW_y = pinv * V * W
                  let fluxE_y = pinv * V * (E + P)

                  let fluxp_z = W
                  let fluxU_z = pinv * W * U
                  let fluxV_z = pinv * W * V
                  let fluxW_z = pinv * W * W + P
                  let fluxE_z = pinv * W * (E + P)

                  -- all s_F and s_G are in form: s_F[p, j, i] = ...
                  let s_F_p = MJ * (eps_x * fluxp_x + eps_y * fluxp_y + eps_z * fluxp_z)
                  let s_F_U = MJ * (eps_x * fluxU_x + eps_y * fluxU_y + eps_z * fluxU_z)
                  let s_F_V = MJ * (eps_x * fluxV_x + eps_y * fluxV_y + eps_z * fluxV_z)
                  let s_F_W = MJ * (eps_x * fluxW_x + eps_y * fluxW_y + eps_z * fluxW_z)
                  let s_F_E = MJ * (eps_x * fluxE_x + eps_y * fluxE_y + eps_z * fluxE_z)

                  let s_G_p = MJ * (nta_x * fluxp_x + nta_y * fluxp_y + nta_z * fluxp_z)
                  let s_G_U = MJ * (nta_x * fluxU_x + nta_y * fluxU_y + nta_z * fluxU_z)
                  let s_G_V = MJ * (nta_x * fluxV_x + nta_y * fluxV_y + nta_z * fluxV_z)
                  let s_G_W = MJ * (nta_x * fluxW_x + nta_y * fluxW_y + nta_z * fluxW_z)
                  let s_G_E = MJ * (nta_x * fluxE_x + nta_y * fluxE_y + nta_z * fluxE_z)

                  let r_Hp = MJ * (tau_x * fluxp_x + tau_y * fluxp_y + tau_z * fluxp_z)
                  let r_HU = MJ * (tau_x * fluxU_x + tau_y * fluxU_y + tau_z * fluxU_z)
                  let r_HV = MJ * (tau_x * fluxV_x + tau_y * fluxV_y + tau_z * fluxV_z)
                  let r_HW = MJ * (tau_x * fluxW_x + tau_y * fluxW_y + tau_z * fluxW_z)
                  let r_HE = MJ * (tau_x * fluxE_x + tau_y * fluxE_y + tau_z * fluxE_z)

                  in  ( s_F_p, s_F_U, s_F_V, s_F_W, s_F_E
                      , s_G_p, s_G_U, s_G_V, s_G_W, s_G_E
                      , r_Hp,  r_HU,  r_HV,  r_HW,  r_HE
                      , mj_p_grav
                      )
                ) -- end tabulate over Cuda block threads

              -- implicit SYNC-THREADS, i.e., end writing to shared memory

              -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              -- 1. probably we should enable nvcc to do better  --
              --    register allocation by grouping together all --
              --    the computation for r_rhsps and then the one --
              --    for r_rhsUs and so on.                       --
              -- 2. we should also fused all the remaining into  --
              --    one tabulate_2d computation.                 --
              -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                             

              -- ALL FUSED VERSION
              let (r_rhsps', r_rhsUs', r_rhsVs', r_rhsWs', r_rhsEs') =
                unzip5 <|
                imap (iota Nq2)
                  (\flat_tid ->
                    -- j = threadIdx.y; i = threadIdx.x
                    let (j,i) = get_ji flat_tid
                    --let (s_Di, s_Dj) = (s_D[i, :], s_D[j, :])

                    -- helper definition
                    let helperInner s_F_s s_G_s r_H r_rhs_s ctv =
                      -- all parallelism should be sequentialized in here!
                      let r_rhs_k =
                             #[sequential]
                             map4 (\ Dni Dnj Fj Gi -> Dni * Fj + Dnj * Gi)
                                  (#[unsafe] s_D[i*Nq : i*Nq + Nq] :> [Nq]real)
                                  (#[unsafe] s_D[j*Nq : j*Nq + Nq] :> [Nq]real)
                                  (#[unsafe] s_F_s[j*Nq : j*Nq+Nq] :> [Nq]real)
                                  (#[unsafe] s_G_s[i : Nq*Nq : Nq] :> [Nq]real)
                                  -- s_Di s_Dj s_F_s[j, :] s_G_s[:, i]
                          |> #[sequential] sum
                      let r_rhs_k' = r_rhs_k + ctv
                      in  #[sequential]
                          map2  (\ ind r_rhs_el ->
                                    r_rhs_el + (r_H * #[unsafe] s_D[ind*Nq + k]) +
                                      if ind == k then r_rhs_k' else zero
                                ) (iota Nq) r_rhs_s

                    let MJ_p_grav = MJ_p_gravs[flat_tid]
                    let r_rhsW' = helperInner s_F_Ws s_G_Ws r_HWs[flat_tid] r_rhsWs[flat_tid] (- MJ_p_grav)
                    let r_rhsp' = helperInner s_F_ps s_G_ps r_Hps[flat_tid] r_rhsps[flat_tid] zero
                    let r_rhsU' = helperInner s_F_Us s_G_Us r_HUs[flat_tid] r_rhsUs[flat_tid] zero
                    let r_rhsV' = helperInner s_F_Vs s_G_Vs r_HVs[flat_tid] r_rhsVs[flat_tid] zero
                    let r_rhsE' = helperInner s_F_Es s_G_Es r_HEs[flat_tid] r_rhsEs[flat_tid] zero
                    in  (r_rhsp', r_rhsU', r_rhsV', r_rhsW', r_rhsE')
                  )

              in  (r_rhsps', r_rhsUs', r_rhsVs', r_rhsWs', r_rhsEs')
           -- end loop k < Nq

           -- result of the big kernel
           let mgiss = imap (iota Nq2) (\ flat_tid -> let (j,i) = get_ji flat_tid in vgeo[e, d_MJI, :, j, i] )
           let h_lift (r_rhs_r: [Nq2][Nq]real) (d_i: i64) : [Nq][Nq][Nq]real =
                  imap (iota Nq2)
                              (\ flat_tid -> -- inner parallelism sequentialize
                                  let (j,i) = get_ji flat_tid
                                  let h rg mji rh = rh + mji * rg
                                  in  #[sequential]
                                      map3 h (#[unsafe] r_rhs_r[flat_tid])
                                             (#[unsafe] mgiss[flat_tid])
                                             (#[unsafe] rhs[e, d_i, :, j, i])
                              )
              |>  #[unsafe] unflatten Nq Nq
           in  [ h_lift r_rhsps d_p
               , h_lift r_rhsUs d_U
               , h_lift r_rhsVs d_V
               , h_lift r_rhsWs d_W
               , h_lift r_rhsEs d_E
               ]
    )
  in rhs'

-- ==
-- entry: objfun
-- random input { [20000][5][8][8][8]f32 [20000][5][8][8][8]f32 [20000][14][8][8][8]f32 [8][8]f32}

entry objfun [nelem]
        ( rhs:  [nelem][nvar] [Nq][Nq][Nq]real )
        ( Q:    [nelem][nvar] [Nq][Nq][Nq]real ) 
        ( vgeo: [nelem][nvgeo][Nq][Nq][Nq]real )
        ( D:    [Nq][Nq]real ) = 
  volumerhs grav rhs (Q, vgeo, D)

entry revdiff [Nq][nelem]
        ( rhs:  [nelem][nvar] [Nq][Nq][Nq]real )
        ( Q:    [nelem][nvar] [Nq][Nq][Nq]real ) 
        ( vgeo: [nelem][nvgeo][Nq][Nq][Nq]real )
        ( D:    [Nq][Nq]real )
        ( rhs_: [nelem][5][Nq][Nq][Nq]real ) =
  vjp (volumerhs grav rhs) (Q, vgeo, D) rhs_
