-- Developed with Ahmad Al-Sibahi

let logsumexp = map f32.exp >-> f32.sum >-> f32.log

let log_normalize x = map (\y -> y-logsumexp x) x

let dotprod xs ys = map2 (*) xs ys |> f32.sum

let normpdf x mu sigma = f32.exp(-x**2/2)/f32.sqrt(2*f32.pi)

let logpdf x mu sigma = f32.log (normpdf x mu sigma)

let gmm_log_likelihood (log_proportion: []f32) (mean: []f32) (stddev: []f32) (data: []f32) =
  let cluster_lls =
    map3 (\log_proportion1 mean1 stddev1 ->
            map (\x -> log_proportion1 + logpdf x mean1 stddev1) data)
         log_proportion mean stddev
  in map logsumexp (transpose cluster_lls) |> f32.sum

let diff_w (w_i: f32) (mu_i: f32) (sigma_i: f32) (x_k: f32) =
  (f32.exp(((x_k-mu_i)**2)/2*(sigma_i**2) + w_i + f32.log(2*f32.pi)/2))/sigma_i

let diff_mu (w_i: f32) (mu_i: f32) (sigma_i: f32) (x_k: f32) =
  (-(x_k-mu_i)*f32.exp((x_k-mu_i)**2/(2*sigma_i**2)+w_i-f32.log(2*f32.pi)/2))/(sigma_i**3)

let diff_sigma (w_i: f32) (mu_i: f32) (sigma_i: f32) (x_k: f32) =
  ((x_k-mu_i)**2-f32.exp((x_k-mu_i)**2/(2*sigma_i**2)+w_i-f32.log(2*f32.pi)/2))/(sigma_i**4)
  -
  f32.exp((x_k-mu_i)**2/(2*sigma_i**2)+w_i-f32.log(2*f32.pi)/2)

import "lib/github.com/athas/vector/vector"

module vec3 = cat_vector (cat_vector vector_1 vector_1) vector_1
type vec3 't = vec3.vector t

let diff (log_proportion: vec3 f32) (mean: vec3 f32) (stddev: vec3 f32) (data: []f32): vec3 (vec3 f32) =
  map (\x_k ->
         let gs = vec3.map (\((w_i, mu_i), sigma_i) ->
                              f32.exp(1/(w_i + logpdf x_k mu_i sigma_i)))
                           (vec3.zip (vec3.zip log_proportion mean) stddev)
                  |> vec3.reduce (+) 0
         in vec3.map (\((w_i, mu_i), sigma_i) ->
                        vec3.from_array [1/gs * diff_w w_i mu_i sigma_i x_k,
                                         1/gs * diff_mu w_i mu_i sigma_i x_k,
                                         1/gs * diff_sigma w_i mu_i sigma_i x_k])
                     (vec3.zip (vec3.zip log_proportion mean) stddev))
      data
  |> reduce (\xss yss ->
               vec3.map (\(xs, ys) ->
                           vec3.map (uncurry (+)) (vec3.zip xs ys))
                        (vec3.zip xss yss))
            (vec3.from_array (replicate 3 (vec3.from_array [0,0,0])))


-- ==
-- random input { 100 [1000000]f32 }

let main (n: i32) (data: []f32) =
  let eps = 0.001
  let step (log_proportion, mean, stddev) =
    let d = diff log_proportion mean stddev data
    in (vec3.map (uncurry (+)) (vec3.zip log_proportion (vec3.map (*eps) (vec3.get 0 d))),
        vec3.map (uncurry (+)) (vec3.zip mean (vec3.map (*eps) (vec3.get 1 d))),
        vec3.map (uncurry (+)) (vec3.zip stddev (vec3.map (*eps) (vec3.get 2 d))))
  in iterate n step (vec3.from_array (map f32.log [0.1,0.5,0.4]),
                     vec3.from_array [4,5,6],
                     vec3.from_array [7,8,9])
