{-# LANGUAGE ScopedTypeVariables #-}
module Util where

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (- x))

-- From https://mrob.com/pub/ries/lanczos-gamma.html
log_gamma :: forall a. (Ord a, Floating a) => a -> a
log_gamma z | z < 0.5 = log (pi / sin (pi * z)) - log_gamma (1.0 - z)
            | otherwise = log (sqrt (2 * pi)) + log sum - base + log base * (z' + 0.5)
    where lct :: [a]
          lct = [-0.5395239384953e-5, 0.1208650973866179e-2, -1.231739572450155,
                 24.01409824083091, -86.50532032941677, 76.18009172947146, 1.000000000190015]
          g :: a
          g = 5.0
          z' :: a
          z' = z - 1
          gosum :: [a] -> a -> a -> a
          gosum [p] i s = s + p
          gosum (p:ps) i s = gosum ps (i - 1) (s + p / (z' + i))
          base :: a
          base = z' + g + 0.5
          sum :: a
          sum = gosum lct ((fromInteger . toInteger . length $ lct) - 1) 0.0

log_beta :: forall a. (Ord a, Floating a) => a -> a -> a
log_beta z w = log_gamma z + log_gamma w - log_gamma (z + w)

-- Not really factorial if not integers, but "good enough"
fact :: forall a. (Ord a, Floating a) => a -> a
fact n | n <= 0.1 = 1.0
       | otherwise = n * fact (n - 1)

log_fact :: forall a. (Ord a, Floating a) =>  a -> a
log_fact n | n <= 0.1 = 0.0
           | n < 10.0 = log (fact n)
           | otherwise = n * log n - n + 0.5 * log n + 0.5 * log (2 * pi)

log_binomial :: forall a. (Ord a, Floating a) => a -> a -> a
log_binomial n k = log_fact n - (log_fact k + log_fact (n - k))