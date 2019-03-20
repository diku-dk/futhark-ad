{-# LANGUAGE ScopedTypeVariables #-}
module Util where

import Unsafe.Coerce
import Data.Tensor
import Data.Singletons
import Control.Monad
import Control.Monad.Random.Strict

-- TODO Use better tensor library
tzipWith :: forall s a b c. (a -> b -> c) -> Tensor s a -> Tensor s b -> Tensor s c
tzipWith f t u = unsafeCoerce ((\s i -> f (tt s i) (uu s i)) :: [Int] -> [Int] -> c)
      where tt :: [Int] -> [Int] -> a
            tt = unsafeCoerce t
            uu :: [Int] -> [Int] -> b
            uu = unsafeCoerce u

tzipWith3 :: forall s a b c d. (a -> b -> c -> d) -> Tensor s a -> Tensor s b -> Tensor s c -> Tensor s d
tzipWith3 f t u v = unsafeCoerce ((\s i -> f (tt s i) (uu s i) (vv s i)) :: [Int] -> [Int] -> d)
      where tt :: [Int] -> [Int] -> a
            tt = unsafeCoerce t
            uu :: [Int] -> [Int] -> b
            uu = unsafeCoerce u
            vv :: [Int] -> [Int] -> c
            vv = unsafeCoerce v

tmax :: forall s a. Ord a => [Tensor s a] -> Tensor s a
tmax ts = unsafeCoerce ((\s i -> maximum (fmap (\t -> t s i) tts)) :: [Int] -> [Int] -> a)
      where tts :: [[Int] -> [Int] -> a]
            tts = fmap unsafeCoerce ts

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (- x))

logsumexp :: forall s a. (Ord a, Floating a, SingI s) => [Tensor s a] -> Tensor s a
logsumexp xs = log (sum (fmap (\x -> exp (x - maxx)) xs))
      where maxx :: Tensor s a
            maxx = tmax xs
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

sample_standard_uniform :: (Random a, Floating a, MonadRandom m) => m a
sample_standard_uniform = getRandom

sample_standard_normal :: (Random a, Floating a, MonadRandom m) => m a
sample_standard_normal = do
    u1 <- getRandom
    u2 <- getRandom
    return $ (sqrt . ((-2) *) . log $ u1) * sin (2 * pi * u2)

-- In PyTorch the gradient is constructed manually: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Distributions.h#L188
sample_standard_gamma :: (Random a, Ord a, Floating a, MonadRandom m) => a -> m a
sample_standard_gamma concentration | concentration >= 1.0 = do
                                          z <- sample_standard_normal
                                          u <- sample_standard_uniform
                                          let v = (1 + c * z) ** 3.0
                                          if z > - 1 / c && log u < 0.5 * z ** 2 + d - d * v + d * log v then
                                                 return $ d * v
                                          else sample_standard_gamma concentration                            
                                    | concentration < 0.0 = error "concentration must be larger than 0.0"
                                    | otherwise = do
                                          v <- sample_standard_gamma (concentration + 1.0)
                                          u <- sample_standard_uniform
                                          return $ v * (u ** (1.0 / concentration))
       where d = concentration - (1.0 / 3.0)
             c = 1 / sqrt (9 * d)

sample_dirichlet :: (Random a, Ord a, Floating a, MonadRandom m) => [a] -> m [a]
sample_dirichlet concentrations | length concentrations == 0 || any (< 0) concentrations = error "concentrations must be non-empty and positive"
                                | otherwise = do
                                        gs <- mapM (\conc -> sample_standard_gamma conc) concentrations
                                        return $ map (\g -> g / sum gs) gs

sample_beta :: (Random a, Ord a, Floating a, MonadRandom m) => a -> a -> m a
sample_beta concentration1 concentration0 = head <$> sample_dirichlet [concentration1, concentration0]