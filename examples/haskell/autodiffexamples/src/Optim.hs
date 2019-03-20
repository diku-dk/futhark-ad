{-# LANGUAGE GADTs, ExistentialQuantification, TypeFamilies, ConstraintKinds,
             DataKinds, TypeOperators, LambdaCase, TypeFamilyDependencies, ScopedTypeVariables,
             MultiParamTypeClasses, FlexibleInstances, TypeApplications, BangPatterns #-}
module Optim where

import Control.Monad
import Control.Monad.Random.Strict
import Data.List.Extras.Argmax

sgd :: forall n m a. (Floating n, Floating m) => Rational -> n -> a -> (n -> a -> n)
                   -> (n -> a -> m) -> Int -> ([m], n)
sgd lr params dat gradf lossf iters = go iters [] params
    where go :: Int -> [m] -> n -> ([m], n)
          go it !losses !params | it < 0 = (reverse losses, params)
                                | otherwise = go (it - 1) (lossf params dat : losses) (update_params params)
          update_params :: n -> n
          update_params params =
            params - fromRational lr * gradf params dat

pso :: forall n n' m a. (MonadRandom m, Ord n, Random n, Floating n, Ord n', Floating n')
                      => Rational -> Rational -> Rational 
                      -> Int -> n -> n -> a
                      -> (n -> a -> n') -> Int -> m ([n'], n)
pso alr llr glr nparticles paramslower paramsupper dat lossf iters = do
      ps <- initial_particles
      vs <- initial_velocities
      let gl = global ps
      go iters [] ps ps gl vs
    where go :: Int -> [n'] -> [n] -> [n] -> n -> [n] -> m ([n'], n)
          go it !losses !xs !ps !gl !vs | it < 0 = return (reverse losses, gl)
                                        | otherwise = do
                                          vs' <- mapM (\case 
                                              (x, p, v) -> do
                                                rp <- getRandomR (0.0 :: n, 1.0 :: n)
                                                rg <- getRandomR (0.0 :: n, 1.0 :: n)
                                                return $ alr' * v + llr' * rp * (p - x) + glr' * (gl - x)
                                            ) (zip3 xs ps vs)
                                          let xs' = zipWith (+) xs vs'
                                          let ps' = zipWith (\x p -> if lossf x dat < lossf p dat then x else p) xs' ps
                                          let gl' = global ps'
                                          go (it - 1) (lossf gl' dat : losses) xs' ps' gl' vs'
          alr' :: n
          alr' = fromRational alr
          llr' :: n
          llr' = fromRational llr
          glr' :: n
          glr' = fromRational glr
          initial_particles :: m [n]
          initial_particles = do
            mapM (\_ -> getRandomR (paramslower, paramsupper)) [1..nparticles]
          initial_velocities :: m [n]
          initial_velocities = do
            mapM (\_ -> getRandomR (- (paramsupper - paramslower), paramsupper - paramslower)) [1..nparticles]
          global :: [n] -> n
          global = argmin (flip lossf dat)
            
