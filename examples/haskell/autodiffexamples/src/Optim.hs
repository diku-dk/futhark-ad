{-# LANGUAGE GADTs, ExistentialQuantification, TypeFamilies, ConstraintKinds,
             DataKinds, TypeOperators, LambdaCase, TypeFamilyDependencies, ScopedTypeVariables,
             MultiParamTypeClasses, FlexibleInstances, TypeApplications, BangPatterns #-}
module Optim where

import System.Random
import Control.Monad
import Control.Monad.State.Strict
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

pso :: forall g n m a. (RandomGen g, Ord n, Floating n, Ord m, Floating m) => g -> Rational -> Rational -> Rational 
                      -> Int -> n -> n -> a
                      -> (n -> a -> m) -> Int -> ([m], n)
pso rgen alr llr glr nparticles paramslower paramsupper dat lossf iters = flip evalState rgen $ do
      ps <- initial_particles
      vs <- initial_velocities
      let gl = global ps
      go iters [] ps ps gl vs
    where go :: Int -> [m] -> [n] -> [n] -> n -> [n] -> State g ([m], n)
          go it !losses !xs !ps !gl !vs | it < 0 = return (reverse losses, gl)
                                        | otherwise = do
                                          vs' <- mapM (\case 
                                              (x, p, v) -> do
                                                rp <- randu 0 1
                                                rg <- randu 0 1
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
          initial_particles :: State g [n]
          initial_particles = do
            mapM (\_ -> randu paramslower paramsupper) [1..nparticles]
          initial_velocities :: State g [n]
          initial_velocities = do
            mapM (\_ -> randu (- (paramsupper - paramslower)) (paramsupper - paramslower)) [1..nparticles]
          global :: [n] -> n
          global = argmin (flip lossf dat)
          randu :: n -> n -> State g n
          randu l u = do
            modify (snd . next)
            g <- get
            let i = fst . next $ g
            let (gl, gu) = genRange g
            let normalized = (fromInteger . toInteger $ (gu - i)) / (fromInteger . toInteger $ (gu - gl))
            return $ normalized * (u - l) + l
            
