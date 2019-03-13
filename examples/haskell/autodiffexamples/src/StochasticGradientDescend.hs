{-# LANGUAGE GADTs, ExistentialQuantification, TypeFamilies, ConstraintKinds,
             DataKinds, TypeOperators, LambdaCase, TypeFamilyDependencies, ScopedTypeVariables,
             MultiParamTypeClasses, FlexibleInstances, TypeApplications, BangPatterns #-}
module StochasticGradientDescend where

import Debug.Trace

sgd :: forall n a. Floating n => Rational -> n -> a -> (n -> a -> n)
                   -> (n -> a -> Double) -> Int -> ([Double], n)
sgd lr params xs gradf lossf iters = go iters [] params
    where go :: Int -> [Double] -> n -> ([Double], n)
          go it !losses !params | it < 0 = (reverse losses, params)
                                | otherwise = go (it - 1) (lossf params xs : losses) (update_params params)
          update_params :: n -> n
          update_params params =
            params - fromRational lr * gradf params xs