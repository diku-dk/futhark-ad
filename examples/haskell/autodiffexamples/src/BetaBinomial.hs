{-# LANGUAGE DataKinds, OverloadedLists, TypeFamilies, TypeOperators, ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module BetaBinomial where

import Data.Tensor
import Data.Singletons
import GHC.TypeNats
import Data.Foldable
import Util

betabinomial_logpmf :: (Ord a, Floating a) => a -> a -> a -> a -> a
betabinomial_logpmf n a b x = log_binomial n x + log_beta (x + a) (n - x + b) - log_beta a b