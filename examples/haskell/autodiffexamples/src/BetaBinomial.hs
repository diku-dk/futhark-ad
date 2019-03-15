{-# LANGUAGE DataKinds, OverloadedLists, TypeFamilies, TypeOperators, ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module BetaBinomial where

import Data.Tensor
import Data.Singletons
import GHC.TypeNats
import Data.Foldable
import Util

betabinomial_logpmf :: forall n. (Ord n, Floating n) => Integer -> Rational -> Rational -> n -> n
betabinomial_logpmf n a b x = log_binomial n' x + log_beta (x + a') (n' - x + b') - log_beta a' b'
    where a' :: n
          a' = fromRational a
          b' :: n
          b' = fromRational b
          n' :: n
          n' = fromInteger n