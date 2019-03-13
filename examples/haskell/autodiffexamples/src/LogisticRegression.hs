{-# LANGUAGE DataKinds, OverloadedLists, TypeFamilies, TypeOperators, ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module LogisticRegression where

import Data.Tensor
import Data.Singletons
import GHC.TypeNats

test_data :: (Floating a) => (Tensor '[5, 2] a, Vector 5 Bool)
test_data = ([13.0, 15.0,
              23.0, 12.0,
              15.0, 0.0,
              17.0, -14.0,
              12.0, -5.0],
              [True, True, False, False, False])

start_ws :: (Floating a) => Vector 3 a
start_ws = [0.0, 0.0, 0.0]

sigmoid :: (Floating a, SingI s) => Tensor s a -> Tensor s a
sigmoid xs = fmap (\ v -> 1 / (1 + exp (- v))) xs

logistic_regression :: forall a n b. (Floating a, Eq a, KnownNat n, KnownNat b) => Vector (1 + n) a -> Tensor '[b, n] a -> Vector b a
logistic_regression ws xs = 
    sigmoid (ws `dot` transpose cat)
  where ones :: Tensor '[b, 1] a
        ones = reshape (expand ([1.0] :: Vector 1 a) :: Vector b a)
        cat :: Tensor '[b, 1 + n] a
        cat = concatenate i1 ones xs