{-# LANGUAGE DataKinds, OverloadedLists, TypeFamilies, TypeOperators, ScopedTypeVariables #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module LogisticRegression where

import Data.Tensor
import Data.Singletons
import GHC.TypeNats
import Data.Foldable
import Util

test_data :: (Floating a) => (Tensor '[5, 2] a, Vector 5 Bool)
test_data = ([13.0, 15.0,
              23.0, 12.0,
              15.0, 0.0,
              17.0, -14.0,
              12.0, -5.0],
              [True, True, False, False, False])

start_ws :: (Floating a) => Vector 3 a
start_ws = [-5.0, -5.0, 5.0]

add_const_ones :: forall a n b. (Floating a, Eq a, KnownNat n, KnownNat b) => Tensor '[b, n] a -> Tensor '[b, 1 + n] a
add_const_ones xs = concatenate i1 ones xs
    where ones :: Tensor '[b, 1] a
          ones = reshape (expand ([1.0] :: Vector 1 a) :: Vector b a)

logistic_regression :: forall a n b. (Floating a, Eq a, KnownNat n, KnownNat b) => Vector (1 + n) a -> Tensor '[b, n] a -> Vector b a
logistic_regression ws xs = 
    sigmoid (ws `dot` transpose (add_const_ones xs))


cross_entropy :: forall a n. (Floating a, Eq a, KnownNat n) =>
                 Vector n a -> Vector n a -> a
cross_entropy ysp ysc = foldl (+) 0 ((expand (reshape (1 / batchsize) :: Vector 1 a) :: Vector n a)
                                        * ((- ysc * log ysp) + (- (1 - ysc) * log ysp)))
    where batchsize :: Scalar a
          batchsize =  [fromInteger . toInteger . head $ shape ysp]

-- From https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#gradient-descent
grad_ws :: forall a n b. (Floating a, Eq a, KnownNat n, KnownNat b) =>
           Vector (1 + n) a -> Tensor '[b, n] a -> Vector b a -> Vector (1 + n) a
grad_ws ws xs ysc = (expand (reshape (1 / batchsize) :: Vector 1 a) :: Vector (1 + n) a) * 
                        (transpose (add_const_ones xs) `dot` ((logistic_regression ws xs) - ysc))
    where batchsize :: Scalar a
          batchsize =  [fromInteger . toInteger . head $ shape ysc]