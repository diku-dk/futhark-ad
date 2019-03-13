{-# LANGUAGE DataKinds, ScopedTypeVariables #-}
module Main where

import LogisticRegression
import StochasticGradientDescend
import Data.Tensor

main :: IO ()
main = do
    let (xs, ysc) = test_data
    let ysc' :: Vector 5 Double = fmap (\x -> if x then 1.0 else 0.0) ysc
    let ysp = logistic_regression start_ws xs
    let (ls, final_ws) = sgd 0.3 (start_ws :: Vector 3 Double) xs (\ pm zs -> grad_ws pm zs ysc')
                                 (\ pm zs -> cross_entropy (logistic_regression pm zs) ysc') 5
    print $ ls
    print $ final_ws
    print $ fmap (> 0.5) (logistic_regression final_ws xs)
