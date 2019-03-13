module Main where

import LogisticRegression

main :: IO ()
main = do
    print test_data
    print (sigmoid . fst $ test_data)
    print (logistic_regression start_ws . fst $ test_data)
