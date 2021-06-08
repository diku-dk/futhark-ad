module Main where

import Data.List
import Data.String

split :: [Int] -> [a] -> [[a]]
split [] as = [as]
split (i : is) as = take i as : split is (drop i as)

vectorize :: [String] -> String
vectorize xs = "[" ++ intercalate ", " xs ++ "]"

matrixize :: [[String]] -> String
matrixize xss = "[" ++ intercalate ",\n" (map vectorize xss) ++ "]"

i64 :: String -> String
i64 = (++ "i64")

process :: String -> String
process s =
  intercalate "\n" $
    [ vectorize $ concat alphas,
      matrixize means,
      matrixize icf,
      matrixize xs',
      w_g ++ " " ++ i64 w_m
    ]
  where
    (xs', w_g, w_m)
      | length xs_and_wishart == 2 && n_int > 1 =
        let (x : [[w_g, w_m]]) = xs_and_wishart
         in (replicate n_int x, w_g, w_m)
      | otherwise =
        let [xs, [[w_g, w_m]]] = split [n_int] xs_and_wishart
         in (xs, w_g, w_m)
    [alphas, means, icf, xs_and_wishart] = split [k_int, k_int, k_int] rest
    (k_int, n_int) = (read k :: Int, read n :: Int)
    ((_d : k : [n]) : rest) = map words $ lines s

main :: IO ()
main = getContents >>= putStr . process
