module Convert where

import Data.List
import Data.String
import System.Environment

split :: [Int] -> [a] -> [[a]]
split [] as = [as]
split (i : is) as = take i as : split is (drop i as)

vectorize :: [String] -> String
vectorize xs = "[" ++ intercalate ", " xs ++ "]"

matrixize :: [[String]] -> String
matrixize xss = "[" ++ intercalate ",\n" (map vectorize xss) ++ "]"

process :: String -> String
process s =
  intercalate "\n" $
    [ unwords [d, k, n],
      alphas',
      means',
      icf',
      xs',
      unwords wishart
    ]
  where
    (means', icf', xs') = (matrixize means, matrixize icf, matrixize xs)
    alphas' = vectorize $ concat alphas
    [alphas, means, icf, xs, [wishart]] = split [k_int, k_int, k_int, n_int] rest
    (k_int, n_int) = (read k, read n)
    ((d : k : [n]) : rest) = map words $ lines s

main :: IO ()
main = do
  args <- getArgs
  case args of
    [file] -> do
      s <- readFile file
      let file_name = reverse $ snd $ break (== '.') $ reverse file
      writeFile (file_name ++ "in") $ process s
    _ -> putStrLn "Wrong usage."
