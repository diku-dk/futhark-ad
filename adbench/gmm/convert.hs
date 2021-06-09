{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import Data.ByteString.Char8 (readInt)
import Data.Word (Word8)

vectorize :: Word8 -> ByteString -> ByteString
vectorize sep xs =
  let xs' = if BS.last xs == sep then BS.init xs else xs
   in "[" <> BS.map f xs' <> "]"
  where
    f :: Word8 -> Word8
    f w
      | w == sep = 44
      | otherwise = w

alphas :: Int -> IO ()
alphas k = do
  putStr "["
  replicateM_ k doLine
  putStr "]"
  where
    doLine = BS.putStr =<< ((flip BS.snoc) 44 . BS.init) <$> BS.getLine

matrix_line :: ByteString -> ByteString
matrix_line = BS.cons 91 . BS.map f
  where
    f w
      | w == 10 = 93
      | w == 32 = 44
      | otherwise = w

matrixize :: Int -> IO ()
matrixize k = do
  putStr "["
  replicateM_ k (BS.putStr =<< matrix_line <$> BS.getLine)
  putStr "]"

i64 :: ByteString -> ByteString
i64 = (<> "i64")

main :: IO ()
main = do
  [_d, k, n] <- map (maybe (error "oops") fst . readInt) . BS.split 32 <$> BS.getLine
  alphas k
  matrixize k -- means
  matrixize k -- icf
  next1 <- BS.getLine
  next2 <- BS.getLine
  next3 <- BS.getLine
  putStr "["
  if next3 == mempty
    then do
      replicateM_ n $ BS.putStr $ matrix_line next1
      putStr "]"
      let [w_g, w_m] = BS.split 32 next3
      BS.putStr $ w_g <> " " <> i64 w_m
    else do
      BS.putStr $ matrix_line next1
      BS.putStr $ matrix_line next2
      BS.putStr $ matrix_line next3
      replicateM_ (n - 3) (BS.putStr =<< matrix_line <$> BS.getLine)
      putStr "]"
      (w_g : w_m : _) <- BS.split 32 <$> BS.getLine
      BS.putStr $ w_g <> " " <> i64 w_m
