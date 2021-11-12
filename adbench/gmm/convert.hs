{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import Data.ByteString.Char8 (readInt)
import Data.Word (Word8)
import System.IO (isEOF)
import Data.Char (isSpace)
import Data.ByteString.Internal (isSpaceWord8)

c2w :: Char -> Word8
c2w = toEnum . fromEnum

(>:) :: ByteString -> Word8 -> ByteString
(>:) = BS.snoc

(<:) :: Word8 -> ByteString -> ByteString
(<:) = BS.cons

strip :: ByteString -> ByteString
strip = BS.dropWhile isSpaceWord8 . (BS.reverse . BS.dropWhile isSpaceWord8 . BS.reverse) 

alphas :: Int -> IO ()
alphas k = do
  putStr "["
  BS.putStr =<< BS.getLine
  replicateM_ (k - 1) doLine
  putStr "]"
  where
    doLine = do
      l <- BS.getLine
      BS.putStr $ c2w ',' <: l

matrix_row :: ByteString -> ByteString
matrix_row bs = c2w '[' <: BS.map f (strip bs) >: c2w ']'
  where
    f w
      | w == c2w ' ' = c2w ','
      | otherwise = w

matrixize :: Int -> IO ()
matrixize k = do
  putStr "["
  BS.putStr =<< matrix_row <$> BS.getLine
  replicateM_ (k - 1) $ do l <- BS.getLine; BS.putStr $ c2w ',' <: matrix_row l
  putStr "]"

main :: IO ()
main = do
  [_d, k, n] <- map (maybe (error "oops") fst . readInt) . BS.split 32 <$> BS.getLine
  alphas k
  putStr "\n"
  matrixize k
  putStr "\n"
  matrixize k
  putStr "\n"
  next1 <- BS.getLine
  next2 <- BS.getLine
  eof <- isEOF
  putStr "["
  if eof
    then do
      BS.putStr $ matrix_row next1
      replicateM_ (n -1) (BS.putStr $ c2w ',' <: matrix_row next1)
      putStr "]\n"
      let (w_g : w_m : _) = BS.split 32 next2
      BS.putStr $ w_g <> " " <> w_m <> "i64"
    else do
      BS.putStr $ matrix_row next1
      BS.putStr $ c2w ',' <: matrix_row next2
      replicateM_ (n - 2) $ do l <- BS.getLine; BS.putStr $ c2w ',' <: matrix_row l
      putStr "]\n"
      (w_g : w_m : _) <- BS.split 32 <$> BS.getLine
      BS.putStr $ w_g <> " " <> w_m <> "i64"
