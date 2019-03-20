{-# LANGUAGE ScopedTypeVariables, OverloadedLists, DataKinds, FlexibleContexts #-}
module DirichletMixtureModel where

import Util
import Data.Tensor
import GHC.TypeNats
import Data.Singletons
import Data.Singletons.Prelude.Foldable

normal_logpdf :: forall a s. (Floating a, SingI s, KnownNat (Product s)) => a -> a -> Tensor s a -> Tensor s a
normal_logpdf mu sigma xs = - 0.5 * log (2 * pi) - log sigmat - ((xs - mut) ** 2) / (2 * sigmat ** 2)
    where mut :: Tensor s a
          mut = reshape (expand ([mu] :: Vector 1 a) :: Tensor '[Product s] a)
          sigmat :: Tensor s a
          sigmat = reshape (expand ([sigma] :: Vector 1 a) :: Tensor '[Product s] a)

mixture_logpdf :: forall a s. (Ord a, Floating a, SingI s, KnownNat (Product s)) => [a] -> [a] -> [a] -> Tensor s a -> Tensor s a
mixture_logpdf ws mus sigmas xs = logsumexp comp_pdfs
    where comp_pdfs :: [Tensor s a]
          comp_pdfs = zipWith3 (\w mu sigma -> log w + normal_logpdf mu sigma xs) wst mus sigmas
          wst :: [Tensor s a]
          wst = map (\w -> reshape (expand ([w] :: Vector 1 a) :: Tensor '[Product s] a)) ws