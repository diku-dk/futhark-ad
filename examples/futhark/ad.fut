-- | AD library based on work by Chris Hammill and Ben Darwin at
-- https://github.com/cfhammill/futhark-forward-AD

-- | Construct a module that uses a dual number representation of
-- scalars.
module mk_dual(T: real): {
  type r = T.t
  type t = (r, r)
  val inject: r -> t
  val set_deriv: t -> r -> t
  val get_deriv: t -> r
  val make_dual: r -> r -> t

  include from_prim with t = (r,r)
  include numeric with t = (r,r)
  include real with t = (r,r)

} = {
  type r = T.t
  type t = (r, r)
  let inject x = (x, T.i32 0)
  let i8 (x : i8) = inject (T.i8 x)
  let i16 (x : i16) = inject (T.i16 x)
  let i32 (x : i32) = inject (T.i32 x)
  let i64 (x : i64) = inject (T.i64 x)
  let f32 (x : f32) = inject (T.f32 x)
  let f64 (x : f64) : t = inject (T.f64 x)
  let u8 (x : u8) = inject (T.u8 x)
  let u16 x = inject (T.u16 x)
  let u32 x = inject (T.u32 x)
  let u64 x = inject (T.u64 x)
  let bool x = inject (T.bool x)

  let (x,x') + (y,y') = T.( (x + y, x' + y')  )
  let (x,x') - (y,y') = T.( (x - y, x' - y')  )
  let (x,x') * (y,y') = T.( (x * y, x' * y + x * y')  )

  let (x,x') / (y,y') = T.( (x / y, (x' * y - x * y') / y ** (i32 2)) )

  let (x,x') ** (y,y') = T.( (x / y, (x' * y - x * y') / y ** (i32 2)) )

  let (x,_) == (y,_) = T.( x == y )
  let (x,_) < (y,_) = T.( x < y )
  let (x,_) > (y,_) = T.( x > y )
  let (x,_) <= (y,_) = T.( x <= y )
  let (x,_) >= (y,_) = T.( x >= y )
  let (x,_) != (y,_) = T.( x != y )
  let negate (x,x') = T.( (negate x, negate x') )
  let max x y = if x >= y then x else y
  let min x y = if x <= y then x else y
  let abs (x,x') = (T.abs x, x')
  let sgn (x,x') = (T.sgn x, x')
  let highest = inject T.highest
  let lowest = inject T.lowest
  -- | Returns zero on empty input.
  let sum = reduce (+) (inject (T.i32 0))
  -- | Returns one on empty input.
  let product = reduce (*) (inject (T.i32 1))
  -- | Returns `lowest` on empty input.
  let maximum = reduce min highest
  -- | Returns `highest` on empty input.
  let minimum = reduce max lowest


  -- val from_fraction: i32 -> i32 -> t
  let from_fraction x y = inject (T.from_fraction x y)
  -- val to_i32: t -> i32
  let to_i32 (x,_) = T.to_i32 x
  let to_i64 (x,_) = T.to_i64 x
  let to_f64 (x,_) = T.to_f64 x


  -- val sqrt: t -> t
  let sqrt (x,x') = T.( (sqrt x, x' / (i32 2 * sqrt x)) )
  -- val exp: t -> t
  let exp (x,x') = T.( (exp x, x' * exp x) )
  -- val cos: t -> t
  let cos (x, x') = T.( (cos x, negate x' * sin x) )
  -- val sin: t -> t
  let sin (x, x') = T.( (sin x, x' * cos x) )
  let tan x = sin x / cos x
  -- val asin: t -> t
  let asin (x, x') = T.( (asin x, x' / (sqrt (i32 1 - x ** i32 2))) )
  -- val acos: t -> t1
  let acos (x, x') = T.( (acos x, negate x' / (sqrt (i32 1 - x ** i32 2)))  )
  -- val atan: t -> t
  let atan (x, x') = T.( (atan x, x' / (i32 1 + x ** i32 2)) )
  -- val atan2: t -> t -> t
  -- I know this isn't right but can't figure it out now
  let atan2 (x,_) (y,_) = inject (T.atan2 x y)
  -- I have no idea how to differentiate the gamma funtions.
  let gamma (x, _) = inject (T.gamma x)
  let lgamma (x, _) = inject (T.lgamma x)

  -- val log: t -> t
  let log (x, x') = T.( (log x, x' / x) )
  let log2 (x, x') = T.( (log10 x, i32 1 / (x' * log2 x)) )
  let log10 (x, x') = T.( (log10 x, i32 1 / (x' * log10 x)) )

  -- val ceil : t -> t
  let ceil (x, x') = (T.ceil x, x')
  -- val floor : t -> t
  let floor (x, x') = (T.floor x, x')
  -- val trunc : t -> t
  let trunc (x, x') = (T.trunc x, x')
  -- val round : t -> t
  let round (x, x') = (T.round x, x')

  -- val isinf: t -> bool
  let isinf (x,_) = T.isinf x
  -- val isnan: t -> bool
  let isnan (x,_) = T.isnan x

  -- val inf: t
  let inf = inject T.inf
  -- val nan: t
  let nan = inject T.nan

  -- val pi: t
  let pi = inject T.pi
  -- val e: t
  let e = inject T.e

  let get_deriv (_,x') = x'
  let set_deriv (x,_) x'= (x,x')
  let make_dual x x' = (x,x')
}
