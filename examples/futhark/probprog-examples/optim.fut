import "lib/github.com/diku-dk/cpprandom/random"

module type optimizable = {
  module param: real
  module loss: real
  type data
  val eval_loss: param.t -> data -> loss.t
}

module type grad_optimizable = {
  include optimizable
  val grad: param.t -> data -> param.t
}

module type bound_optimizable = {
  include optimizable
  val param_lower: param.t
  val param_upper: param.t
}

module type optimizer = {
  module param: real
  module loss: real
  type data

  type options
  val default_options: options

  val run: options -> param.t -> data -> i32 -> ([]loss.t, param.t)
  val run': param.t -> data -> i32 -> ([]loss.t, param.t)
}

module stochastic_gradient_descent (optable: grad_optimizable): optimizer
                                                                with data = optable.data
                                                                with param.t = optable.param.t
                                                                with loss.t = optable.loss.t
                                                                with options = {learning_rate: f64} = {
  module param = optable.param
  module loss = optable.loss
  type data = optable.data

  type options = {learning_rate: f64}
  let default_options: options = {learning_rate=0.1}

  let run ({learning_rate}: options) (p: param.t) (xs: data) (n_iters: i32) =
  loop (losses, p) = ([optable.eval_loss p xs], p) for _i < n_iters do
    let p = p param.- param.f64 learning_rate param.* optable.grad p xs
    let losses = losses ++ [optable.eval_loss p xs]
    in (losses, p)

  let run' = run default_options
}
