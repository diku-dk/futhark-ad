import "util"
import "optim"


module T = tensor_ops f64

module betabinomial_model: {
  include bound_optimizable with param.t = (f64, f64, f64)
                            with loss.t = f64
                            with data = []f64
  val betabinomial_logpmf: param.t -> data
} = {
    
}
