# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import futhark_data
import gzip

torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class PyTorchGMM(torch.nn.Module):

    '''Test class for GMM differentiation by PyTorch.'''

    def prepare(self, inputs):
        super().__init__()
        '''Prepares calculating. This function must be run before
        any others.'''
        self.inputs = to_torch_tensors(
            (inputs[0], inputs[1], inputs[2]), grad_req = True
        )

        self.params = to_torch_tensors(
            (inputs[3], inputs[4], inputs[5])
        )

        self.objective = torch.zeros(1, device=device)
        self.gradient = torch.empty(0, device=device)

    def output(self):
        '''Returns calculation result.'''

        return self.objective.item(), self.gradient.numpy()

    def calculate_objective(self, times):
        '''Calculates objective function many times.'''

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(times):
            self.objective = gmm_objective(*self.inputs, *self.params)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / times

    def calculate_jacobian(self, times):
        '''Calculates objective function jacobian many times.'''

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(times):
            self.objective, self.gradient = torch_jacobian(
                gmm_objective,
                self.inputs,
                self.params
            )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / times

def load(filename):
   f = gzip.open(filename + ".in.gz")
   return futhark_data.load(f)

def test(runs = 5):
   g = load("gmm/data/10k/gmm_d10_K50")
   gmm = PyTorchGMM()
   gmm.prepare(list(g))
   gmm.to(device)
   f_time = gmm.calculate_objective(runs)
   j_time = gmm.calculate_jacobian(runs)
   print(f_time * 1000)
   print(gmm.objective)
   print(j_time * 1000)
   print(gmm.gradient)

def logsumexp(x):
    mx = torch.max(x)
    emx = torch.exp(x - mx)
    return torch.log(sum(emx)) + mx


def logsumexpvec(x):
    '''The same as "logsumexp" but calculates result for each row separately.'''

    mx = torch.max(x, 1).values
    lset = torch.logsumexp(torch.t(x) - mx, 0)
    return torch.t(lset + mx)


def log_gamma_distrib(a, p):
    return torch.special.multigammaln(a, p)


def sqsum(x):
    return sum(x ** 2)


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]

    out = torch.sum(
        0.5 * wishart_gamma * wishart_gamma *
        (torch.sum(Qdiags ** 2, dim = 1) + torch.sum(icf[:,p:] ** 2, dim = 1)) -
        wishart_m * sum_qs
    )

    C = n * p * (math.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - log_gamma_distrib(0.5 * n, p))


def constructL(d, icf):
    constructL.Lparamidx = d

    def make_L_col(i):
        nelems = d - i - 1
        col = torch.cat([
            torch.zeros(i + 1, dtype = torch.float64),
            icf[constructL.Lparamidx:(constructL.Lparamidx + nelems)]
        ])

        constructL.Lparamidx += nelems
        return col

    columns = [make_L_col(i) for i in range(d)]
    return torch.stack(columns, -1)


def Qtimesx(Qdiag, L, x):

    f = torch.einsum('ijk,mik->mij', L, x)
    return Qdiag * x + f


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)
    Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])
    
    xcentered = torch.stack(tuple( x[i] - means for i in range(n) ))
    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return CONSTANT + slse - n * logsumexp(alphas) \
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)

def gmm_jacobian(inputs, params):
    torch_jacobian(gmm_objective, inputs = inputs, params = params)

def to_torch_tensor(param, grad_req = False, dtype = torch.float64):
    '''Converts given single parameter to torch tensors. Note that parameter
    can be an ndarray-like object.
    
    Args:
        param (ndarray-like): parameter to convert.
        grad_req (bool, optional): defines flag for calculating tensor
            jacobian for created torch tensor. Defaults to False.
        dtype (type, optional): defines a type of tensor elements. Defaults to
            torch.float64.

    Returns:
        torch tensor
    '''

    return torch.tensor(
        param,
        dtype = dtype,
        requires_grad = grad_req,
        device=device
    )



def to_torch_tensors(params, grad_req = False, dtype = torch.float64):
    '''Converts given multiple parameters to torch tensors. Note that
    parameters can be ndarray-lake objects.
    
    Args:
        params (enumerable of ndarray-like): parameters to convert.
        grad_req (bool, optional): defines flag for calculating tensor
            jacobian for created torch tensors. Defaults to False.
        dtype (type, optional): defines a type of tensor elements. Defaults to
            torch.float64.

    Returns:
        tuple of torch tensors
    '''

    return tuple(
        torch.tensor(param, dtype = dtype, requires_grad = grad_req, device=device)
        for param in params
    )



def torch_jacobian(func, inputs, params = None, flatten = True):
    '''Calculates jacobian and return value of the given function that uses
    torch tensors.

    Args:
        func (callable): function which jacobian is calculating.
        inputs (tuple of torch tensors): function inputs by which it is
            differentiated.
        params (tuple of torch tensors, optional): function inputs by which it
            is doesn't differentiated. Defaults to None.
        flatten (bool, optional): if True then jacobian will be written in
            1D array row-major. Defaults to True.

    Returns:
        torch tensor, torch tensor: function result and function jacobian.
    '''

    def recurse_backwards(output, inputs, J, flatten):
        '''Recursively calls .backward on multi-dimensional output.'''

        def get_grad(tensor, flatten):
            '''Returns tensor gradient flatten representation. Added for
            performing concatenation of scalar tensors gradients.'''

            if tensor.dim() > 0:
                if flatten:
                    return tensor.grad.flatten()
                else:
                    return tensor.grad
            else:
                return tensor.grad.view(1)


        if output.dim() > 0:
            for item in output:
                recurse_backwards(item, inputs, J, flatten)
        else:
            for inp in inputs:
                inp.grad = None

            output.backward(retain_graph = True)

            J.append(torch.cat(
                list(get_grad(inp, flatten) for inp in inputs)
            ))

    if params != None:
        res = func(*inputs, *params)
    else:
        res = func(*inputs)

    J = []
    recurse_backwards(res, inputs, J, flatten)

    J = torch.stack(J)

    if flatten:
        J = J.t().flatten()
    return res, J
