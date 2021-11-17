import torch
import json
from pprint import pprint
import sys
import futhark_data
import time
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import chain

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# bs, n, d, h
parameters = [
            # (2, 3, 4, 3)
            #, (3, 5, 10, 5)
            #, (10, 100, 50, 20)
              (1024, 20, 300, 192)
            #(1024, 300, 80, 256)
             ]

def get_time():
   return time.time() * 10**6

def equal(m1, m2):
   grads_equal = True
   for n in m1.grads.keys():
     grads_equal = grads_equal and torch.allclose(m1.grads[n], m2.grads[n], 0.0001, 0.0001)
   return torch.allclose(m1.output, m2.output, 0.0001, 0.0001) \
     and torch.allclose(m1.loss, m2.loss, 0.0001, 0.0001) \
     and grads_equal

def gen_filename(bs, n, d, h, directory="data", ext=None):
  path = f"{directory}/lstm-bs{bs}-n{n}-d{d}-h{h}"

  return path if ext is None else f"{path}.{ext}"

def report_time(model, ft, rt, filename=None):
  if filename:
    print(f"{model}   {filename}:")
  else:
    print(f"{model}:")
  print(f"forward time: {ft*1000}μs")
  print(f"grad time   : {rt*1000}μs")
  print(f"relative: {rt/ft}")

def read(filename):
   with open(filename + ".json",'r') as f:
    d = json.load(f)
    for name, p in d.items():
        d[name] = torch.tensor(p, dtype=torch.float64)
    return d

def gen_data(verbose=True):
  for params in parameters:
    (bs,n,d,h) = params
    filename = gen_filename(bs, n, d, h, ext=None)
    model = RNNLSTM(*params, filename)
    model.run(runs = 1, gen_data=True)

def print_values(name, model):
  print(f"Values for {name}:")
  print("Output:")
  pprint(model.output)
  print(f"Loss: {model.loss}")
  print("Grads:")
  pprint(model.grads)

def test(mkdata=False, verbose=False, runs=1):
  if mkdata: gen_data(verbose=False)
  for params in parameters:
    (bs,n,d,h) = params
    filename = gen_filename(bs, n, d, h, ext=None)
    tensors = read(filename)
    model = RNNLSTM(*params, filename, tensors)
    ft_model, rt_model = model.run(runs, tensors['input'].to(device), tensors['target'].to(device))
    naive = NaiveLSTM(tensors)
    ft_naive, rt_naive = naive.run(runs, filename=filename)

    if verbose:
          print_values("torch.nn.LSTM", model)
          print_values("naive", naive)
          if equal(model, naive):
            print(f"test data {filename} validates!")
          else:
            print(f"Error: test data {filename} doesn't validate!")
          print()

    report_time("naive        ", ft_naive, rt_naive, filename)
    report_time("torch.nn.LSTM", ft_model, rt_model, filename)
    print()

class NaiveLSTM(nn.Module):
  def __init__(  self
               , tensors
               , activation_h=nn.Tanh
               , activation_o=nn.Sigmoid
               , activation_f=nn.Sigmoid
               , activation_i=nn.Sigmoid
               , activation_j=nn.Tanh
               ):
      super().__init__()


      # it is fine to hard code these
      self.activation_h = activation_h()
      self.activation_o = activation_o()
      self.activation_f = activation_f()
      self.activation_i = activation_i()
      self.activation_j = activation_j()

      self.input_ = tensors['input']
      self.target = tensors['target']

      # parameters of the (recurrent) hidden layer
      self.W_i, self.W_f, self.W_j, self.W_o = \
           tuple(nn.Parameter(torch.transpose(t, 0, 1)) for t in torch.chunk(tensors['weight_ih_l0'], 4))

      self.b_ii, self.b_if, self.b_ij, self.b_io = \
           tuple(nn.Parameter(t) for t in torch.chunk(tensors['bias_ih_l0'], 4))

      self.b_hi, self.b_hf, self.b_hj, self.b_ho = \
           tuple(nn.Parameter(t) for t in torch.chunk(tensors['bias_hh_l0'], 4))

      self.U_i, self.U_f, self.U_j, self.U_o = \
           tuple(nn.Parameter(torch.transpose(t, 0, 1)) for t in torch.chunk(tensors['weight_hh_l0'], 4))

      # initial hidden state
      self.h_0 = nn.Parameter(tensors['hidn_st0'][0])
      self.c_0 = nn.Parameter(tensors['cell_st0'][0])

      # output layer (fully connected)
      self.W_y = nn.Parameter(torch.transpose(tensors['weight'], 0, 1))
      self.b_y = nn.Parameter(tensors['bias'])


      self.input_ = nn.Parameter(torch.transpose(tensors['input'], 0, 1))
      self.target = nn.Parameter(torch.transpose(tensors['target'], 0, 1))

  def step(self, x_t, h, c):
      #  forward pass for a single time step
      j = self.activation_j(
          torch.matmul(x_t, self.W_j) + torch.matmul(h, self.U_j) + self.b_ij + self.b_hj
      )
      i = self.activation_i(
          torch.matmul(x_t, self.W_i) + torch.matmul(h, self.U_i) + self.b_ii + self.b_hi
      )
      f = self.activation_f(
          torch.matmul(x_t, self.W_f) + torch.matmul(h, self.U_f) + self.b_if + self.b_hf
      )
      o = self.activation_o(
          torch.matmul(x_t, self.W_o) + torch.matmul(h, self.U_o) + self.b_io + self.b_ho
      )

      c = f * c + i * j

      h = o * self.activation_h(c)

      return h, c # returning new hidden and cell state

  def iterate_series(self, x, h, c):
      # apply rnn to each time step and give an output (many-to-many task)
      batch_size, n_steps, dimensions = x.shape

      # can use cell states list here but only the last cell is required
      hidden_states = []
      # iterate over time axis (1)
      for t in range(n_steps):
          # give previous hidden state and input from the current time step
          h, c = self.step(x[:,t], h, c)
          hidden_states.append(h)
      hidden_states = torch.stack(hidden_states, 1)

      # fully connected output
      y_hat = hidden_states.reshape(batch_size * n_steps, -1) # flatten steps and batch size (bs * )
      y_hat = torch.matmul(y_hat, self.W_y) + self.b_y
      y_hat = y_hat.reshape(batch_size, n_steps, -1) # regains structure
      return y_hat, hidden_states[:, -1], c

  def forward(self, x, h, c):
      if h is None:
          h = self.h_0
      if c is None:
          c = self.c_0
      y_hat, h, c = self.iterate_series(x, h, c)
      self.output = torch.transpose(y_hat, 0, 1)
      return y_hat, h, c

  def vjp(self, input_, target, runs):
      self.zero_grad()
      x = input_ if self.input_ == None else self.input_
      y = target if self.target == None else self.target
      h = c = None

      # get predictions (forward pass)
      y_hat, h, c = self(x, h, c)
  
      loss = torch.mean((y_hat - y)**2)
      # backprop
      loss.backward(gradient=torch.tensor(1.0))

      start  = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      for i in range(runs):

        self.zero_grad()
        x = input_ if self.input_ == None else self.input_
        y = target if self.target == None else self.target
        h = c = None
        # get predictions (forward pass)
        y_hat, h, c = self(x, h, c)
  
        loss = torch.mean((y_hat - y)**2)
        # backprop
        loss.backward(gradient=torch.tensor(1.0))
      torch.cuda.synchronize()
      end.record()

      self.loss = loss
      d = {n: p.grad for n, p in self.named_parameters()}
      self.grads = {  'weight_ih_l0': torch.concat([torch.transpose(g, 0, 1) for g in [d['W_i'], d['W_f'], d['W_j'], d['W_o']]])
                    , 'weight_hh_l0': torch.concat([torch.transpose(g, 0, 1) for g in [d['U_i'], d['U_f'], d['U_j'], d['U_o']]])
                    , 'bias_ih_l0'  : torch.concat([d['b_ii'], d['b_if'], d['b_ij'], d['b_io']])
                    , 'bias_hh_l0'  : torch.concat([d['b_hi'], d['b_hf'], d['b_hj'], d['b_ho']])
                    , 'weight'      : torch.transpose(d['W_y'], 0, 1)
                    , 'bias'        : d['b_y']
                   }
      return (start.elapsed_time(end) / runs)

  def run(self, runs, filename=None, input_=None, target=None):
      input_ = self.input_ if input_ is None else input_
      target = self.target if target is None else target
      self.to(device)
      start  = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      for i in range(runs):
        self.forward(input_, None, None)
      end.record()
      vjp_time = self.vjp(input_, target, runs)
      return (start.elapsed_time(end) / runs), vjp_time

class RNNLSTM(nn.Module):
  def __init__( self
              , bs
              , n
              , d
              , h
              , filename
              , tensors=None):

    super(RNNLSTM,self).__init__()
    self.num_layers = 1
    self.bs = bs
    self.n = n
    self.h = h
    self.d = d
    self.filename = filename
    self.lstm = nn.LSTM(input_size = self.d
                     , hidden_size = self.h
                     , num_layers = self.num_layers
                     , bias = True
                     , batch_first = False
                     , dropout = 0
                     , bidirectional = False
                     , proj_size = 0)
    self.linear = nn.Linear(self.h, self.d)

    if tensors is None:
      self.hidn_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
      self.cell_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
    else:
      self.hidn_st0 = tensors['hidn_st0'].to(device)
      self.cell_st0 = tensors['cell_st0'].to(device)
      with torch.no_grad():
        for n, p in chain(self.lstm.named_parameters(), self.linear.named_parameters()):
          p.copy_(tensors[n].clone().detach())

  def dump(self, input_, target):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    d = { 'input' :   input_
        , 'target':   target
        , 'hidn_st0': self.hidn_st0
        , 'cell_st0': self.cell_st0
        }
    for name, p in chain(self.lstm.named_parameters(), self.linear.named_parameters()):
      d[name] = p

    d_futhark = {}
    for name, p in d.items():
      xs = p.cpu().detach().numpy()
      if name == 'hidn_st0':
        d_futhark[name] = xs[0,:,:].T
      elif name == 'cell_st0':
        d_futhark[name] = xs[0,:,:].T
      elif name == 'weight':
        d_futhark[name] = xs.T
      else:
        d_futhark[name] = xs

    d_futhark['loss_adj'] = np.float64(1.0)

    with open(self.filename + ".json",'w') as f:
       json.dump({name: p.tolist() for name, p in d.items()}, f)

    with open(self.filename + ".in",'wb') as f:
      for xs in d_futhark.values():
        futhark_data.dump(xs, f, True)

  def dump_output(self):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    with open(self.filename + ".out",'wb') as f:
      #futhark_data.dump(self.output.cpu().detach().numpy().reshape(self.bs*self.n, -1),f, True)
      futhark_data.dump(self.loss.cpu().detach().numpy(),f, True)
    with open(self.filename + ".J",'wb') as f:
      for n, g in self.grads.items():
        if n == 'weight':
          futhark_data.dump(g.cpu().detach().numpy().T,f,True)
        else:
          futhark_data.dump(g.cpu().detach().numpy(),f,True)

  def read(self):
    d = json.load(self.filename + ".json",'w')
    for name, p in d.items():
        d[name] = torch.tensor(p, dtype=torch.float64)
    with torch.no_grad():
      for name, p in chain(self.lstm.named_parameters(), self.linear.named_parameters()):
        p.copy_(d[name])
    return d['input'], d['target']

  def forward(self, input_):
   outputs, st = self.lstm(input_, (self.hidn_st0, self.cell_st0))
   output = torch.reshape(self.linear(torch.cat([t for t in outputs])), (self.n, self.bs, self.d))
   self.output = output
   return output

  def vjp(self, input_, target, runs):

    self.zero_grad()
    output = self(input_)
    loss_function = nn.MSELoss(reduction='mean')
    loss = loss_function(output, target)
    loss.backward(gradient=torch.tensor(1.0))

    start  = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(runs):
      self.zero_grad()
      output = self(input_)
      loss_function = nn.MSELoss(reduction='mean')
      loss = loss_function(output, target)
      loss.backward(gradient=torch.tensor(1.0))
    torch.cuda.synchronize()
    end.record()
    self.loss = loss
    self.grads = \
      {n: p.grad for n, p in chain(self.lstm.named_parameters(), self.linear.named_parameters())}
    return (start.elapsed_time(end) / runs)

  def run(self, runs, input_=None, target=None, gen_data=False):
    if not gen_data and (input_ is None or target is None):
       print("Error: must input and target data!")
       exit(1)
    if gen_data:
      input_ = torch.randn(self.n, self.bs, self.d).to(device)
      target = torch.randn(self.n, self.bs, self.d).to(device)
    self.to(device)
    self.forward(input_)
    start  = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(runs):
      self.forward(input_)
    torch.cuda.synchronize()
    end.record()
    vjp_time = self.vjp(input_, target, runs)
    if gen_data:
      self.dump(input_, target)
      self.dump_output()
    return (start.elapsed_time(end) / runs), vjp_time
