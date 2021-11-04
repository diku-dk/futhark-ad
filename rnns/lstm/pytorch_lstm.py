import torch
import json
import sys
import futhark_data
import time
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import chain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# layers, bs, n, d, h
parameters = [  (1, 2, 3, 4, 3)
#             , (1, 3, 5, 10, 5)
#             , (1, 3, 100, 50, 20)
#             , (1, 3, 20, 300, 192)
             ]

def read(filename):
   with open(filename + ".json",'r') as f:
    d = json.load(f)
    for name, p in d.items():
        d[name] = torch.tensor(p, dtype=torch.float32)
    return d

def gen_data():
  for params in parameters:
    (_,bs,n,d,_,) = params
    input_ = torch.randn(n, bs, d).to(device)
    target = torch.randn(n, bs, d).to(device)
    model = RNNLSTM(*params)
    model.test(input_, target)
    d = read(model.filename)
    naive = NaiveLSTM(d)
    naive.test(input_, target)

class NaiveLSTM(nn.Module):
    def __init__(  self
                 , params
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

        # parameters of the (recurrent) hidden layer
        self.W_i, self.W_f, self.W_j, self.W_o = \
                     torch.chunk(params['weight_ih_l0'], 4)

        self.b_i, self.b_f, self.b_j, self.b_o = \
                   torch.chunk(params['bias_ih_l0'], 4)

        self.U_i, self.U_f, self.U_j, self.U_o = \
             torch.chunk(params['weight_hh_l0'], 4)
        
        # initial hidden state
        self.h_0 = params['hidn_st0']
        self.c_0 = params['cell_st0']
        
        # output layer (fully connected)
        self.W_y = params['weight']
        self.b_y = params['bias']

        self.input_ = params['input']
        self.target = params['target']
        self.grads = None
                
    def step(self, x_t, h, c):
        #  forward pass for a single time step
        # hint: a more clever implementation could combine all these and select the different parts later
        j = self.activation_j(
            torch.matmul(self.W_j, x_t) + torch.matmul(self.U_j,h) + self.b_j
        )
        i = self.activation_i(
            torch.matmul(self.W_i, x_t) + torch.matmul(self.U_i,h) + self.b_i
        )
        f = self.activation_f(
            torch.matmul(self.W_f, x_t) + torch.matmul(self.U_f,h) + self.b_f
        )        
        o = self.activation_o(
            torch.matmul(self.W_o, x_t) + torch.matmul(self.U_o,h) + self.b_o
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
            h, c = self.step(x[:, t], h, c)
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, 1)
        
        # fully connected output
        y_hat = hidden_states.reshape(batch_size * n_steps, -1) # flatten steps and batch size (bs * )
        y_hat = torch.matmul(y_hat, self.W_y) + self.b_y
        y_hat = y_hat.reshape(batch_size, n_steps, -1) # regains structure
        return y_hat, hidden_states[:, -1], c
    
    def forward(self, x, h, c):
        # x: b, t, d
        batch_size = x.shape[0] 
        if h is None:
            h = self.h_0.repeat_interleave(batch_size, 0)
        if c is None:
            c = self.c_0.repeat_interleave(batch_size, 0)
        y_hat, h, c = self.iterate_series(x, h, c)
        return y_hat, h, c
    
    def auto_regression(self, x_0, h, c, steps):
        # one-to-many task (steps = \delta')
        x_prev = x_0
        y_hat = []
        # iterate over time axis (1)
        for t in range(steps):
            # give previous hidden state and input from the current time step
            h, c = self.step(x_prev, h, c)
            # here we need to apply the output layer on each step individually
            x_prev = torch.matmul(h, self.W_y) + self.b_y
            
            y_hat.append(x_prev)
        y_hat = torch.stack(y_hat, 1)
        return y_hat, h, c
    
    def many_to_one(self, x, h, c):
        # not required
        # returns the last prediction and the hidden state
        y_hat, h, c = self(x, h, c)
        return y_hat[:, -1], h, c # only return the last prediction
        
    def many_to_many_async(self, x, h, c, steps):
        # not required
        # combines many-to-one and one-to-many
        x_0, h, c = self.many_to_one(x, h, c)
        return self.auto_regression(x_0, h, c, steps)

    def vjp(self, input_, target):
        x = self.input_ if self.input_ else input_
        y = self.target if self.target else target
        x, y = x.to(device), y.to(device)
        h = c = None
        # reset gradients
        #optimizer.zero_grad()
        
        # get predictions (forward pass)
        y_hat, h, c = self(x, h, c)
                 
        # calculate mean squared error
        loss = torch.mean((y_hat - y)**2)        
        # backprop
        loss.backward()
        print("loss")
        print(loss)
        #self.grads = dict(chain(self.lstm.named_parameters(), self.linear.named_parameters()))
        #print("grads")
        #for name, p in self.grads.items():
        #  print(name)
        #  print(p.size())
        #  print(p)
    

    def test(self, input_, target, verbose=True):
        input_ = input_ if self.input_ == None else self.input_
        target = target if self.target == None else self.target
        #self.to(device)
        forward_start = time.time()
        self.forward(input_, None, None)
        forward_end = time.time()
        vjp_start  = time.time()
        self.vjp(input_, target)
        vjp_end = time.time()
        #self.dump(input_, target)
        #self.dump_output()
        if verbose:
          print(self.filename)
          print(f"forward time: {forward_end-forward_start}")
          print(f"grad time   : {vjp_end-vjp_start}")
          print(f"total time  : {vjp_end - forward_start}")
          print()

class RNNLSTM(nn.Module):
  def __init__( self
              , num_layers
              , bs
              , n
              , d
              , h):

    super(RNNLSTM,self).__init__()
    self.bs = bs
    self.n = n
    self.num_layers = num_layers
    self.h = h
    self.d = d
    self.output_size = h
    self.filename =  (f"data/lstm-{self.num_layers}"
                      f"-{self.bs}"
                      f"-{self.n}"
                      f"-{self.d}"
                      f"-{self.h}"
                      f"-{self.output_size}")
    self.lstm = nn.LSTM(input_size = d
                     , hidden_size = h
                     , num_layers = num_layers
                     , bias = True
                     , batch_first = False
                     , dropout = 0
                     , bidirectional = False
                     , proj_size = 0)

    self.hidn_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
    self.cell_st0 =torch.zeros(self.num_layers, self.bs, self.h).to(device)
    self.linear = nn.Linear(self.output_size, self.d)
    self.res = None
    self.grads = None
    self.input_ = None
    self.target = None

    if os.path.isfile(self.filename):
      self.input_, self.target = read()

  def dump(self, input_, target):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    d = {}
    d['input']    = input_
    d['target']    = target
    d['hidn_st0'] = self.hidn_st0
    d['cell_st0'] = self.cell_st0
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

    d_futhark['loss_adj'] = np.float32(1.0)

    with open(self.filename + ".json",'w') as f:
       json.dump({name: p.tolist() for name, p in d.items()}, f)

    with open(self.filename + ".in",'w') as f:
      for xs in d_futhark.values():
        futhark_data.dump(xs, f, False)

  def dump_output(self):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    with open(self.filename + ".out",'w') as f:
      futhark_data.dump(self.res.cpu().detach().numpy(),f, False)
      for g in self.grads.values():
        futhark_data.dump(g.cpu().detach().numpy(),f,False)

  def read(self):
    d = json.load(self.filename + ".json",'w')
    for name, p in d.items():
        d[name] = torch.tensor(p, dtype=torch.float32)
    with torch.no_grad():
      for name, p in chain(self.lstm.named_parameters(), self.linear.named_parameters()):
        p.copy_(d[name])
    return d['input'], d['target']

  def forward(self, input_):
   input_ = self.input_ if self.input_ else input_
   outputs, st = self.lstm(input_, (self.hidn_st0, self.cell_st0))
   print("st")
   print(st)
   print("outputs")
   print(outputs)
   output = torch.reshape(self.linear(torch.cat([t for t in outputs])), (self.n, self.bs, self.d))
   self.res = output
   return output

  def vjp(self, input_, target):
    input_ = self.input_ if self.input_ else input_
    target = self.target if self.target else target
    self.zero_grad()
    output = self(input_)
    print("output")
    print(output)
    print("target")
    print(target)
    loss_function = nn.MSELoss(reduction='mean')
    loss = loss_function(output, target)
    loss.backward(gradient=torch.tensor(1.0))
    print("loss")
    print(loss)
    self.grads = dict(chain(self.lstm.named_parameters(), self.linear.named_parameters()))
    print("grads")
    for name, p in self.grads.items():
      print(name)
      print(p.size())
      print(p)

  def test(self, input_, target, verbose=True):
    input_ = self.input_ if self.input_ else input_
    target = self.target if self.target else target
    self.to(device)
    forward_start = time.time()
    self.forward(input_)
    forward_end = time.time()
    vjp_start  = time.time()
    self.vjp(input_, target)
    vjp_end = time.time()
    self.dump(input_, target)
    self.dump_output()
    if verbose:
      print(self.filename)
      print(f"forward time: {forward_end-forward_start}")
      print(f"grad time   : {vjp_end-vjp_start}")
      print(f"total time  : {vjp_end - forward_start}")
      print()

if __name__ == '__main__':
  gen_data()
