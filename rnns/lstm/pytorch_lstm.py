import torch
import json
import sys
import futhark_data
import time
import os
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

  # layers, bs, n, d, h
parameters = [ (1, 3, 100, 50, 20)
#             , (1, 3, 20, 300, 192)
             ]

def gen_data():
  for params in parameters:
    model = RNNLSTM(*params)
    model.test()

class RNNLSTM(torch.nn.Module):
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
    self.hidn_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
    self.cell_st0 =torch.zeros(self.num_layers, self.bs, self.h).to(device)
    self.input_ = torch.randn(self.n, self.bs, self.d).to(device)
    self.target = torch.randn(self.n, self.bs, self.d).to(device)
    self.filename =  (f"data/lstm-{self.num_layers}"
                      f"-{self.bs}"
                      f"-{self.n}"
                      f"-{self.d}"
                      f"-{self.h}"
                      f"-{self.output_size}")
    self.lstm = torch.nn.LSTM(input_size = d
                     , hidden_size = h
                     , num_layers = num_layers
                     , bias = True
                     , batch_first = False
                     , dropout = 0
                     , bidirectional = False
                     , proj_size = 0)
    self.linear = torch.nn.Linear(self.output_size, self.d)
    self.res = None
    self.grad_res = None

  def dump(self):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    d = {}
    d['input']    = self.input_
    d['hidn_st0'] = self.hidn_st0
    d['cell_st0'] = self.cell_st0
    for name, p in self.lstm.named_parameters():
      d[name] = p

    for name, p in self.linear.named_parameters():
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

    with open(self.filename + ".json",'w') as f:
       json.dump({name: p.tolist() for name, p in d.items()}, f)

    with open(self.filename + ".in",'wb') as f:
      for xs in d_futhark.values():
        futhark_data.dump(xs, f, True)

  def dump_output(self):
    if not os.path.exists(os.path.dirname(self.filename)):
      os.makedirs(os.path.dirname(self.filename))
    with open(self.filename + ".out",'w') as f:
      futhark_data.dump(self.res.cpu().detach().numpy(),f, True)
      futhark_data.dump(self.res.cpu().detach().numpy(),f, True)

  def forward(self, input_):
   outputs, st = self.lstm(self.input_, (self.hidn_st0, self.cell_st0))
   output = torch.reshape(self.linear(torch.cat([t for t in outputs])), (self.n, self.bs, self.d))
   self.res = output
   return output, st

  def grad(self, input_):
    self.zero_grad()
    output = self(input_)
    loss_function = torch.nn.MSELoss()
    loss = loss_function(input_, self.target)
    loss.backward()
    self.grad_res = model.grad()

  def test(self, verbose=True):
    self.to(device)
    forward_start = time.time()
    self.forward(self.input_)
    forward_end = time.time()
    grad_start  = time.time()
    self.grad(self.input_)
    grad_end = time.time()
    self.dump()
    self.dump_output()
    if verbose:
      print (f"forward time: {forward_end-forward_start}, grad time: {grad_end-grad_start}")


if __name__ == '__main__':
  gen_data()
