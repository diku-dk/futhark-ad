import torch
import json
import sys
import futhark_data
import time
from torch.utils.data import Dataset, DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class CustomData(Dataset):
  def __init__(self, input_, target):
    self.input_ = input_
    self.target = target

  def __len__(self):
    return len(self.target)

  def __getitem__(self,idx):
    return input_[idx], target[idx]


class RNNLSTM(torch.nn.Module):
  def __init__( self
              , n = 3
              , length = 100
              , num_layers = 1
              , hidden_size = 20
              , num_features = 50
                , output_size = 20):

    super(RNNLSTM,self).__init__()
    self.n = n
    self.length = length
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.num_features = num_features
    self.output_size = output_size # generally same as hidden_size
    self.hidn_st0 = torch.zeros(self.num_layers, self.n, self.hidden_size).to(device)
    self.cell_st0 =torch.zeros(self.num_layers, self.n, self.hidden_size).to(device)
    self.input_ = torch.randn(self.length, self.n, self.num_features).to(device)
    self.target = torch.randn(self.length, self.n, self.num_features).to(device)
    self.training_data = CustomData(self.input_, self.target)
    self.dataloader = DataLoader(self.training_data)

    self.lstm = torch.nn.LSTM(input_size = num_features
                     , hidden_size = hidden_size
                     , num_layers = num_layers
                     , bias = True
                     , batch_first = False
                     , dropout = 0
                     , bidirectional = False
                     , proj_size = 0)

    self.linear = torch.nn.Linear(self.output_size, self.num_features)
    self.res = None

  def dump(self):
    d = {}
    d['input']    = self.input_
    d['hidn_st0'] = self.hidn_st0
    d['cell_st0'] = self.cell_st0
    for name, p in self.lstm.named_parameters():
      d[name] = p

    for name, p in self.linear.named_parameters():
      d[name] = p

    filename = (f"variables-{self.n}"
                f"-{self.num_features}"
                f"-{self.length}"
                f"-{self.num_layers}"
                f"-{self.hidden_size}")
    with open(filename + ".json",'w') as f:
       json.dump({name: p.tolist() for name, p in d.items()}, f)
    
    with open(filename + ".in",'wb') as f:
      for name, p in d.items():
        p_ = p.cpu()
        if self.n == 1:
          if name == 'input':
              futhark_data.dump(p_.detach().numpy()[:,0,:],f, True)
          elif name == 'hidn_st0':
              futhark_data.dump(p_.detach().numpy()[0,0,:],f, True)
          elif name == 'cell_st0':
              futhark_data.dump(p_.detach().numpy()[0,0,:],f, True)
          elif name == 'weight':
              futhark_data.dump(p_.detach().numpy().T,f, True)
          else:
              futhark_data.dump(p_.detach().numpy(),f, True)
        else:
          if name == 'hidn_st0':
              futhark_data.dump(p_.detach().numpy()[0,:,:].T,f, True)
          elif name == 'cell_st0':
              futhark_data.dump(p_.detach().numpy()[0,:,:].T,f, True)
          elif name == 'weight':
              futhark_data.dump(p_.detach().numpy().T,f, True)
          else:
              futhark_data.dump(p_.detach().numpy(),f, True)

    with open(filename + ".out",'w') as f:
      futhark_data.dump(self.res.cpu().detach().numpy(),f, False)

  def forward(self, input_):
   outputs, _ = self.lstm(input_, (self.hidn_st0, self.cell_st0))
   output = torch.reshape(self.linear(torch.cat([t for t in outputs])), (self.length, self.n, self.num_features))
   self.res = output
   return output

if __name__ == '__main__':
  rnnlstm = RNNLSTM()
  start = time.time()
  rnnlstm.to(device)
  y_hat = rnnlstm.forward(rnnlstm.input_.to(device))
  print("y_hat %s:" % (time.time() - start))
  print(y_hat)
  rnnlstm.dump()

