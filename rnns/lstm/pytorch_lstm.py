import torch
import json
import sys
import futhark_data

torch.set_default_tensor_type(torch.DoubleTensor)

def lstm_bias_y(lstm, input_, hidn_st0, cell_st0, bias_y):
  output, (hidn_st, cell_st) = lstm(input_, (hidn_st0, cell_st0))
  return output + bias_y[None,:], (hidn_st, cell_st)

class RNNLSTM(torch.nn.Module):
  def __init__( self
              , n = 3
              , length = 5
              , num_layers = 1
              , hidden_size = 10
              , num_features = 10
              , output_size = 10):

    super(RNNLSTM,self).__init__()
    self.n = 3
    self.length = 5
    self.num_layers = 1
    self.hidden_size = 10
    self.num_features = 10
    self.output_size = 10 # generally same as hidden_size
    self.hidn_st0 = torch.zeros(self.num_layers, self.n, self.hidden_size)
    self.cell_st0 =torch.zeros(self.num_layers, self.n, self.hidden_size)
    self.input0 = torch.randn(self.length, self.n, self.num_features)

    self.lstm = torch.nn.LSTM(input_size = num_features
                     , hidden_size = hidden_size
                     , num_layers = num_layers
                     , bias = True
                     , batch_first = False
                     , dropout = 0
                     , bidirectional = False
                     , proj_size = 0)

    self.linear = torch.nn.Linear(self.num_features, self.output_size)

  def dump(self, input_):
    d = {}
    d['input']    = self.input0
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
        print(name)
        if self.n == 1:
          if name == 'input':
              futhark_data.dump(p.detach().numpy()[:,0,:],f, True)
          elif name == 'hidn_st0':
              print(p.detach().numpy()[0,0,:])
              futhark_data.dump(p.detach().numpy()[0,0,:],f, True)
          elif name == 'cell_st0':
              futhark_data.dump(p.detach().numpy()[0,0,:],f, True)
          elif name == 'weight':
              futhark_data.dump(p.detach().numpy().T,f, True)
          else:
              futhark_data.dump(p.detach().numpy(),f, True)
        else:
          if name == 'hidn_st0':
              print(p.detach().numpy()[0,:,:].T)
              futhark_data.dump(p.detach().numpy()[0,:,:].T,f, True)
          elif name == 'cell_st0':
              futhark_data.dump(p.detach().numpy()[0,:,:].T,f, True)
          elif name == 'weight':
              futhark_data.dump(p.detach().numpy().T,f, True)
          else:
              futhark_data.dump(p.detach().numpy(),f, True)
    
  def forward(self, input_):
   outputs, _ = self.lstm(input_, (self.hidn_st0, self.cell_st0))
   output = self.linear(torch.cat([t for t in outputs]))
   return output

if __name__ == '__main__':
  rnnlstm = RNNLSTM()
  rnnlstm.dump(rnnlstm.input0)
  print(rnnlstm.forward(rnnlstm.input0))
