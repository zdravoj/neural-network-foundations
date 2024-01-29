import torch

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    # fan in: n_inputs
    # fan out: n_outputs
    # fan_in**0.5 is kaiming initialization
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
    # biases are initialized zero
    self.bias = torch.zeros(fan_out) if bias else None
  

  def __call__(self, x): # W @ x + b
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  # list of parameter tensors
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    # epsilon, for increasing entropy / reducing overfitting
    self.eps = eps
    # for running mean and variance
    self.momentum = momentum
    # for grad tracking in forward / backward pass
    # false reduces overhead when grad tracking unnecessary
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim) # stretch, squeeze
    self.beta = torch.zeros(dim) # shift
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0, 1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else: # predictions / forward pass only
      xmean = self.running_mean # training set mean
      xvar = self.running_var # training set variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    # self.out is used mainly for plotting / network statistics
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad(): # disables computational graph overhead for backward pass
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


# self-explanatory
class Tanh:

  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []


# handles character embeddings in the multidimensional space
# equivalent to matrix C
class Embedding:

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
  
  def __call__(self, IX):
    self.out = self.weight[IX] # indexing operation
    return self.out
  
  def parameters(self):
    return [self.weight]


# concatenates embedding vectors
class FlattenConsecutive:

  def __init__(self, n):
    self.n = n

  def __call__(self, x):
    B, T, C = x.shape # incoming dimensions
    x = x.view(B, T//self.n, C*self.n) # group dimensions
    if x.shape[1] == 1: # if middle dimension is 1, squeeze out
      x = x.squeeze(dim=1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []


# container for NN layers
# when called, runs an input on the layers sequentially
class Sequential:

  def __init__(self, layers):
    self.layers = layers # pass in layers as normal

  def __call__(self, x): # complete forward pass through all layers
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]


class ExampleModel:

  def __init__(self, n_in, n_out, n_hidden, n_layers, block_size):
    self.C = torch.randn((n_out, n_in))
    self.layers = [Linear(n_in * block_size, n_hidden), Tanh()]
    for _ in range(n_layers):
      self.layers.append(Linear(n_hidden, n_hidden))
      self.layers.append(Tanh())
    self.layers.append(n_hidden, n_out)
  
    with torch.no_grad():
      self.layers[-1].gamma *= 0.1
      for layer in self.layers[:-1]:
        if isinstance(layer, Linear):
          layer.weight *= 1.0

    self.parameters = [self.C] + [p for layer in self.layers for p in layer.parameters()]
    self.parameter_count = sum(p.nelement() for p in self.parameters)
    for p in self.parameters:
      p.requires_grad = True
