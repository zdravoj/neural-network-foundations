import random
from micrograd_clone.engine_clone import Value

class Module:
    """
    Parent Class of Micrograd neural network classes. Created to mimic/match
    the NN module class in PyTorch.

    METHODS
    zero_grad: sets gradient of all parameters to zero
        AN IMPORTANT STEP!!! if gradients are not zeroed before backpropagation,
        the gradients will continue to accumulate rather than display the
        Values' true derivatives of the current iteration, and the neural
        network will operate incorrectly (at a minimum).
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    # parent class has no parameters
        # empty until initialized as an NN class
    def parameters(self):
        return []

class Neuron(Module):
    """
    Atomic class of neural networks. A multi-layer perceptron is made up of
    layers of Neurons.

    PARAMETERS:
        nin: number of inputs in Neuron (size of input vector)
        nonlin: boolean, if output is non-linear
        act: activation function, ReLU is default

    ATTRIBUTES:
        nin: int, number of inputs
        nonlin: boolean, if output is non-linear
        act: string, activation function (ReLU or tanh, ReLU is default)
    
    PUBLIC METHODS:
        parameters: list of weights and biases
    """

    # nin = number of inputs
    def __init__(self, nin, nonlin=True, act='ReLU'):
        # on creation, set random weights between -1 and 1
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # on creation, bias is zero
        self.b = Value(0)
        # use activation function if non-linear
        self.nonlin = nonlin
        # activation function
        self.act = act

    # internal call method
        # ex. n = Neuron(2), x = [2.0, 3.0], n(x)
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        # return sum if linear
        if not self.nonlin:
            return act
        # for non-linear, choose activation function
        elif self.act == 'tanh':
            return act.tanh()
        else:
            return act.relu()

    # weights and biases
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """
    A Layer of Neurons. A multi-layer perceptrion is made up of Layers of
    Neurons.

    PARAMETERS:
        nin: int, number of inputs for Neurons (size of input vector)
        nout: int, number of Neurons in Layer
        **kwargs: additional parameters for Neurons in layer

    ATTRIBUTES:
        neurons: list of Neurons in Layer
    
    PUBLIC METHODS:
        parameters: list of parameters for each Neuron in Layer
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    # internal call method
        # ex. Layer(2, 3), n(x)
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # only returns list if multiple Neurons in Layer
        return out[0] if len(out) == 1 else out

    # list of parameters for each Neuron in Layer
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    A multi-layer perceptron (MLP). A multi-layer perceptrion is made up of Layers of
    Neurons.

    PARAMETERS:
        nin: int, number of inputs for Neurons (size of input vector)
        nouts: list of ints, ordered list of Layer sizes

    ATTRIBUTES:
        sz: list of Layer sizes
        layers: list of Layers
    
    PUBLIC METHODS:
        parameters: list of parameters for each Neuron for each Layer in MLP
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"