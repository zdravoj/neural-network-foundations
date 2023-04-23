from math import exp

class Value:
    """
    Stores a single scalar value and its gradient.
    
    The Value Class is the atomic class of Micrograd. Instances of it act as
    scalar values such as the input values, weights, and biases of the model,
    but also hold essential information about those scalars; particularly,
    the Value also holds:
        1. child Values that produced it,
        2. the mathematical operation used on the child Values to create the 
           Value, and
        3. the gradient (chain-ruled derivative from next Values).
    All of these attributes are crucial in backpropagation for updating the
    weights and biases of the Neurons in a neural network.

    ATTRIBUTES
    data: the scalar value
    grad: the gradient
    _backward: the backpropagation function
    _prev: child Values
    _op: mathematical operation

    PUBLIC METHODS
    tanh: tanh activation
    relu: ReLU activation
    backward: backpropagates through all child Values that interacted to
        create the final Value

    PRIMARY PRIVATE METHODS
    __add__: adds two Values, or a Value and an int/float
    __mul__: multiplies two Values, or a Value and an int/float
    __pow__: exponentiates Value to a Value, or a Value to an int/float

    SECONDARY PRIVATE METHODS
    __neg__: negates Value
    __sub__: subtracts two Values (self - other)
    __truediv__: divides two Values

    TERTIARY PRIVATE METHODS
        The tertiary private methods exist solely for the purpose of Value
        instances interacting cleanly with ints/floats. In the primary and
        secondary private methods, (Value + int) is the same as 
        Value.__add__(int), but breaks down if (int + Value) because Python
        interprets this as int.__add__(Value). Python recognizes the tertiary
        private method __radd__ such that (int + Value) is computed as
        int.__radd__(Value) and returns a Value.
    __radd__
    __rsub__
    __rmul__
    __rtruediv__
    """

    # initializes instance of class
    def __init__(self, data, _children=(), _op=''):
        self.data = data # scalar or value
        self.grad = 0 # gradient
        self._backward = lambda: None # backpropagation function for this node,
            # initially set to none because leaf nodes have no backpropagation
        self._prev = set(_children) # child nodes that produced this node
        self._op = _op # the op that produced this node (for graphs/debugging)

    # adds two Values, or a Value and an int/float
    def __add__(self, other):
        # converts int/float to Value
        other = other if isinstance(other, Value) else Value(other)
        # new Value node (addition)
        out = Value(self.data + other.data, (self, other), '+')

        # backpropagation function for addition 
            # dy/dx for addition is Constant, 1.0
        def _backward():
            # += so grad accumulates if self and other are same node
            self.grad += out.grad
            other.grad += out.grad
        # so out's _op (forward pass) and _backward (backpropagation)
            # are in agreement
        out._backward = _backward

        return out

    # multiplies two Values, or a Value and an int/float
    def __mul__(self, other):
        # converts int/float to Value
        other = other if isinstance(other, Value) else Value(other)
        # new Value node (multiplication)
        out = Value(self.data * other.data, (self, other), '*')

        # backpropagation function for multiplication
            # dy/dx for multiplication is n
        def _backward():
            # += so grad accumulates if self and other are same node
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        # so out's _op (forward pass) and _backward (backpropagation)
            # are in agreement
        out._backward = _backward

        return out

    # exponentiates Value to a Value, or a Value to an int/float
    def __pow__(self, other):
        # raises error if other is not int/float
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # new Value node (exponentiation)
        out = Value(self.data**other, (self,), f'**{other}')

        # backpropagation function for exponentiation
            # dy/dx for exponentiation is nx**(n-1)
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        # so out's _op (forward pass) and _backward (backpropagation)
            # are in agreement
        out._backward = _backward

        return out

    # tanh activation function
    def tanh(self):
        # define value
        x = self.data
        # apply tanh to value
        t = (exp(2*x) - 1)/(exp(2*x) + 1)
        # new Value node (tanh)
        out = Value(t, (self, ), 'tanh')

        # backpropagation function for tanh
            # dy/dx for tanh is 1 - tanh(x)**2
        def _backward():
            self.grad = (1 - t**2) * out.grad
        # so out's _op (forward pass) and _backward (backpropagation)
            # are in agreement
        out._backward = _backward
        
        return out

    # ReLU activation function
    def relu(self):
        # apply ReLU to Value
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # backpropagation function for ReLU
            # dy/dx for ReLU is binary, 1 if above threshold
        def _backward():
            self.grad += (out.data > 0) * out.grad
        # so out's _op (forward pass) and _backward (backpropagation)
            # are in agreement
        out._backward = _backward

        return out

    # backpropagates through all Value children that interacted to
        # create the final Value
    def backward(self):

        # topological order all of the children in the graph
            # orders all nodes in the graph such that all pointers (child node
            # points to created node) are pointing to node at a larger index
            # in the array
        # ordered list of nodes
        topo = []
        # distinct nodes visited
        visited = set()
        # recursive, per Value in graph
        def build_topo(val):
            # skips if Value already visited
            if val not in visited:
                visited.add(val)
                # visit children
                for child in val._prev:
                    build_topo(child)
                # add Value to list after children
                topo.append(val)
        # call on output Value
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # set loss gradient 
        self.grad = 1
        # backpropagation from output Value (loss) to first Value in network
        for val in reversed(topo):
            val._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self): # how Value is represented when printed
        return f"Value(data={self.data}, grad={self.grad})"