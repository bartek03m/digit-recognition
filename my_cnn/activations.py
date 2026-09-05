import numpy as np

class ReLU:
    """ReLU activation"""
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        # kill negative values
        return np.maximum(0, input)
        
    def backward(self):
        # 1 where positive, 0 otherwise
        return (self.input > 0).astype(float)
    
    
class Softmax:
    """Softmax activation"""
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        # subtract max to prevent overflow
        max_vals = np.max(input, axis=1, keepdims=True)
        exps = np.exp(input - max_vals)
        # normalize so it sums to 1
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def backward(self):
        return 1
    
    
class Linear:
    """No activation (identity)"""
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
         
    def backward(self):
        return 1
    
    
def get_activation(name):
    """Return an activation function by name"""
    if name is None:
        return Linear()
    if name.lower() == 'relu':
        return ReLU()
    if name.lower() == 'softmax':
        return Softmax()
    return Linear()