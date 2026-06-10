import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
        
    def backward(self):
        return (self.input > 0).astype(float)
    
    
class Softmax:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        max_vals = np.max(input, axis=1, keepdims=True)
        exps = np.exp(input - max_vals)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def backward(self):
        return 1
    
    
class Linear:
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
         
    def backward(self):
        return 1
    
    
def get_activation(name):
    if name is None:
        return Linear()
    if name.lower() == 'relu':
        return ReLU()
    if name.lower() == 'softmax':
        return Softmax()
    return Linear()