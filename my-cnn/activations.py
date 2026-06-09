import numpy as np

class ReLU:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output
    
    
class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        max_vals = np.max(input, axis=1, keepdims=True)
        exps = np.exp(input - max_vals)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
    
    
class Linear:
    def __init__(self):
        self.input = None
        self.output = None 

    def forward(self, input):
        self.input = input
        self.output = input
        return self.output
    
    
def get_activation(name):
    if name is None:
        return Linear()
    if name.lower() == 'relu':
        return ReLU()
    if name.lower() == 'softmax':
        return Softmax()
    return Linear()