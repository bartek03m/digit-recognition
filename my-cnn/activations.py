import numpy as np

class ReLU:
    def forward(self, input):
        return np.maximum(0, input)
    
class Softmax:
    def forward(self, input):
        input = input - np.max(input)
        exps = np.exp(input)
        return exps / np.sum(exps)
class Linear:
    def forward(self, input):
        return input
    
def get_activation(name):
    if name is None:
        return Linear()
    if name.lower() == 'relu':
        return ReLU()
    if name.lower() == 'softmax':
        return Softmax()
    return Linear()