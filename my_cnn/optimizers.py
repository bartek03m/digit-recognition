import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer"""
    def __init__(self):
        self.layers = None
        self.learning_rate = None

    def lazy_load(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        
    def update(self):
        """Update weights and biases for trainable layers"""
        for layer in self.layers:
            # only layers with weights
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                # w = w - lr * dL/dw
                layer.weights -= self.learning_rate * layer.weight_gradient
                layer.biases -= self.learning_rate * layer.bias_gradient
    

def get_optimizer(name):
    """Return an optimizer by name"""
    if name is None:
        return SGD()
    elif name.lower() == 'sgd':
        return SGD()
    else:
        return SGD() 
