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
    

class Momentum:
    """SGD with Momentum optimizer"""
    def __init__(self, beta=0.9):
        self.layers = None
        self.learning_rate = None
        self.beta = beta

    def lazy_load(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def update(self):
        """Update weights and biases with momentum"""
        for layer in self.layers:
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                if not hasattr(layer, 'v_w') or layer.v_w is None:
                    layer.v_w = np.zeros_like(layer.weights)
                    layer.v_b = np.zeros_like(layer.biases)

                # v = beta * v + grad
                layer.v_w = self.beta * layer.v_w + layer.weight_gradient
                layer.v_b = self.beta * layer.v_b + layer.bias_gradient

                # w = w - lr * v
                layer.weights -= self.learning_rate * layer.v_w
                layer.biases -= self.learning_rate * layer.v_b


def get_optimizer(name):
    """Return an optimizer by name"""
    if name is None:
        return SGD()
    name_str = str(name).lower()
    if name_str == 'sgd':
        return SGD()
    elif name_str in ('momentum', 'sgd_momentum'):
        return Momentum()
    else:
        return SGD() 
