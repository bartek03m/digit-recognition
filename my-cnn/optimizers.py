class SGD:
    def __init__(self):
        self.layers = None
        self.learning_rate = None

    def lazy_load(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        
    def update(self):
        for layer in self.layers:
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                layer.weights -= self.learning_rate * layer.weight_gradient
                layer.biases -= self.learning_rate * layer.bias_gradient

def get_optimizer(name):
    if name is None:
        return SGD()
    elif name.lower() == 'sgd':
        return SGD()
    else:
        return SGD() 
