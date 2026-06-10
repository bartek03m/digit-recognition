class SGD:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def update(self):
        for layer in self.layers:
            if hasattr(layer, 'weight_gradient') and layer.weight_gradient is not None:
                layer.weights -= self.learning_rate * layer.weight_gradient
                layer.biases -= self.learning_rate * layer.bias_gradient
