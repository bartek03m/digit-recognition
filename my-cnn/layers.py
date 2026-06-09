import numpy as np
import activations

# input format (B, H, W, C)  
# zapamietaj inputy i outputy wszedzie
# przeczytaj o funkcji straty loss i jak sie ją liczy 
# backward dla kazdej z klas
  
class Input:
    def __init__(self, shape, batch_shape=1):
        # Add validation
        self.shape = shape
        self.batch_size = batch_shape
        
    def forward(self, input):
        # Add validation
        return input
    

class Rescaling:
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, input):
        return input * self.scale
    
    
class Convolution2D:
    def __init__(
            self,
            no_of_filters: int,
            kernel_size: int,
            strides=1,
            activation_func=None,
        ):
            self.no_of_filters = no_of_filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation_func = activations.get_activation(activation_func)
            self.weights = None
            self.biases = None
            self.input = None
            self.output = None
            self.output_activated = None
    
    def forward(self, input):
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        
        if self.weights is None:
            self.weights = np.random.normal(0, 1, (self.no_of_filters, self.kernel_size, self.kernel_size, input_channels)) * 0.1
            self.biases = np.random.normal(0, 1, self.no_of_filters)
        pad = self.kernel_size // 2
        input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        
        output_height = len(range(0, input_height, self.strides))
        output_width = len(range(0, input_width, self.strides))
        self.output = np.zeros((batch_size, output_height, output_width, self.no_of_filters))
        
        for out_i, img_i in enumerate(range(0, input_height, self.strides)):
            for out_j, img_j in enumerate(range(0, input_width, self.strides)):
                for batch in range(batch_size):              
                    patch = input[batch, img_i : img_i+self.kernel_size, img_j : img_j+self.kernel_size, :]
                    for k in range(self.no_of_filters):
                        self.output[batch, out_i, out_j, k] = np.sum(patch * self.weights[k]) + self.biases[k]
        self.output_activated = self.activation_func.forward(self.output)
        return self.output_activated


class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.input = None
        self.output = None
    
    def forward(self, input):
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        
        output_height = len(range(0, input_height, self.pool_size))
        output_width = len(range(0, input_width, self.pool_size))
        self.output = np.zeros((batch_size, output_height, output_width, input_channels))
        
        for out_i, img_i in enumerate(range(0, input_height, self.pool_size)):
            for out_j, img_j in enumerate(range(0, input_width, self.pool_size)):
                for b in range(batch_size):              
                    pool = input[b, img_i : img_i+self.pool_size, img_j : img_j+self.pool_size, :]
                    self.output[b, out_i, out_j] = np.max(pool, axis=(0, 1))
                
        return self.output

                
class Flatten:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        self.input = input
        self.output = input.reshape((input.shape[0], -1))
        return self.output
    
    
class Dense:
    def __init__(self, neurons, activation_func=None):
        self.neurons = neurons
        self.activation_func = activations.get_activation(activation_func)
        self.weights = None
        self.biases = None
        self.input = None
        self.output = None
        self.output_activated = None

    def forward(self, input):
        self.input = input
        n_inputs = input.shape[1]
        if self.weights is None:
            self.weights = np.random.randn(n_inputs, self.neurons) * 0.1
            self.biases = np.zeros(self.neurons)
        
        self.output = np.dot(input, self.weights) + self.biases
        self.output_activated = self.activation_func.forward(self.activation_func.forward(self.output))
        return self.output_activated
    


