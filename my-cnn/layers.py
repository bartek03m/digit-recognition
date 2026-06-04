import numpy as np
import activations

class Input:
    def __init__(self, shape: tuple):
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        self.shape = shape
        
    def forward(self, input):
        if input.shape != self.shape:
            raise ValueError("Incorret shapes")
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
    
    def forward(self, input):
        input_height, input_width, input_channels = input.shape
        
        if self.weights is None:
            self.weights = np.random.normal(0, 1, (self.no_of_filters, self.kernel_size, self.kernel_size, input_channels))
            self.biases = np.random.normal(0, 1, self.no_of_filters)
        pad = self.kernel_size // 2
        input = np.pad(input, ((pad, pad), (pad, pad), (0, 0)))
        
        output_height = len(range(0, input_height, self.strides))
        output_width = len(range(0, input_width, self.strides))
        output = np.zeros((output_height, output_width, self.no_of_filters))
        
        for out_i, img_i in enumerate(range(0, input_height, self.strides)):
            for out_j, img_j in enumerate(range(0, input_width, self.strides)):              
                patch = input[img_i : img_i+self.kernel_size, img_j : img_j+self.kernel_size, :]
                for k in range(self.no_of_filters):
                    output[out_i, out_j, k] = np.sum(patch * self.weights[k]) + self.biases[k]
          
        return self.activation_func.forward(output)


class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, input):
        input_height, input_width, input_channels = input.shape
        
        output_height = len(range(0, input_height, self.pool_size))
        output_width = len(range(0, input_width, self.pool_size))
        output = np.zeros((output_height, output_width, input_channels))
        
        for out_i, img_i in enumerate(range(0, input_height, self.pool_size)):
            for out_j, img_j in enumerate(range(0, input_width, self.pool_size)):              
                pool = input[img_i : img_i+self.pool_size, img_j : img_j+self.pool_size, :]
                output[out_i, out_j] = np.max(pool, axis=(0, 1))
                
        return output

                
class Flatten:
    def forward(self, input):
        return input.flatten()
    


