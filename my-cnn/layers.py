import numpy as np
import activations
 
class Input:
    def __init__(self, shape, batch_shape=1):
        # Add validation
        self.shape = shape
        self.batch_size = batch_shape
        
    def forward(self, input):
        # Add validation
        return input
    
    def backward(self, upstream_gradient):
        return upstream_gradient
    

class Rescaling:
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, input):
        return input * self.scale

    def backward(self, upstream_gradient):
        return upstream_gradient * self.scale
    
    
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
            self.weight_gradient = None
            self.bias_gradient = None
    
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
        output = np.zeros((batch_size, output_height, output_width, self.no_of_filters))
        
        for out_row, win_row in enumerate(range(0, input_height, self.strides)):
            for out_col, win_col in enumerate(range(0, input_width, self.strides)):
                for batch in range(batch_size):              
                    patch = input[batch, win_row : win_row+self.kernel_size, win_col : win_col+self.kernel_size, :]
                    for k in range(self.no_of_filters):
                        output[batch, out_row, out_col, k] = np.sum(patch * self.weights[k]) + self.biases[k]
        output = self.activation_func.forward(output)
        return output
    
    def backward(self, upstream_gradient):
        batch_size, input_height, input_width, input_channels = self.input.shape
        upstream_gradient *= self.activation_func.backward()

        self.weight_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.sum(upstream_gradient, axis=(0, 1, 2))

        pad = self.kernel_size // 2
        input_padded = np.pad(self.input, ((0,0),(pad,pad),(pad,pad),(0,0)))
        downstream_padded = np.zeros_like(input_padded)
                
        for out_row, win_row in enumerate(range(0, input_height, self.strides)):
            for out_col, win_col in enumerate(range(0, input_width, self.strides)):
                for b in range(batch_size):              
                    patch = input_padded[b, win_row : win_row+self.kernel_size, win_col : win_col+self.kernel_size, :]
                    for k in range(self.no_of_filters):
                        d = upstream_gradient[b, out_row, out_col, k]
                        self.weight_gradient[k] += d * patch
                        downstream_padded[b, win_row : win_row+self.kernel_size, win_col : win_col+self.kernel_size, :] += d * self.weights[k]
        if pad > 0:
            return downstream_padded[:, pad : -pad, pad : -pad, :]
        return downstream_padded


class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.input = None
    
    def forward(self, input):
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        
        output_height = len(range(0, input_height, self.pool_size))
        output_width = len(range(0, input_width, self.pool_size))
        output = np.zeros((batch_size, output_height, output_width, input_channels))
        
        for out_row, win_row in enumerate(range(0, input_height, self.pool_size)):
            for out_col, win_col in enumerate(range(0, input_width, self.pool_size)):
                for b in range(batch_size):              
                    pool = input[b, win_row : win_row+self.pool_size, win_col : win_col+self.pool_size, :]
                    output[b, out_row, out_col] = np.max(pool, axis=(0, 1))
                
        return output
    
    def backward(self, upstream_gradient):
        batch_size, input_height, input_width, input_channels = self.input.shape
        downstream_gradient = np.zeros(self.input.shape)
        
        for out_row, win_row in enumerate(range(0, input_height, self.pool_size)):
            for out_col, win_col in enumerate(range(0, input_width, self.pool_size)):
                for b in range(batch_size):              
                    pool = self.input[b, win_row : win_row+self.pool_size, win_col : win_col+self.pool_size, :]
                    for c in range(input_channels):
                        flat_idx = np.argmax(pool[:, :, c])
                        local_i, local_j = np.unravel_index(flat_idx, (self.pool_size, self.pool_size))
                        downstream_gradient[b, win_row + local_i, win_col + local_j, c] = upstream_gradient[b, out_row, out_col, c]
                
        return downstream_gradient

                
class Flatten:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        output = input.reshape((input.shape[0], -1))
        return output
    
    def backward(self, upstream_gradient):
        return upstream_gradient.reshape(self.input.shape)
    
    
class Dense:
    def __init__(self, neurons, activation_func=None):
        self.neurons = neurons
        self.activation_func = activations.get_activation(activation_func)
        self.weights = None
        self.biases = None
        self.input = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        self.input = input
        n_inputs = input.shape[1]
        if self.weights is None:
            self.weights = np.random.randn(n_inputs, self.neurons) * 0.1
            self.biases = np.zeros(self.neurons)
        
        output = np.dot(input, self.weights) + self.biases
        output = self.activation_func.forward(output)
        return output
    
    def backward(self, upstream_gradient):
        upstream_gradient *= self.activation_func.backward()
        self.weight_gradient = np.dot(self.input.T, upstream_gradient)
        self.bias_gradient = np.sum(upstream_gradient, axis=0)
        downstream_gradient = np.dot(upstream_gradient, self.weights.T)
        return downstream_gradient