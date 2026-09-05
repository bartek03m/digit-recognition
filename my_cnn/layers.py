import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import activations
 
class Input:
    """Input layer, passes data through unchanged"""
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, input):
        return input
    
    def backward(self, upstream_gradient):
        return upstream_gradient
    

class Rescaling:
    """Scales input by a constant factor"""
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, input):
        # multiply each pixel value by the scale factor
        return input * self.scale

    def backward(self, upstream_gradient):
        # chain rule - gradient flows through scaled by the same factor
        return upstream_gradient * self.scale
    
    
class Convolution2D:
    """2D convolution layer with same-padding"""
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
        
        # init weights on first call
        if self.weights is None:
            # shape: (num_filters, kH, kW, input_channels)
            self.weights = np.random.normal(0, 1, (self.no_of_filters, self.kernel_size, self.kernel_size, input_channels)) * 0.1
            self.biases = np.random.normal(0, 1, self.no_of_filters)

        # zero-pad edges so output keeps the same size
        pad = self.kernel_size // 2
        input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        
        # extract all kernel-sized patches
        windows = sliding_window_view(input, (self.kernel_size, self.kernel_size), axis=(1,2))
        
        # skip windows according to stride
        windows = windows[:, ::self.strides, ::self.strides, :, :, :]
        
        # rearrange to [N, H_out, W_out, kH, kW, C]
        windows = windows.transpose(0, 1, 2, 4, 5, 3)
        
        output_height, output_width = windows.shape[1], windows.shape[2]
        
        # flatten patches into rows: (N*H_out*W_out, kH*kW*C)
        patches = windows.reshape(batch_size * output_height * output_width, -1)
        # flatten filters into rows: (num_filters, kH*kW*C)
        weights_flat = self.weights.reshape(self.no_of_filters, -1)
        
        # big matrix multiply
        output = patches @ weights_flat.T + self.biases
        
        # put it back into image shape
        output = output.reshape(batch_size, output_height, output_width, self.no_of_filters)
        
        # run through activation
        output = self.activation_func.forward(output)
        return output
    
    def backward(self, upstream_gradient):
        batch_size, input_height, input_width, input_channels = self.input.shape
        # undo activation
        upstream_gradient *= self.activation_func.backward()
        
        # pad again like in forward
        pad = self.kernel_size // 2
        input = np.pad(self.input, ((0,0),(pad,pad),(pad,pad),(0,0)))
        
        # same window extraction as forward
        windows = sliding_window_view(input, (self.kernel_size, self.kernel_size), axis=(1,2))
        windows = windows [:, ::self.strides, ::self.strides, :, :, :]
        windows = windows.transpose(0, 1, 2, 4, 5, 3)

        _, output_height, output_width, _ = upstream_gradient.shape
        
        # dL/db = sum of gradients over all positions
        self.bias_gradient = np.sum(upstream_gradient, axis=(0, 1, 2))
        
        # flatten for matrix math
        patches = windows.reshape(batch_size * output_height * output_width, -1)
        upstream_flat = upstream_gradient.reshape(batch_size * output_height * output_width, self.no_of_filters)
        
        # dL/dW = gradient^T @ patches
        self.weight_gradient = (upstream_flat.T @ patches).reshape(self.weights.shape)

        # dL/dX - gradient going to the previous layer (col2im)
        weights_flat = self.weights.reshape(self.no_of_filters, -1)
        grad_cols = (upstream_flat @ weights_flat).reshape(
            batch_size, output_height, output_width, self.kernel_size, self.kernel_size, input_channels
        )

        # put patch gradients back into their original positions
        downstream_padded = np.zeros_like(input)
        for oh in range(output_height):
            for ow in range(output_width):
                r = oh * self.strides
                c = ow * self.strides
                # overlapping patches just add up
                downstream_padded[:, r:r+self.kernel_size, c:c+self.kernel_size, :] += grad_cols[:, oh, ow, :, :, :]

        # strip padding
        if pad > 0:
            return downstream_padded[:, pad : -pad, pad : -pad, :]
        return downstream_padded


class MaxPooling2D:
    """2D max pooling layer"""
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.input = None
    
    def forward(self, input):
        self.input = input
        # split into pool_size x pool_size windows
        windows = sliding_window_view(input, (self.pool_size, self.pool_size), axis=(1, 2))
        # non-overlapping - step by pool_size
        windows = windows[:, ::self.pool_size, ::self.pool_size, :, :, :]
        # pick the max from each window
        output = np.max(windows, axis=(4, 5))
        return output
    
    def backward(self, upstream_gradient):
        batch_size, input_height, input_width, input_channels = self.input.shape
        downstream_gradient = np.zeros(self.input.shape)

        # get the same windows as forward
        windows = sliding_window_view(self.input, (self.pool_size, self.pool_size), axis=(1, 2))
        windows = windows[:, ::self.pool_size, ::self.pool_size, :, :, :]

        # find where the max was in each window
        flat_windows = windows.reshape(*windows.shape[:4], -1)
        max_indices = np.argmax(flat_windows, axis=4)  # (N, out_h, out_w, C)
        _, output_height, output_widht, _ = upstream_gradient.shape

        # convert flat index back to 2D row/col offset within the pool window
        max_i = max_indices // self.pool_size  # row offset
        max_j = max_indices % self.pool_size   # col offset

        # build index arrays with broadcasting-friendly shapes
        n_idx = np.arange(batch_size)[:, None, None, None]       # batch axis
        oh_idx = np.arange(output_height)[None, :, None, None]   # output row axis
        ow_idx = np.arange(output_widht)[None, None, :, None]    # output col axis
        c_idx = np.arange(input_channels)[None, None, None, :]   # channel axis

        # actual position in the full input
        abs_i = oh_idx * self.pool_size + max_i
        abs_j = ow_idx * self.pool_size + max_j

        # only the max element gets the gradient
        np.add.at(downstream_gradient, (n_idx, abs_i, abs_j, c_idx), upstream_gradient)
        return downstream_gradient

                
class Flatten:
    """Flattens spatial dimensions into a 1D vector"""
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        # squash (H, W, C) into one dimension
        output = input.reshape((input.shape[0], -1))
        return output
    
    def backward(self, upstream_gradient):
        # reshape back to (N, H, W, C)
        return upstream_gradient.reshape(self.input.shape)
    
    
class Dense:
    """Fully connected layer"""
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
        # init weights on first call
        if self.weights is None:
            self.weights = np.random.randn(n_inputs, self.neurons) * 0.1
            self.biases = np.zeros(self.neurons)
        
        # y = x @ W + b
        output = np.dot(input, self.weights) + self.biases
        output = self.activation_func.forward(output)
        return output
    
    def backward(self, upstream_gradient):
        # undo activation
        upstream_gradient *= self.activation_func.backward()
        # dL/dW
        self.weight_gradient = np.dot(self.input.T, upstream_gradient)
        # dL/db
        self.bias_gradient = np.sum(upstream_gradient, axis=0)
        # dL/dX - pass to prev layer
        downstream_gradient = np.dot(upstream_gradient, self.weights.T)
        return downstream_gradient