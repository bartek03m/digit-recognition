import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import activations
 
class Input:
    """Input layer, passes data through unchanged"""
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, input):
        return np.asarray(input, dtype=np.float32)
    
    def backward(self, upstream_gradient):
        return np.asarray(upstream_gradient, dtype=np.float32)
    

class Rescaling:
    """Scales input by a constant factor"""
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, input):
        return np.asarray(input, dtype=np.float32) * np.float32(self.scale)

    def backward(self, upstream_gradient):
        return np.asarray(upstream_gradient, dtype=np.float32) * np.float32(self.scale)
    
    
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
            self.patches = None
            self.output_shape = None
    
    def forward(self, input):
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        batch_size, input_height, input_width, input_channels = input.shape
        
        # init weights on first call with He (Kaiming) normal
        if self.weights is None:
            fan_in = self.kernel_size * self.kernel_size * input_channels
            std = np.sqrt(2.0 / fan_in)
            self.weights = np.random.normal(
                0.0,
                std,
                (self.no_of_filters, self.kernel_size, self.kernel_size, input_channels),
            ).astype(np.float32)
            self.biases = np.zeros(self.no_of_filters, dtype=np.float32)

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
        self.patches = patches
        self.output_shape = (output_height, output_width)
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
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        batch_size, input_height, input_width, input_channels = self.input.shape
        # undo activation
        upstream_gradient *= self.activation_func.backward()
        
        output_height, output_width = self.output_shape
        
        # dL/db = sum of gradients over all positions
        self.bias_gradient = np.sum(upstream_gradient, axis=(0, 1, 2))
        
        # reuse cached patches from forward pass
        patches = self.patches
        upstream_flat = upstream_gradient.reshape(batch_size * output_height * output_width, self.no_of_filters)
        
        # dL/dW = gradient^T @ patches
        self.weight_gradient = (upstream_flat.T @ patches).reshape(self.weights.shape)

        # dL/dX - gradient going to the previous layer (vectorized col2im)
        weights_flat = self.weights.reshape(self.no_of_filters, -1)
        grad_cols = (upstream_flat @ weights_flat).reshape(
            batch_size, output_height, output_width, self.kernel_size, self.kernel_size, input_channels
        )

        pad = self.kernel_size // 2
        padded_h = input_height + 2 * pad
        padded_w = input_width + 2 * pad
        downstream_padded = np.zeros((batch_size, padded_h, padded_w, input_channels), dtype=np.float32)

        # coordinate grids for vectorized col2im
        oh_idx = (np.arange(output_height) * self.strides)[:, None, None, None]
        ow_idx = (np.arange(output_width) * self.strides)[None, :, None, None]
        kh_idx = np.arange(self.kernel_size)[None, None, :, None]
        kw_idx = np.arange(self.kernel_size)[None, None, None, :]

        abs_i = oh_idx + kh_idx  # (output_height, 1, kernel_size, 1)
        abs_j = ow_idx + kw_idx  # (1, output_width, 1, kernel_size)

        # broadcasted coordinates across batch and channels
        abs_i = np.broadcast_to(abs_i, (batch_size, output_height, output_width, self.kernel_size, self.kernel_size, input_channels))
        abs_j = np.broadcast_to(abs_j, (batch_size, output_height, output_width, self.kernel_size, self.kernel_size, input_channels))
        n_idx = np.arange(batch_size)[:, None, None, None, None, None]
        c_idx = np.arange(input_channels)[None, None, None, None, None, :]

        np.add.at(downstream_padded, (n_idx, abs_i, abs_j, c_idx), grad_cols)

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
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # split into pool_size x pool_size windows
        
        windows = sliding_window_view(input, (self.pool_size, self.pool_size), axis=(1, 2))
        
        # non-overlapping - step by pool_size
        windows = windows[:, ::self.pool_size, ::self.pool_size, :, :, :]
        
        # pick the max from each window
        output = np.max(windows, axis=(4, 5))
        return output
    
    def backward(self, upstream_gradient):
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        batch_size, input_height, input_width, input_channels = self.input.shape
        downstream_gradient = np.zeros_like(self.input)

        # get the same windows as forward
        windows = sliding_window_view(self.input, (self.pool_size, self.pool_size), axis=(1, 2))
        windows = windows[:, ::self.pool_size, ::self.pool_size, :, :, :]

        # find where the max was in each window
        flat_windows = windows.reshape(*windows.shape[:4], -1)
        max_indices = np.argmax(flat_windows, axis=4)  # (N, out_h, out_w, C)
        _, output_height, output_width, _ = upstream_gradient.shape

        # convert flat index back to 2D row/col offset within the pool window
        max_i = max_indices // self.pool_size  # row offset
        max_j = max_indices % self.pool_size   # col offset

        # build index arrays with broadcasting-friendly shapes
        n_idx = np.arange(batch_size)[:, None, None, None]       # batch axis
        oh_idx = np.arange(output_height)[None, :, None, None]   # output row axis
        ow_idx = np.arange(output_width)[None, None, :, None]    # output col axis
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
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        # squash (H, W, C) into one dimension
        output = input.reshape((input.shape[0], -1))
        return output
    
    def backward(self, upstream_gradient):
        # reshape back to (N, H, W, C)
        return np.asarray(upstream_gradient, dtype=np.float32).reshape(self.input.shape)
    
    
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
        input = np.asarray(input, dtype=np.float32)
        self.input = input
        n_inputs = input.shape[1]
        # init weights on first call with He (Kaiming) normal
        if self.weights is None:
            std = np.sqrt(2.0 / n_inputs)
            self.weights = np.random.normal(0.0, std, (n_inputs, self.neurons)).astype(np.float32)
            self.biases = np.zeros(self.neurons, dtype=np.float32)
        
        # y = x @ W + b
        output = np.dot(input, self.weights) + self.biases
        output = self.activation_func.forward(output)
        return output
    
    def backward(self, upstream_gradient):
        upstream_gradient = np.asarray(upstream_gradient, dtype=np.float32)
        # undo activation
        upstream_gradient *= self.activation_func.backward()
        # dL/dW
        self.weight_gradient = np.dot(self.input.T, upstream_gradient)
        # dL/db
        self.bias_gradient = np.sum(upstream_gradient, axis=0)
        # dL/dX - pass to prev layer
        downstream_gradient = np.dot(upstream_gradient, self.weights.T)
        return downstream_gradient