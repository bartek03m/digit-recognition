import numpy as np
import layers

fakeimg = np.random.randint(0, 255, (28, 28, 1))  

input_layer = layers.Input((28, 28, 1))
rescaling_layer = layers.Rescaling(1./2)
conv_layer = layers.Convolution2D(5, 3, 1, activation_func='relu')
pool_layer = layers.MaxPooling2D(4)
conv2_layer = layers.Convolution2D(5, 3, 1, activation_func='relu')
pool2_layer = layers.MaxPooling2D(4)
flatten_layer = layers.Flatten()

out1 = input_layer.forward(fakeimg)
out2 = rescaling_layer.forward(out1)
out3 = conv_layer.forward(out2)
out4 = pool_layer.forward(out3)
out5 = conv2_layer.forward(out4)
out6 = pool2_layer.forward(out5)
out7 = flatten_layer.forward(out6)

print(out7)