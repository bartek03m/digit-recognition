import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist 
import model
import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = np.eye(10)[y_train]
y_test= np.eye(10)[y_test]

test_model = model.Model()

test_model.add(layers.Input((28, 28, 1)))
test_model.add(layers.Rescaling(1./255))
test_model.add(layers.Convolution2D(4, 3, activation_func='relu'))
test_model.add(layers.MaxPooling2D(2))
test_model.add(layers.Convolution2D(8, 3, activation_func='relu'))
test_model.add(layers.MaxPooling2D(2))
test_model.add(layers.Flatten())
test_model.add(layers.Dense(32, activation_func='relu'))
test_model.add(layers.Dense(10, activation_func='softmax'))

test_model.compile(optimizer='sgd', loss='crossentropy', learning_rate=0.05)

test_model.fit(x_train, y_train, epochs=2, batch_size=32)