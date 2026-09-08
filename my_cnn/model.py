import pickle
import numpy as np
from . import optimizers
from . import losses

class Model:
    """Sequential neural network model"""
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
        pass

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, learning_rate=0.001):
        self.optimizer = optimizers.get_optimizer(optimizer)
        self.optimizer.lazy_load(self.layers, learning_rate)
        self.loss = losses.get_loss(loss)

    def predict(self, X):
        # run through all layers
        out = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
             out = self.layers[i].forward(out)
        return out
    
    def back_prop(self, grad):
        """Backpropagate gradient through all layers"""
        for i in range(len(self.layers)-1, -1, -1):
           grad = self.layers[i].backward(grad)
        return grad

    def save(self, filepath="model.pkl"):
        # clear cache before saving to keep file size small
        for layer in self.layers:
            if hasattr(layer, 'input'):
                layer.input = None
            if hasattr(layer, 'activation_func') and hasattr(layer.activation_func, 'input'):
                layer.activation_func.input = None
            if hasattr(layer, 'weight_gradient'):
                layer.weight_gradient = None
            if hasattr(layer, 'bias_gradient'):
                layer.bias_gradient = None
            if hasattr(layer, 'v_w'):
                layer.v_w = None
            if hasattr(layer, 'v_b'):
                layer.v_b = None
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath="model.pkl"):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {filepath}")
        return model
        
    def fit(self, X, y, epochs=3, batch_size=32, shuffle=True):
        samples = X.shape[0]
        total_batches = (samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0

            # shuffle indices at the start of each epoch
            if shuffle:
                indices = np.random.permutation(samples)
            else:
                indices = np.arange(samples)

            # go through data in batches
            for batch_idx, i in enumerate(range(0, samples, batch_size), 1):
                batch_indices = indices[i : i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # forward
                y_pred = self.predict(X_batch)
                
                # loss
                loss_val = self.loss.forward(y_batch, y_pred)

                total_loss += loss_val * len(X_batch)
                
                # count correct
                preds = np.argmax(y_pred, axis=1)
                trues = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == trues)

                # backward
                start_grad = self.loss.backward(y_batch, y_pred, len(X_batch))

                self.back_prop(start_grad)

                # update weights
                self.optimizer.update()

                # show progress
                processed = i + len(X_batch)
                curr_loss = total_loss / processed
                curr_acc = correct_predictions / processed
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {curr_loss:.4f} | Acc: {curr_acc:.4f}", end='', flush=True)

            print()
                
    def evaluate(self, X, y):
        y_pred = self.predict(X)

        loss_val = self.loss.forward(y, y_pred)

        preds = np.argmax(y_pred, axis=1)
        trues = np.argmax(y, axis=1)
        acc = np.sum(preds==trues) / len(y)

        return loss_val, acc        


