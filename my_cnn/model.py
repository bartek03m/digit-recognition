import pickle
import numpy as np
from . import optimizers
from . import losses

class Model:
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
        out = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
             out = self.layers[i].forward(out)
        return out
    
    def back_prop(self, grad):
        for i in range(len(self.layers)-1, -1, -1):
           grad = self.layers[i].backward(grad)
        return grad

    def save(self, filepath="model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model zapisany pomyślnie do pliku: {filepath}")

    @classmethod
    def load(cls, filepath="model.pkl"):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model załadowany pomyślnie z pliku: {filepath}")
        return model
        
    def fit(self, X, y, epochs=3, batch_size=32):
        samples = X.shape[0]

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1} / {epochs}")
            total_loss = 0
            correct_predictions = 0
            for i in range(0, samples, batch_size):
                print(".", end='', flush=True)
                X_batch = X[i : i+batch_size]
                y_batch = y[i : i+batch_size]

                y_pred = self.predict(X_batch)
                
                loss_val = self.loss.forward(y_batch, y_pred)

                total_loss += loss_val * len(X_batch)
                
                preds = np.argmax(y_pred, axis=1)
                trues = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == trues)

                start_grad = self.loss.backward(y_batch, y_pred, len(X_batch))

                self.back_prop(start_grad)

                self.optimizer.update()

                
            print(f"\nLoss: {total_loss/samples:.4f}, Accuracy: {correct_predictions/samples:.4f}")
                
    def evaluate(self, X, y):
        y_pred = self.predict(X)

        loss_val = self.loss.forward(y, y_pred)

        preds = np.argmax(y_pred, axis=1)
        trues = np.argmax(y, axis=1)
        acc = np.sum(preds==trues) / len(y)

        return loss_val, acc        


