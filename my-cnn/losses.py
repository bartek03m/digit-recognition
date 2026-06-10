import numpy as np

class CrossEntropy:
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        total_loss = -np.sum(y_true * np.log(y_pred + 1e-15))
        batch_size = y_true.shape[0]
        return total_loss / batch_size