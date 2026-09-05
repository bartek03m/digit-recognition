import numpy as np

class CrossEntropy:
    """Cross-entropy loss for classification"""
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        # -sum(y * log(p)), epsilon to avoid log(0)
        total_loss = -np.sum(y_true * np.log(y_pred + 1e-15))
        batch_size = y_true.shape[0]
        # average loss over batch
        return total_loss / batch_size
    
    def backward(self, y_true, y_pred, batch_size):
        # shortcut gradient for softmax + cross-entropy
        return (y_pred - y_true) / batch_size
    
    
def get_loss(name):
    """Return a loss function by name"""
    if name is None:
        return CrossEntropy()
    elif name.lower() == 'crossentropy':
        return CrossEntropy()
    else:
        return CrossEntropy()
