import numpy as np

class CrossEntropy:
    """Cross-entropy loss for classification"""
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        # -sum(y * log(p)), epsilon to avoid log(0)
        total_loss = -np.sum(y_true * np.log(y_pred + np.float32(1e-15)))
        batch_size = y_true.shape[0]
        # average loss over batch
        return np.float32(total_loss / batch_size)

    def backward(self, y_true, y_pred, batch_size):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        # shortcut gradient for softmax + cross-entropy
        return (y_pred - y_true) / np.float32(batch_size)
    
    
def get_loss(name):
    """Return a loss function by name"""
    if name is None:
        return CrossEntropy()
    elif name.lower() == 'crossentropy':
        return CrossEntropy()
    else:
        return CrossEntropy()
