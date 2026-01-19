import numpy as np  

class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
    
        self.y_true = y_true

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        B = logits.shape[0]
        loss = -np.sum(np.log(self.probs[np.arange(B), y_true])) / B
        return loss

    def backward(self):
        B = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(B), self.y_true] -= 1
        return grad / B
