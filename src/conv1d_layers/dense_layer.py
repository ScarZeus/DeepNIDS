import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, weight_scale=0.01):
        self.W = np.random.randn(input_dim, output_dim) * weight_scale
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx
