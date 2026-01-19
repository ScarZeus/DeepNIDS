import numpy as np

class BatchNorm1D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum


        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=(0, 1))     
            var  = np.var(x, axis=(0, 1))     

            self.running_mean = (
                self.momentum * self.running_mean +
                (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var +
                (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var  = self.running_var

        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)

        x_hat = self.x_centered * self.std_inv
        out = self.gamma * x_hat + self.beta

        self.x_hat = x_hat
        return out
