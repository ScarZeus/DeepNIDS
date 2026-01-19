import numpy as np

class MaxPool1D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.x = x
        B, T, C = x.shape
        out_time = T // self.pool_size

        out = np.zeros((B, out_time, C))
        self.max_idx = np.zeros_like(out, dtype=int)

        for b in range(B):
            for c in range(C):
                for t in range(out_time):
                    start = t * self.pool_size
                    end = start + self.pool_size

                    window = x[b, start:end, c]
                    out[b, t, c] = np.max(window)
                    self.max_idx[b, t, c] = np.argmax(window)

        return out

    def backward(self, dout):
        B, T, C = self.x.shape
        dx = np.zeros_like(self.x)
        out_time = dout.shape[1]

        for b in range(B):
            for c in range(C):
                for t in range(out_time):
                    start = t * self.pool_size
                    idx = self.max_idx[b, t, c]
                    dx[b, start + idx, c] += dout[b, t, c]

        return dx
