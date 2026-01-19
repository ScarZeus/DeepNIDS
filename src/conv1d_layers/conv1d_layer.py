import numpy as np

class Conv1dLayer:
    def __init__(self, num_filters, kernel_size, weight_scale=0.01):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.weight_scale = weight_scale

        self.filters = None
        self.bias = None

        self.dfilters = None
        self.dbias = None

    def forward(self, x):
        
        self.x = x
        batch, time, channels = x.shape

        if self.filters is None:
            self.filters = np.random.randn(
                self.num_filters,
                self.kernel_size,
                channels
            ) * self.weight_scale
            self.bias = np.zeros(self.num_filters)

        output_time = time - self.kernel_size + 1
        out = np.zeros((batch, output_time, self.num_filters))

        for b in range(batch):
            for f in range(self.num_filters):
                for t in range(output_time):
                    window = x[b, t:t + self.kernel_size, :]
                    out[b, t, f] = (
                        np.sum(window * self.filters[f]) + self.bias[f]
                    )

        return out

    def backward(self, dout):
        
        x = self.x
        batch, time, channels = x.shape
        _, output_time, _ = dout.shape

        dx = np.zeros_like(x)
        dfilters = np.zeros_like(self.filters)
        dbias = np.zeros_like(self.bias)

        for b in range(batch):
            for f in range(self.num_filters):
                for t in range(output_time):
                    window = x[b, t:t + self.kernel_size, :]

                    dfilters[f] += dout[b, t, f] * window

                    dx[b, t:t + self.kernel_size, :] += (
                        dout[b, t, f] * self.filters[f]
                    )

                    dbias[f] += dout[b, t, f]

        self.dfilters = dfilters
        self.dbias = dbias
        return dx
