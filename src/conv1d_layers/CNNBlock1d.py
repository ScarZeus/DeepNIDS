from activations.relu import ReLU
from conv1d_layers.batch_normalization.batch_normalization import BatchNorm1D
from conv1d_layers.conv1d_layer import Conv1dLayer
from conv1d_layers.max_pooling_layer import MaxPool1D


class Conv1DBlock:
    def __init__(self, num_filters, kernel_size, pool_size=2, stride=1):
        self.conv = Conv1dLayer(num_filters, kernel_size)
        self.bn = None          
        self.relu = ReLU()
        self.pool = MaxPool1D(pool_size)

    def forward(self, x, training=True):
        out = self.conv.forward(x)

        
        if self.bn is None:
            self.bn = BatchNorm1D(out.shape[-1])

        out = self.bn.forward(out, training=training)
        out = self.relu.forward(out)
        out = self.pool.forward(out)
        return out

    def backward(self, dout):
        dout = self.pool.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.bn.backward(dout)
        dout = self.conv.backward(dout)
        return dout
