from activations.flatter import Flatten
from activations.relu import ReLU
from conv1d_layers.dense_layer import Dense


class DenseBlock:
    def __init__(self, output_dim):
        self.flatten = Flatten()
        self.dense = None
        self.relu = ReLU()
        self.output_dim = output_dim

    def forward(self, x):
        out = self.flatten.forward(x)

        if self.dense is None:
            self.dense = Dense(out.shape[1], self.output_dim)

        out = self.dense.forward(out)
        out = self.relu.forward(out)
        return out

    def backward(self, dout):
        dout = self.relu.backward(dout)
        dout = self.dense.backward(dout)
        dout = self.flatten.backward(dout)
        return dout
