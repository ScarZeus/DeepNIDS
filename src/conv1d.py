from conv1d_layers.CNNBlock1d import Conv1DBlock
from conv1d_layers.dense1d_block import DenseBlock


class CNN1D:
    def __init__(self, num_classes):
        self.block1 = Conv1DBlock(num_filters=32, kernel_size=3)
        self.block2 = Conv1DBlock(num_filters=64, kernel_size=5)
        self.fc = DenseBlock(output_dim=num_classes)

    def forward(self, x, training=True):
        out = self.block1.forward(x, training)
        out = self.block2.forward(out, training)
        out = self.fc.forward(out)
        return out

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.block1.backward(dout)
        return dout
