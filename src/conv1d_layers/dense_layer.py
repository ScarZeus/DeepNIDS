import numpy as np

class DenseLayer:
    def __init__(self,input_dim,output_dim,activation=None):
        self.W = np.random.randn(input_dim,output_dim) * 0.01
        self.b = np.zeros((1,output_dim))   
        self.activation = activation

        def forward(self,x):
            pass

        def backward(self,x):
            pass