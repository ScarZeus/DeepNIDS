import numpy as np

class Conv1dLayer:
    def __init__(self,num_filters,kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = None
        

    def forward(self,x):
        batch,features,channels = x.shape
        
        if self.filters is None:
            self.filters = np.random.randn(self.filters,self.kernel_size,features) * 0.01


    def backward(self,x):
        pass

        
