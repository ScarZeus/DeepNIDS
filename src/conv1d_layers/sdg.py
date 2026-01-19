class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "filters"):
                layer.filters -= self.lr * layer.dfilters
                layer.bias -= self.lr * layer.dbias
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
