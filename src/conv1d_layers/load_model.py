import numpy as np

def load_model(filename, layers):
    data = np.load(filename)
    for i, layer in enumerate(layers):
        if hasattr(layer, "filters"):
            layer.filters = data[f"filters_{i}"]
            layer.bias = data[f"bias_{i}"]
        if hasattr(layer, "W"):
            layer.W = data[f"W_{i}"]
            layer.b = data[f"b_{i}"]
