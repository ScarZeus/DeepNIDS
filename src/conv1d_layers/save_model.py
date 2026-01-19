import numpy as np

def save_model(filename, layers):
    params = {}
    for i, layer in enumerate(layers):
        if hasattr(layer, "filters"):
            params[f"filters_{i}"] = layer.filters
            params[f"bias_{i}"] = layer.bias
        if hasattr(layer, "W"):
            params[f"W_{i}"] = layer.W
            params[f"b_{i}"] = layer.b
    np.savez(filename, **params)
