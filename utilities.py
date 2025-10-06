import numpy as np

def format_image(x, n_classes):
    return (x * (255 // (n_classes - 1))).astype(np.uint8)