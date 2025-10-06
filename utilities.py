import numpy as np

def format_image(x, n_classes):
    return (x * (255 // (n_classes - 1))).astype(np.uint8)

# Since we are picking the argmax, this works for both probabilities and log-likelihoods
def label_image_from_probabilities(probs):
    labeled_X = np.argmax(probs, axis=2)
    Y = format_image(labeled_X, probs.shape[-1])
    return Y