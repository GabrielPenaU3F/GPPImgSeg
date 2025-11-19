import numpy as np
from scipy.optimize._lsap import linear_sum_assignment

'''
    Input: image in any scale
    Output: image in range 0-255
'''
def format_regular_image(img):
    img = img.astype(np.float32)
    if img.ndim == 3:  # color image
        for c in range(img.shape[2]):
            channel = img[..., c]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            img[..., c] = 255 * channel
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = 255 * img
    return np.clip(img, 0, 255)

'''
    Input: image with intensity values in 0, 1, ..., n_classes 
'''
def format_labeled_image(x, n_classes):
    return (x * (255 // (n_classes - 1))).astype(np.uint8)

# Since we are picking the argmax, this works for both probabilities and log-likelihoods
def label_image_from_probabilities(probs):
    labeled_X = np.argmax(probs, axis=2)
    Y = format_labeled_image(labeled_X, probs.shape[-1])
    return Y

def align_labels(pred, gt, return_type='img'):
    n_classes = len(np.unique(pred))
    # If there is only one class, it cannot be further aligned
    if n_classes == 1:
        return pred

    pred = normalize_labels(pred)
    gt = normalize_labels(gt)
    pred_flat = pred.ravel()
    true_flat = gt.ravel()
    n_pred = pred.max() + 1
    n_true = gt.max() + 1

    # Confusion matrix
    cost_matrix = np.zeros((n_pred, n_true), dtype=int)
    for i in range(n_pred):
        for j in range(n_true):
            cost_matrix[i, j] = np.sum((pred_flat == i) & (true_flat == j))

    # Asignación óptima
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)

    # Crear mapa de reasignación
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Aplicar la reasignación
    pred_aligned = np.vectorize(lambda x: mapping.get(x, x))(pred)

    # Format and output
    if return_type == 'img':
        pred_aligned = format_labeled_image(pred_aligned, n_classes)
    elif return_type == 'labels':
        pass

    return pred_aligned

def normalize_labels(x):
    """Convierte intensidades arbitrarias (p.ej. 0, 90, 180)
       en etiquetas consecutivas (0, 1, 2)."""
    x = x.astype(int)
    unique_vals, normalized = np.unique(x, return_inverse=True)
    return normalized.reshape(x.shape)