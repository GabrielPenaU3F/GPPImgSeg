import numpy as np

shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def format_image(x, n_classes):
    return (x * (255 // (n_classes - 1))).astype(np.uint8)

# Since we are picking the argmax, this works for both probabilities and log-likelihoods
def label_image_from_probabilities(probs):
    labeled_X = np.argmax(probs, axis=2)
    Y = format_image(labeled_X, probs.shape[-1])
    return Y

'''
    This extracts the probabilities of the neighbors. Each row in the stack corresponds to a neighbor
    in the order defined by shifts
'''
def get_neighbor_stack(pixel_representations, neighborhood):
    # probs: (h, w, n_classes)
    h, w, k = pixel_representations.shape
    padded = np.pad(pixel_representations, ((1,1),(1,1),(0,0)), mode='reflect')  # (h+2, w+2, k)
    shifts = make_shifts_from_mode(mode=neighborhood.get_type(), radius=neighborhood.get_radius(),
                                   k=neighborhood.get_k(), custom_shifts=neighborhood.get_custom_shifts())
    neighbor_list = [padded[1+dy:1+dy+h, 1+dx:1+dx+w, :] for dy,dx in shifts]
    # stack -> (h, w, neighbors, k)
    neighbor_stack = np.stack(neighbor_list, axis=2)
    return neighbor_stack

def make_shifts_from_mode(mode='8', radius=None, k=None, custom_shifts=None):
    """
    Returns a list of offsets (dy,dx) according to the chosen mode.
    - mode: '4', '8', 'radius', 'k', 'custom'
    - radius: only if mode == 'radius' (int)
    - k: only if mode == 'k' (int) -> k neares neighbors under euclidean metric
    """

    # Classic modes
    if mode == '4':
        return [(-1,0),(0,-1),(0,1),(1,0)]
    if mode == '8':
        return [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # construct grid of offsets up to some radius_max
    # choose a reasonable radius_max if not given
    if mode in ('radius', 'k'):
        if radius is None:
            raise ValueError("Radius required for mode='radius'")
        # generate offsets in a square [-radius, radius] except center
        offsets = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dy == 0 and dx == 0:
                    continue
                offsets.append((dy, dx))

        # filter by circular radius if mode == 'radius'
        offsets = np.array(offsets)
        if mode == 'radius':
            dquad = np.sum(offsets**2, axis=1)
            offsets = offsets[dquad <= radius**2]
            return offsets
        # mode == 'k': sort by distance and take k nearest
        if mode == 'k':
            if k is None:
                raise ValueError("k must be provided for mode='k'")
            offsets = sorted(offsets, key=lambda o: (o[0]**2 + o[1]**2, abs(o[0]), abs(o[1])))
            return offsets[:k]

    raise ValueError(f"Unknown mode={mode}")