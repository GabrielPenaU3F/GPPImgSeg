import numpy as np

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
    shifts = make_shifts_from_mode(mode=neighborhood.get_type(), radius=neighborhood.get_radius(),
                                   k=neighborhood.get_k(), distance_fn=neighborhood.get_distance_fn())

    max_dy = max(abs(dy) for dy, dx in shifts) if len(shifts) > 0 else 0
    max_dx = max(abs(dx) for dy, dx in shifts) if len(shifts) > 0 else 0
    padded = np.pad(pixel_representations, ((max_dy, max_dy), (max_dx, max_dx), (0, 0)), mode='reflect')

    neighbor_list = []
    for dy, dx in shifts:
        # slice desplazado: start en (pad_top + dy, pad_left + dx)
        start_y = max_dy + dy
        start_x = max_dx + dx
        neighbor = padded[start_y:start_y + h, start_x:start_x + w, :]
        neighbor_list.append(neighbor)

    # stack -> (h, w, neighbors, k)
    neighbor_stack = np.stack(neighbor_list, axis=2)
    return neighbor_stack

def make_shifts_from_mode(mode='8', radius=None, k=None, distance_fn=None):
    """
      Returns a list of offsets (dy, dx) according to the chosen mode.

      Parameters
      ----------
      mode : str
          One of {'4', '8', 'k', 'radius'}.
      radius : int, optional
          Neighborhood radius (required if mode='radius' or mode='k').
      k : int, optional
          Number of nearest neighbors to include (only if mode='k').
      distance_fn : callable, optional
          A function distance_fn(dy, dx) -> float. Defaults to Euclidean (see Neighborhood class).

      Returns
      -------
      list[tuple[int, int]]
          List of offsets (dy, dx).
      """

    ""

    # --- 1. Classic modes ---
    if mode == '4':
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if mode == '8':
        return [(-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)]

    # --- 2. Radius-based or k-nearest ---
    if mode in ('radius', 'k'):
        if radius is None:
            raise ValueError(f"radius required for mode='{mode}'")

        # generate offsets in a square [-r, r], excluding (0,0)
        offsets = [(dy, dx)
                   for dy in range(-radius, radius + 1)
                   for dx in range(-radius, radius + 1)
                   if not (dy == 0 and dx == 0)]

        # compute distances
        distances = np.array([distance_fn(dy, dx) for dy, dx in offsets])

        # --- mode='radius': keep those within the radius ---
        if mode == 'radius':
            mask = distances <= radius
            return [offsets[i] for i in np.where(mask)[0]]

        # --- mode='k': pick k nearest ---
        if mode == 'k':
            if k is None:
                raise ValueError("k must be provided for mode='k'")
            sorted_pairs = sorted(zip(distances, offsets), key=lambda t: t[0])
            return [offset for _, offset in sorted_pairs[:k]]

    raise ValueError(f"Unknown mode={mode}")