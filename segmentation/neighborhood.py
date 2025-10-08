import numpy as np


class Neighborhood:

    def __init__(self, neighborhood_type: str, radius=0, k=0, distance_fn=None):
        self.neighborhood_type = neighborhood_type
        self.radius = radius
        self.k = k
        if distance_fn is None:
            self.distance_fn = lambda dy, dx: np.sqrt(dy**2 + dx**2)
        else:
            self.distance_fn = distance_fn

    def get_type(self):
        return self.neighborhood_type

    def get_radius(self):
        return self.radius

    def get_k(self):
        return self.k

    def get_distance_fn(self):
        return self.distance_fn