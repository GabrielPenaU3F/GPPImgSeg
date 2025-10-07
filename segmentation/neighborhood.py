class Neighborhood:

    def __init__(self, neighborhood_type: str, radius=0, k=0, custom_shifts=None):
        self.neighborhood_type = neighborhood_type
        self.radius = radius
        self.k = k
        self.custom_shifts = custom_shifts

    def get_type(self):
        return self.neighborhood_type

    def get_radius(self):
        return self.radius

    def get_k(self):
        return self.k

    def get_custom_shifts(self):
        return self.custom_shifts