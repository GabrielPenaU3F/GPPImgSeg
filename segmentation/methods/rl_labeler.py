import numpy as np

from segmentation.utilities import label_image_from_probabilities, get_neighbor_stack

class RelaxationLabeler:

    def label(self, probs, neighborhood, n_iter=10, return_type='img', compatibility_matrix=None):

        h, w, k = probs.shape # height, width and number of classes
        if compatibility_matrix is None:
            compatibility_matrix = self.compatibility_matrix(k)
        else:
            compatibility_matrix = self.validate_compatibility_matrix(compatibility_matrix, k)

        for _ in range(n_iter):
            '''
                1)  Build the stack of neighbor probabilities: this is a (h, w, 8, k) tensor
                    It contains 8 layers of the probs matrix, each corresponding to the 'probs' of one
                    of the neighbors
            '''
            neighbor_stack = get_neighbor_stack(probs, neighborhood) # (h, w, 8, k)

            '''
                2)  Sum over 8-neighbors -> vecs is a (h, w, k) tensor
                    For each pixel, this contains the sum of each class probabilities (over its neighbors)
            '''
            vecs = neighbor_stack.sum(axis=2) # (h, w, k)

            '''
                3)  For each pixel, we calculate S (the support) in a vectorized fashion:
                    S[i,j,l] = sum_lp R[l,lp] * vecs[i,j,lp]
                    Thus S is a (h, w, k) tensor.
                
                    This takes into account, linearly, the compatibilities and the probabilities of
                    belonging to the respective class.
                
                    Equivalent to np.einsum('ijk,lk->ijl', vecs, compatibilities)
            '''
            S = vecs @ compatibility_matrix.T # (h, w, k)

            ''' 
                4)  Update probabilities and normalize
            '''
            probs_new = probs * (1.0 + S)

            probs_new = np.clip(probs_new, 1e-8, None)
            probs_new /= probs_new.sum(axis=2, keepdims=True)
            probs = probs_new

        if return_type == 'img':
            return label_image_from_probabilities(probs)

        elif return_type == 'probs':
            return probs

    # Simple discrete metric-type compatibility
    def compatibility_matrix(self, n_classes):
        R = 2 * np.eye(n_classes) - np.ones((n_classes, n_classes))
        return R

    def validate_compatibility_matrix(self, compatibility_matrix, n_classes):
        if compatibility_matrix.shape == (n_classes, n_classes): return True
        else: raise ValueError('The provided compatibility matrix is not adequate for this number of classes')

# image_path = '../../resources/test_img.bmp'
# img = Image.open(image_path)
# X = np.array(img)
# init_labels = nmc(X, n_iter=10, n_classes=3, return_type='raw')
# init_probabilities = ml_labeling(X, init_labels, return_type='probs')
# neighborhood = Neighborhood('8')
# Y = relaxation_labeling(init_probabilities, neighborhood, n_iter=10, return_type='img')
# Y = Image.fromarray(Y)
# Y.show()
