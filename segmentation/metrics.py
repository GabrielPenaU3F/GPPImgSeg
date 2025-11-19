import numpy as np
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix

from utilities.image_format_utilities import normalize_labels, align_labels


class SegmentationComparator:

    def boundary_f1(self, gt, pred, tol=2, connectivity='8'):
        """
        Compute a Boundary F1 score between ground truth and prediction.

        Parameters
        ----------
        gt : array_like
            Ground-truth label image (H x W). Labels can be ints.
        pred : array_like
            Predicted label image (H x W).
        tol : int, optional
            Tolerance radius (in pixels). Default 2.
        connectivity : '4' or '8', optional
            Connectivity for boundary extraction. '8' is default.

        Returns
        -------
        bf1 : float
            Boundary F1 in [0, 1].
        """
        gt = np.asarray(gt)
        pred = np.asarray(pred)

        if gt.shape != pred.shape:
            raise ValueError("gt and pred must have the same shape")

        # find_boundaries returns a boolean array marking boundaries between labels
        mode = 'thick'  # 'thick' gives a thicker boundary which is often more robust
        if connectivity == '4':
            boundary_mode = {'connectivity': 1}
        else:
            boundary_mode = {'connectivity': 2}

        # Extract boundaries
        gt_b = find_boundaries(gt, mode=mode, **boundary_mode)
        pred_b = find_boundaries(pred, mode=mode, **boundary_mode)

        # If both have no boundaries (constant images), define BF1 = 1 if equal else 0
        if not gt_b.any() and not pred_b.any():
            return 1.0 if np.array_equal(gt, pred) else 0.0

        # if one has no boundaries and the other does, BF1 = 0
        if not gt_b.any() or not pred_b.any():
            return 0.0

        # Dilate boundaries by tolerance
        if tol > 0:
            selem = disk(tol)  # circular structuring element
            gt_b_dil = binary_dilation(gt_b, structure=selem)
            pred_b_dil = binary_dilation(pred_b, structure=selem)
        else:
            gt_b_dil = gt_b
            pred_b_dil = pred_b

        # Precision: fraction of predicted boundary pixels that fall near a GT boundary
        pred_matches = (pred_b & gt_b_dil).sum()
        pred_total = pred_b.sum()
        precision = pred_matches / pred_total if pred_total > 0 else 0.0

        # Recall: fraction of GT boundary pixels that fall near a predicted boundary
        gt_matches = (gt_b & pred_b_dil).sum()
        gt_total = gt_b.sum()
        recall = gt_matches / gt_total if gt_total > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        bf1 = 2.0 * precision * recall / (precision + recall)
        return float(bf1)

    def adjusted_rand(self, gt, pred):
        gt = normalize_labels(gt)
        pred = normalize_labels(pred)
        return adjusted_rand_score(gt.flatten(), pred.flatten())

    def regional_mse(self, gt, pred, return_type='mean'):
        labels = np.unique(gt)
        mse_vals = []

        for lab in labels:
            region = (gt == lab)
            if region.sum() == 0:
                continue

            mse = np.mean((gt[region] - pred[region]) ** 2)
            mse_vals.append(mse)

        mse_vals = np.array(mse_vals)

        if return_type == 'region':
            return mse_vals
        elif return_type == 'mean':
            return mse_vals.mean() if len(mse_vals) > 0 else 0.0

    def compute_confusion_matrix(self, gt, pred):
        """
        Compute confusion matrix between ground truth and prediction.

        Parameters
        ----------
        gt : ndarray (H, W)
        pred : ndarray (H, W)

        Returns
        -------
        cm : ndarray
            Confusion matrix of shape (n_classes, n_classes)
        """
        gt = gt.astype(int).ravel()
        pred = align_labels(pred, gt, return_type='labels')
        pred = pred.astype(int).ravel()

        labels = np.unique(gt)
        cm = confusion_matrix(gt.ravel(), pred.ravel(), labels=labels)

        return cm, labels