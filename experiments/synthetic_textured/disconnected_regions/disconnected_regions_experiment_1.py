import numpy as np

from matplotlib import pyplot as plt
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import PolyaLabeler, GPPLabeler
from segmentation.metrics import SegmentationComparator
from segmentation.neighborhood import Neighborhood
from synthesizers.disconnected_image_generator import DisconnectedRegionsImageGenerator
from utilities.image_format_utilities import label_image_from_probabilities, align_labels, normalize_labels
from utilities.output_utilities import plot_confusion_matrix, plot_regional_mse_bars

k = 3 # Number of regions
seed = 42
generator = DisconnectedRegionsImageGenerator()
img, ground_truth = generator.generate_textured_image(size=(256, 256), n_regions=k, seed=seed,
                                                      smoothness=[0.4, 0.4, 0.4], intensity=[0.7, 0.8, 1.2])

# --- Reference ---

fig1, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0].axis('off')
ax[0].set_title('Original Image')

ax[1].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[1].axis('off')
ax[1].set_title('Ground Truth')

fig1.tight_layout()
plt.show()

# --- Labeling ---

nmc_labels = NMCLabeler(seed).label(img, n_iter=10, n_classes=k, return_type='raw')
nmc_img = align_labels(nmc_labels, ground_truth)

fig1, ax = plt.subplots(1, 3, figsize=(12, 8))

ml_probs = MLLabeler().label(img, nmc_labels, return_type='probs')
ml_labels = label_image_from_probabilities(ml_probs)
ml_img = align_labels(ml_labels, ground_truth)

neighborhood = Neighborhood('radius', radius=4)
rl_labels = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=20, return_type='img')
rl_img = align_labels(rl_labels, ground_truth)

polya_labels = PolyaLabeler().label(ml_probs, neighborhood, initial_total_balls=100, R=10, n_iter=30,
                                    return_type='img')
polya_img = align_labels(polya_labels, ground_truth)

reinforcement_matrix_super = np.ones((3, 3)) + 8 * np.eye(3)
gpp_superdif_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                         R=reinforcement_matrix_super, n_iter=20, return_type='img')
gpp_superdif_img = align_labels(gpp_superdif_labels, ground_truth)

reinforcement_matrix_sub = np.ones((3, 3))
gpp_subdif_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                       R=reinforcement_matrix_sub, n_iter=20, return_type='img')
gpp_subdif_img = align_labels(gpp_subdif_labels, ground_truth)

reinforcement_matrix_hyper = -5 * np.ones((3, 3)) + 25 * np.eye(3)
gpp_hyper_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                      R=reinforcement_matrix_hyper, n_iter=20, return_type='img')
gpp_hyper_img = align_labels(gpp_hyper_labels, ground_truth)

# ---- Plot ----

fig2, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0, 0].imshow(ml_img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('NMC + ML labeling')

ax[0, 1].imshow(rl_img, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('Relaxation labeling')

ax[0, 2].imshow(gpp_subdif_img, cmap='gray', vmin=0, vmax=255)
ax[0, 2].axis('off')
ax[0, 2].set_title('Subdiffusive GPP labeling')

ax[1, 0].imshow(gpp_superdif_img, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title('Superdiffusive GPP labeling')

ax[1, 1].imshow(polya_img, cmap='gray', vmin=0, vmax=255)
ax[1, 1].axis('off')
ax[1, 1].set_title('Polya labeling')

ax[1, 2].imshow(gpp_hyper_img, cmap='gray', vmin=0, vmax=255)
ax[1, 2].axis('off')
ax[1, 2].set_title('Hyperballistic GPP labeling')

fig2.tight_layout()
plt.show()

# --- Metrics ---

# Lista ordenada para automatizar
experiment_names = [
    "NMC + ML",
    "Relaxation",
    "Subdiffusive GPP",
    "Superdiffusive GPP",
    "Polya",
    "Hyperballistic GPP",
]

predictions = [
    ml_img,
    rl_img,
    gpp_subdif_img,
    gpp_superdif_img,
    polya_img,
    gpp_hyper_img,
]

comparator = SegmentationComparator()

adjusted_rands = []
bf1_scores = []
regional_mses = []
for pred in predictions:
    adjusted_rands.append(comparator.adjusted_rand(ground_truth, pred))
    bf1_scores.append(comparator.boundary_f1(ground_truth, pred, tol=2))
    regional_mses.append(comparator.regional_mse(ground_truth, pred, return_type='region'))

adjusted_rand = np.array(adjusted_rands)
bf1_scores = np.array(bf1_scores)

plt.figure(figsize=(10, 5))
plt.bar(experiment_names, adjusted_rands, color='blue')
plt.ylabel("Adjusted Rand Index")
plt.title("Comparison of methods (ARI)")
plt.xticks(rotation=20)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(experiment_names, bf1_scores, color='C0')
plt.ylim(0, 1.0)
plt.ylabel('Boundary F1')
plt.title('Comparison of methods (BF1)')
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

plot_regional_mse_bars(regional_mses, experiment_names)

for pred, name in zip(predictions, experiment_names):
    pred_norm = normalize_labels(pred)
    gt_norm = normalize_labels(ground_truth)
    cm, labels = comparator.compute_confusion_matrix(gt_norm, pred_norm)
    plot_confusion_matrix(cm, labels, title=f"Confusion - {name}")
