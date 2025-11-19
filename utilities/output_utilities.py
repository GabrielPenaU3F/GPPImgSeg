import os

from PIL import Image
from matplotlib import pyplot as plt

def save_frame(X, n_frame, save_directory):
    frame = Image.fromarray(X)
    file = os.path.join(save_directory, f'frame_{n_frame:03d}.png')
    frame.save(file)

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center', color='black')

    plt.colorbar()
    plt.tight_layout()
    plt.show()