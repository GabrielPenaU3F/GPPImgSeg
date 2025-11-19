import os

import numpy as np
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

def plot_regional_mse_bars(mse_regions_list, method_names):
    """
    mse_regions_list: lista de arrays de longitud K con el MSE por región
                      Ej: [ array([mse_r1_exp1, mse_r2_exp1, mse_r3_exp1]),
                            array([mse_r1_exp2, ...]),
                            ...
                          ]

    method_names: lista de nombres de los experimentos (longitud 6)
    """

    n_methods = len(mse_regions_list)      # Ej: 6
    K = mse_regions_list[0].shape[0]       # Cantidad de regiones, ej: 3

    # Posiciones de cada barra
    x = np.arange(n_methods)               # [0, 1, 2, 3, 4, 5]
    width = 0.25                           # ancho de cada barra

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#4C72B0", "#55A868", "#C44E52"]  # Colores por región
    labels = [f"Región {i}" for i in range(K)]

    # Plot de las barras
    for k in range(K):
        ax.bar(x + (k - 1)*width,               # posición desplazada
               [m[k] for m in mse_regions_list],
               width,
               label=labels[k],
               color=colors[k])

    # Línea horizontal del promedio de cada método
    for i in range(n_methods):
        regional_mean = mse_regions_list[i].mean()
        ax.hlines(regional_mean,
                  xmin=i - width,
                  xmax=i + width,
                  colors="black",
                  linestyles="--",
                  linewidth=1)

    # Ejes y rotación de etiquetas
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_ylabel("MSE por región")
    ax.set_title("Comparación del MSE regional entre métodos de segmentación")
    ax.legend(title="Regiones")

    plt.tight_layout()
    plt.show()