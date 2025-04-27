from read_dataset import read_dataset
from shape_model import build_linear_shape_model
import matplotlib.pyplot as plt
import numpy as np

def plot_shape_model(model, n_std=2):
    """Визуализация модели формы."""
    mean_shape = model['mean_shape']
    basis = model['basis']
    
    plt.figure(figsize=(12, 6))
    
    # Визуализация средней формы
    plt.subplot(1, 2, 1)
    plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c='r', label='Средняя форма')
    for i, (x, y) in enumerate(mean_shape):
        plt.text(x, y, str(i), fontsize=12)
    plt.title('Средняя форма')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    
    plt.show()

if __name__ == "__main__":
    img_list, lm_list = read_dataset("Images/jpg", "Land/muct76.csv")
    model = build_linear_shape_model(lm_list)
    plot_shape_model(model)