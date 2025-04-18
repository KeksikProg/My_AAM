import casadi as ca
import numpy as np
from Optimizer import build_functions, optimize
from ShapeModel import shape_model
from Normilizer import normilize
from pathlib import Path
from Visualize import show_texture, plot_model_shape
import matplotlib.pyplot as plt
from read_dataset import read_dataset_from_pts
from warp_images import warp_images_to_mean_shape
from AppearanceModel import appearance_model


if __name__ == "__main__":

    images, blandshapes = read_dataset_from_pts("dataset")

    normalized = normilize(blandshapes[:-1])
    base_shape = np.mean(normalized, axis = 0)
    bland_shapes = normalized
    shape_params_sym, shape_linear_comb = shape_model(base_shape, bland_shapes)
    # Сделали Shape Model как линейную комбинацию

    warp = warp_images_to_mean_shape(images, blandshapes, base_shape) # Тут нужно именно не нормированные точки, которые привязаны к фотографиям
    warp_vectors = np.array([img.flatten() for img in warp])
    mean_texture = np.mean(warp_vectors, axis = 0)
    appearance_deltas = warp_vectors - mean_texture
    w_a, texture_model = appearance_model(mean_texture, appearance_deltas)
    # Сделали Appearance Model как линейную комбинацию

    # На данном этапе мы имеем две модели нужные нам, дальше уже стоит вопрос оптимизации их. Пока оптимизируется только Shape Model
    target_model_shape = np.array(blandshapes[-1]).flatten()
    func = build_functions(shape_params_sym, shape_linear_comb, target_model_shape)

    init_params_shape = np.zeros(len(bland_shapes) + 2)

    
    final_params_shape = optimize(func, init_params_shape, target_model = target_model_shape)

    for par in range(len(final_params_shape[:-2])):
        print(f"Коэффициент блендшейпа {par + 1} равен: {final_params_shape[par]}")
    print(f"Трансляция по x равна: {final_params_shape[-2]}, по y равна: {final_params_shape[-1]}")
