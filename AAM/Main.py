import casadi as ca
import numpy as np
from Optimizer import optimize
from Normilizer import normilize
from read_dataset import read_dataset_from_pts
from build_func import build_func
import matplotlib.pyplot as plt
from warp_utils import warp_piecewise_affine
from scipy.spatial import Delaunay
import cv2
import os


TEXTURE_SIZE =  (128, 128) 
SIZE_DATASET = -1
TARGET_INDEX = 25
EPOCHS = 20

if __name__ == "__main__":
    images, shapes = read_dataset_from_pts("dataset_teach128")
    print("Датасет считался")

    normalized_shapes = normilize(shapes[:SIZE_DATASET])
    base_shape = np.mean(normalized_shapes, axis = 0)
    bland_shapes = np.array(normalized_shapes)
    bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])
    print("Все нормализовалось, блендшейпы формы посчитались")

    tri = Delaunay(base_shape)
    triangles = tri.simplices
    print("Триангуляция прошла")

    target_shape = normalized_shapes[TARGET_INDEX]
    plt.imshow(images[TARGET_INDEX], cmap = "gray")
    plt.scatter(target_shape[:, 0], target_shape[:, 1], c="red", s=5)
    plt.show()

    # Варпинг изображений
    warped_images = [warp_piecewise_affine(img, shape, base_shape, triangles, output_size=TEXTURE_SIZE) for img, shape in zip(images[:SIZE_DATASET], shapes[:SIZE_DATASET])]

    warp_vectors = np.array([img.flatten() for img in warped_images])
    warp_vectors = warp_vectors.astype(np.float32) / 255.0
    mean_texture = np.mean(warp_vectors, axis = 0)
    appearance_deltas = warp_vectors - mean_texture
    print("Блендшейпы текстуры посчитались")

    # Определение целевого изображения и его точек
    target_image = images[TARGET_INDEX]
    print("Целевое изображение определено")

    func = build_func(mean_texture, appearance_deltas, target_image, TEXTURE_SIZE, base_shape, bland_shapes_delt, triangles)
    print("Функция построена")
    
    num_app_params = len(appearance_deltas)
    num_shape_params = len(bland_shapes_delt) + 2
    
    init_params = np.zeros(num_app_params + num_shape_params)
    init_shape = base_shape + (bland_shapes_delt.T @ init_params[num_app_params:-2]).T
    init_shape += np.array([init_params[-2], init_params[-1]])
    
    plt.imshow(images[TARGET_INDEX], cmap = "gray")
    plt.scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=10, label='init landmarks')
    plt.show()
    
    print("Параметры иницилизированы")
    
    print("Пошла оптимизация")
    final_params = optimize(func, init_params, EPOCHS)
    print("Оптимизация закончена")
    
    print(f"Начальные параметры: \n Параметры внешности: {init_params[:num_app_params]} \n Параметры формы: {init_params[num_app_params:-2]} \n Параметры смещения {init_params[-2], init_params[-1]}")
    print("----------------------------------")
    print(f"Финальные параметры: \n Параметры внешности: {final_params[:num_app_params]} \n Параметры формы: {final_params[num_app_params:-2]} \n Параметры смещения {final_params[-2], final_params[-1]}")
    
    rec_shape = base_shape + (bland_shapes_delt.T @ final_params[num_app_params:-2]).T
    rec_shape += np.array([final_params[-2], final_params[-1]])
    
    or_shape = normalized_shapes[TARGET_INDEX]

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    axs[0].imshow(target_image, cmap='gray')
    axs[0].scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=5)
    axs[0].set_title("Начальная форма")
    
    axs[1].imshow(target_image, cmap='gray')
    axs[1].scatter(rec_shape[:, 0], rec_shape[:, 1], c='red', s=5)
    axs[1].set_title("Восстановленная форма")
    
    axs[2].imshow(target_image, cmap='gray')
    axs[2].scatter(or_shape[:, 0], or_shape[:, 1], c='red', s=5)
    axs[2].set_title("Оригинальная форма")

    plt.show()


