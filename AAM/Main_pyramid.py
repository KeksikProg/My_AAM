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


#TEXTURE_SIZE =  (128, 128) 
LEVELS = [32, 64, 128, 256]  # можешь добавить 256
init_params = None
SIZE_DATASET = -1
TARGET_INDEX = 46
EPOCHS = 10

if __name__ == "__main__":


    for lvl in LEVELS:
        print(f"\n===== УРОВЕНЬ {lvl}x{lvl} =====")

        # 1. Загрузка данных
        images, shapes = read_dataset_from_pts(f"dataset_teach{lvl}")
        print("Датасет считался")

        normalized_shapes = normilize(shapes[:SIZE_DATASET])
        base_shape = np.mean(normalized_shapes, axis=0)
        bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])
        print("Форма нормализована")

        tri = Delaunay(base_shape)
        triangles = tri.simplices
        print("Триангуляция построена")

        # 2. Варпинг всех изображений
        warped_images = [
            warp_piecewise_affine(img, shape, base_shape, triangles, output_size=(lvl, lvl))
            for img, shape in zip(images[:SIZE_DATASET], shapes[:SIZE_DATASET])
        ]
        warp_vectors = np.array([img.flatten() for img in warped_images]).astype(np.float32) / 255.0
        mean_texture = np.mean(warp_vectors, axis=0)
        appearance_deltas = warp_vectors - mean_texture
        print("Текстурные блендшейпы готовы")

        # 3. Подготовка целевого изображения
        target_image = images[TARGET_INDEX]
        print("Целевое изображение загружено")

        # 4. Построение функции
        func = build_func(mean_texture, appearance_deltas, target_image,
                          (lvl, lvl), base_shape, bland_shapes_delt, triangles)
        print("Функция построена")

        # 5. Параметры
        num_app_params = appearance_deltas.shape[0]
        num_shape_params = len(bland_shapes_delt) + 2
        if init_params is None:
            init_params = np.zeros(num_app_params + num_shape_params)

        # 6. Отображение начальной формы
        init_shape = base_shape + (bland_shapes_delt.T @ init_params[num_app_params:-2]).T
        init_shape += np.array([init_params[-2], init_params[-1]])
        # plt.imshow(target_image, cmap="gray")
        # plt.scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=5)
        # plt.title(f"Разрешение: {lvl}")
        # plt.show()

        # 7. Оптимизация
        print("Пошла оптимизация")
        init_params = optimize(func, init_params, EPOCHS)
        print("Оптимизация закончена")

    



    print(f"Начальные параметры: \n Параметры внешности: {init_params[:num_app_params]} \n Параметры формы: {init_params[num_app_params:-2]} \n Параметры смещения {init_params[-2], init_params[-1]}")
    print("----------------------------------")
    #print(f"Финальные параметры: \n Параметры внешности: {final_params[:num_app_params]} \n Параметры формы: {final_params[num_app_params:-2]} \n Параметры смещения {final_params[-2], final_params[-1]}")
    
    #rec_shape = base_shape + (bland_shapes_delt.T @ final_params[num_app_params:-2]).T
    #rec_shape += np.array([final_params[-2], final_params[-1]])
    
    or_shape = normilize([shapes[TARGET_INDEX]])[0] 

    fig, axs = plt.subplots(1, 2, figsize=(30, 10))

    axs[0].imshow(target_image, cmap='gray')
    axs[0].scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=5)
    axs[0].set_title("Восстановленная форма")
    
    #axs[1].imshow(target_image, cmap='gray')
    #axs[1].scatter(rec_shape[:, 0], rec_shape[:, 1], c='red', s=5)
    #axs[1].set_title("Восстановленная форма")
    
    axs[1].imshow(target_image, cmap='gray')
    axs[1].scatter(or_shape[:, 0], or_shape[:, 1], c='red', s=5)
    axs[1].set_title("Оригинальная форма")

    plt.show()


