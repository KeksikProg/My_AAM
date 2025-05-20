import casadi as ca
import numpy as np
from Optimizer import optimize
from Normilizer import normilize
from read_dataset import read_dataset_from_pts
from build_func import build_func
import matplotlib.pyplot as plt
from warp_utils import warp_piecewise_affine
from scipy.spatial import Delaunay


TEXTURE_SIZE =  (128, 128) 
SIZE_DATASET = -1


if __name__ == "__main__":
    images, shapes = read_dataset_from_pts("dataset")
    print("Датасет считался")

    normalized_shapes = normilize(shapes[:SIZE_DATASET])
    base_shape = np.mean(normalized_shapes, axis = 0)
    bland_shapes = np.array(normalized_shapes)
    bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])
    print("Все нормализовалось, блендшейпы формы посчитались")

    tri = Delaunay(base_shape)
    triangles = tri.simplices
    print("Триангуляция прошла")

    # Варпинг изображений
    warped_images = []
    for img, shape in zip(images[:SIZE_DATASET], shapes[:SIZE_DATASET]):
        warped = warp_piecewise_affine(
            img,              
            shape,            
            base_shape,       
            triangles,        
            output_size=TEXTURE_SIZE
        )
        warped_images.append(warped)

    warp_vectors = np.array([img.flatten() for img in warped_images])
    warp_vectors = warp_vectors.astype(np.float32) / 255.0
    mean_texture = np.mean(warp_vectors, axis = 0)
    appearance_deltas = warp_vectors - mean_texture
    print("Блендшейпы текстуры посчитались")

    # Определение целевого изображения и его точек
    target_image = images[SIZE_DATASET]
    target_image = warp_piecewise_affine(target_image, shapes[SIZE_DATASET-1], base_shape, triangles, TEXTURE_SIZE)
    target_image = (target_image.astype(np.float32) / 255.0).flatten()
    print("Целевое изображение определено")

    func = build_func(mean_texture, appearance_deltas, target_image, TEXTURE_SIZE)
    print("Функция построена")

    num_app_params = len(appearance_deltas)

    init_params = np.zeros(num_app_params)
    print("Параметры иницилизированы")

    print("Пошла оптимизация")
    final_params = optimize(func, init_params, 5)
    print("Оптимизация закончена")

    print(f"Финальные параметры: {final_params}\n")

    plt.imshow((mean_texture + appearance_deltas.T @ final_params).reshape(TEXTURE_SIZE), cmap='gray')
    plt.title("Восстановленная форма")
    plt.show()


