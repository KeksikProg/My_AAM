import casadi as ca
import numpy as np
from Optimizer import optimize
from Normilizer import normilize
from read_dataset import read_dataset_from_pts
from warp_data_Jp import prepare_warp_data, symbolic_warp
from build_func import build_func
import matplotlib.pyplot as plt
from warp_utils import warp_piecewise_affine

def visualize_landmarks(image, landmarks, title=None):
    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]

    plt.imshow(image, cmap="gray")
    plt.scatter(xs, ys, s=10, c='red')
    if title:
        plt.title(title)
    plt.axis('on')
    plt.show()


TEXTURE_SIZE =  (128, 128) #(350, 350) 
SIZE_DATASET = 190


if __name__ == "__main__":
    #Считывание датасет
    images, shapes = read_dataset_from_pts("dataset")
    print("Датасет считался")

    # Нормализация форм, построение средней формы
    normalized_shapes = normilize(shapes[:SIZE_DATASET])
    base_shape = np.mean(normalized_shapes, axis = 0)
    bland_shapes = np.array(normalized_shapes)
    bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])
    print("Все нормализовалось, блендшейпы формы посчитались")

    # Триангуляция по базовой форме
    triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel = prepare_warp_data(base_shape, TEXTURE_SIZE)
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

    # #Посмотреть среднюю текстуру 
    # plt.imshow(mean_texture.reshape(TEXTURE_SIZE), cmap="gray")
    # plt.show()

    # Определение целевого изображения и его точек
    target_image = images[SIZE_DATASET + 3]
    target_shapes = shapes[SIZE_DATASET + 3]
    target_texture = warp_piecewise_affine(target_image, target_shapes, base_shape, triangles, TEXTURE_SIZE)
    target_texture = (target_texture.astype(np.float32) / 255.0).flatten()
    print("Целевое изображение определено")

    # #Посмотреть варпнутое к средней форме таргетное изображение
    # plt.imshow(target_texture.reshape(TEXTURE_SIZE), cmap="gray")
    # plt.show()
    
    # Все что выше я проверил уже
    ###---###

    func = build_func(base_shape, bland_shapes_delt, mean_texture, appearance_deltas, triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel, target_texture, target_shapes, TEXTURE_SIZE)
    print("Функция построена")

    num_shape_params = len(bland_shapes) + 3
    num_app_params = appearance_deltas.shape[0]
    init_params = np.zeros(num_shape_params + num_app_params)
    print("Параметры иницилизированы")


    print("Пошла оптимизация")
    final_params = optimize(func, init_params, 15)
    print("Оптимизация закончена")

    print("Финальные параметры:\n")
    print(f"Параметры формы: {final_params[:num_shape_params-3]}")
    print(f"Коэффициент поворота: {final_params[num_shape_params-3]}")
    print(f"Параметры смещения: {final_params[num_shape_params-2:num_shape_params]}")
    print(f"Параметры текстуры: {final_params[num_shape_params:]}\n")

    num_shape_params = len(bland_shapes) + 3
    shape_params = final_params[:num_shape_params]
    appearance_params = final_params[num_shape_params:]
    
    reconstructed_texture = mean_texture + appearance_deltas.T @ appearance_params
    reconstructed_texture_img = reconstructed_texture.reshape(TEXTURE_SIZE)

    start_img = mean_texture + appearance_deltas.T @ init_params[num_shape_params:]
    start_img = start_img.reshape(TEXTURE_SIZE)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # print(f"start_img: min={start_img.min()}, max={start_img.max()}")
    # print(f"reconstructed_texture: min={reconstructed_texture.min()}, max={reconstructed_texture.max()}")
    # print(f"target_texture: min={target_texture.min()}, max={target_texture.max()}")

    axs[0].imshow(start_img, cmap='gray')
    axs[0].set_title('Старт формы')
    axs[0].axis('on')

    axs[1].imshow(reconstructed_texture_img, cmap='gray')
    axs[1].set_title('Восстановленная текстура')
    axs[1].axis('on')

    axs[2].imshow(target_texture.reshape(TEXTURE_SIZE), cmap='gray')
    axs[2].set_title('Варовненное целевое изображение (Target)')
    axs[2].axis('on')
    
    plt.show()

