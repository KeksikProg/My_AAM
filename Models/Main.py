import casadi as ca
import numpy as np
from Optimizer import optimize
from Normilizer import normilize, center_shape
from read_dataset import read_dataset_from_pts
from warp_images import warp_images_to_mean_shape
from warp_data_Jp import prepare_warp_data
from build_func import build_func
import matplotlib.pyplot as plt


TEXTURE_SIZE =  (300, 300)#(360, 340) #(128, 128) - 160 изображений 

if __name__ == "__main__":
    #Считывание датасет
    images, blandshapes = read_dataset_from_pts("dataset")
    print("Датасет считался")

    normalized = normilize(blandshapes[:2])
    base_shape = np.mean(normalized, axis = 0)
    bland_shapes = np.array(normalized)
    center_base_shape = center_shape(base_shape, TEXTURE_SIZE)
    print("Все нормализовалось, блендшейпы формы посчитались")

    warp = warp_images_to_mean_shape(images[:2], blandshapes, center_base_shape, TEXTURE_SIZE) # Тут нужно именно не нормированные точки, которые привязаны к фотографиям
    warp_vectors = np.array([img.flatten() for img in warp])
    mean_texture = np.mean(warp_vectors, axis = 0)
    appearance_deltas = warp_vectors - mean_texture
    print("Блендшейпы текстуры посчитались")

    # Определение целевого изображения и его точек
    target_image = images[1]
    target_shapes = blandshapes[1]
    target_texture = warp_images_to_mean_shape([target_image], [target_shapes], center_base_shape, TEXTURE_SIZE)[0]
    print("Целевое изображение определено")
    
    triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel = prepare_warp_data(center_base_shape, TEXTURE_SIZE)
    func = build_func(center_base_shape, bland_shapes, mean_texture, appearance_deltas, triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel, target_texture, target_shapes, TEXTURE_SIZE)
    print("Функция построена")

    num_shape_params = len(bland_shapes) + 2
    num_app_params = appearance_deltas.shape[0]
    init_params = np.zeros(num_shape_params + num_app_params)
    print("Параметры иницилизированы")

    test_params = init_params.copy()
    test_params[0] = 1.0  # сдвинуть один shape параметр
    # test_result = func(test_params)
    # print(f"Loss после маленького сдвига shape param: {test_result['loss']:.8f}")
    
    # test_params = init_params.copy()
    test_params[num_shape_params] = 1.0  # сдвинуть один texture param
    # test_result = func(test_params)
    # print(f"Loss после маленького сдвига texture param: {test_result['loss']:.8f}")

    print("Loss при shape_param = 1.0:", func(test_params + delta * 1.0)['loss'])
    print("Loss при shape_param = 0.5:", func(test_params + delta * 0.5)['loss'])

    print("Пошла оптимизация")
    final_params = optimize(func, init_params)
    print("Оптимизация закончена")

    print("Финальные параметры:")
    print(f"Параметры формы: {final_params[:num_shape_params]}\n")
    print(f"Параметры текстуры: {final_params[num_shape_params:]}\n")

    num_shape_params = len(bland_shapes) + 2
    shape_params = final_params[:num_shape_params]
    appearance_params = final_params[num_shape_params:]
    
    reconstructed_texture = mean_texture + appearance_deltas.T @ appearance_params
    reconstructed_texture_img = reconstructed_texture.reshape(TEXTURE_SIZE)
    
    start_img = mean_texture + appearance_deltas.T @ init_params[num_shape_params:]
    start_img = start_img.reshape(TEXTURE_SIZE)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(start_img, cmap='gray')
    axs[0].set_title('Старт формы')
    axs[0].axis('off')
    
    axs[1].imshow(reconstructed_texture_img, cmap='gray')
    axs[1].set_title('Восстановленная текстура')
    axs[1].axis('off')

    axs[2].imshow(target_texture, cmap='gray')
    axs[2].set_title('Варовненное целевое изображение (Target)')
    axs[2].axis('off')
    
    plt.show()

'''
    #print(target_image[350][740])
    #shape_params_sym, shape_linear_comb = shape_model(base_shape, bland_shapes)
    #w_a, texture_model = appearance_model(mean_texture, appearance_deltas)
    # # Warp целевого изображения к точкам ShapeModel
    # warped_target_vec = warp_images_to_mean_shape([target_image], [target_shapes], center_base_shape, TEXTURE_SIZE)[0].flatten()
    # #show_texture(warped_target_vector.reshape(TEXTURE_SIZE))

    # J_c = ca.jacobian(texture_model - warped_target_vec, w_a)

    # # Создать функцию градиента
    # grad_fn = image_grad(TEXTURE_SIZE)
    # # Считаем градиент по средней текстуре
    # dx, dy = grad_fn(mean_texture.reshape(TEXTURE_SIZE))
    # dx = dx.full().flatten()
    # dy = dy.full().flatten()

    # dx_dm = ca.DM(dx)
    # dy_dm = ca.DM(dy)

    # show_texture(mean_texture.reshape(TEXTURE_SIZE))

    
    # print("Начало варпа")
    # warp_coords = symbolic_warp(shape_params_sym, center_base_shape, bland_shapes, triangles, pixel_triangle_ids, pixel_bary_coords)
    # print("symbolic_warp output shape:", warp_coords.shape)
    # print("Высчитывание Якобиана")
    # dW_dp = ca.jacobian(warp_coords, shape_params_sym)

    # print("Объединение всего в Jp")

    # dW_x = dW_dp[::2, :]  # чётные строки
    # dW_y = dW_dp[1::2, :] # нечётные строки
    
    # print("warp shape:", warped_target_vec.shape)
    # print("dx shape:", dx.shape)
    # print("dy shape:", dy.shape)
    # print("dW_dp shape:", dW_dp.shape)

    # J_p = ca.diag(dx_dm) @ dW_x + ca.diag(dy_dm) @ dW_y
    # # J_p = ca.repmat(dx_dm, 1, dW_x.shape[1]) * dW_x + ca.repmat(dy_dm, 1, dW_y.shape[1]) * dW_y

    # print("Объединение Jp и Jc")
    # J = ca.horzcat(J_p, J_c)  # (num_pixels, num_shape_params + num_app_params)

    # # На данном этапе мы имеем две модели нужные нам, дальше уже стоит вопрос оптимизации их. Пока оптимизируется только Shape Model
    # target_model_shape = np.array(blandshapes[-1]).flatten()
    # func = build_functions(shape_params_sym, shape_linear_comb, target_model_shape)
    # init_params_shape = np.zeros(len(bland_shapes) + 2)
    # final_params_shape = optimize(func, init_params_shape, target_model = target_model_shape)
    # for par in range(len(final_params_shape[:-2])):
    #     print(f"Коэффициент блендшейпа {par + 1} равен: {final_params_shape[par]}")
    # print(f"Трансляция по x равна: {final_params_shape[-2]}, по y равна: {final_params_shape[-1]}")
'''