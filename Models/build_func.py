import casadi as ca
import numpy as np
from warp_data_Jp import symbolic_warp
from warp_images import warp_images_to_mean_shape
from Image_grad import image_grad
import matplotlib.pyplot as plt
from line_interpolate import bilinear_interpolate

def build_func(center_base_shape, bland_shapes, mean_texture, appearance_deltas, 
               triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel,
               target_image, target_shape, TEXTURE_SIZE):

    grad_fn = image_grad(TEXTURE_SIZE)
    dx, dy = grad_fn(mean_texture.reshape(TEXTURE_SIZE))
    dx = dx.full().flatten()
    dy = dy.full().flatten()
    
    # plt.figure()
    # plt.imshow(dx.reshape(TEXTURE_SIZE), cmap='gray')
    # plt.title('dx градиент')
    # plt.colorbar()
    # plt.show()
    
    # plt.figure()
    # plt.imshow(dy.reshape(TEXTURE_SIZE), cmap='gray')
    # plt.title('dy градиент')
    # plt.colorbar()
    # plt.show()


    # Фильтрация валидных пикселей
    dx = dx[is_valid_pixel]
    dy = dy[is_valid_pixel]

    dx_dm = ca.MX(dx)
    dy_dm = ca.MX(dy)

    warped_target = warp_images_to_mean_shape([target_image], [target_shape], center_base_shape, TEXTURE_SIZE)[0].flatten()
    warped_target = warped_target[is_valid_pixel]
    warped_target_dm = ca.MX(warped_target)

    num_shape_params = bland_shapes.shape[0] + 2
    num_app_params = appearance_deltas.shape[0]

    shape_params_sym = ca.MX.sym('shape_params', num_shape_params)
    app_params_sym = ca.MX.sym('app_params', num_app_params)
    all_params = ca.vertcat(shape_params_sym, app_params_sym)

    warp_coords_sym = symbolic_warp(shape_params_sym, center_base_shape, bland_shapes, triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel)
    dW_dp_sym = ca.jacobian(warp_coords_sym, shape_params_sym)

    dW_x = dW_dp_sym[::2, :]
    dW_y = dW_dp_sym[1::2, :]

    J_p_sym = ca.diag(dx_dm) @ dW_x + ca.diag(dy_dm) @ dW_y

    mean_texture_filtered = mean_texture[is_valid_pixel]
    mean_texture_ca = ca.MX(mean_texture_filtered)

    appearance_deltas_filtered = appearance_deltas[:, is_valid_pixel]
    appearance_deltas_ca = ca.MX(appearance_deltas_filtered.T)

    texture_current_sym = bilinear_interpolate(mean_texture_ca + appearance_deltas_ca @ app_params_sym, warp_coords_sym, TEXTURE_SIZE)


    J_c_sym = ca.jacobian(texture_current_sym, app_params_sym)

    J_sym = ca.horzcat(J_p_sym, J_c_sym)
    residuals_sym = texture_current_sym - warped_target_dm

    JTJ_sym = J_sym.T @ J_sym
    JTr_sym = J_sym.T @ residuals_sym
    loss_sym = ca.dot(residuals_sym, residuals_sym) / residuals_sym.shape[0]

    JTJ_fn = ca.Function('JTJ_fn', [all_params], [JTJ_sym])
    JTr_fn = ca.Function('JTr_fn', [all_params], [JTr_sym])
    loss_fn = ca.Function('loss_fn', [all_params], [loss_sym])

    def builder(current_params):
        return {
            "JTJ": JTJ_fn(current_params).full(),
            "JTr": JTr_fn(current_params).full(),
            "loss": float(loss_fn(current_params))
        }

    return builder







'''
import casadi as ca
import numpy as np
from warp_data_Jp import symbolic_warp
from warp_images import warp_images_to_mean_shape
from Image_grad import image_grad

def build_func(center_base_shape, bland_shapes, mean_texture, appearance_deltas, triangles, pixel_triangle_ids, pixel_bary_coords, target_image, target_shape, TEXTURE_SIZE):

    grad_fn = image_grad(TEXTURE_SIZE)
    dx, dy = grad_fn(mean_texture.reshape(TEXTURE_SIZE))
    dx = dx.full().flatten()
    dy = dy.full().flatten()

    dx_dm = ca.DM(dx)
    dy_dm = ca.DM(dy)

    warped_target = warp_images_to_mean_shape([target_image], [target_shape], center_base_shape, TEXTURE_SIZE)[0].flatten()

    num_shape_params = len(bland_shapes) + 2
    num_app_params = appearance_deltas.shape[0]

    shape_params_sym = ca.MX.sym('shape_params', num_shape_params)
    app_params_sym = ca.MX.sym('app_params', num_app_params)
    all_params = ca.vertcat(shape_params_sym, app_params_sym)

    # Варпим координаты в зависимости от shape-параметров
    warp_coords = symbolic_warp(shape_params_sym, center_base_shape, bland_shapes, triangles, pixel_triangle_ids, pixel_bary_coords)
    dW_dp = ca.jacobian(warp_coords, shape_params_sym)

    dW_x = dW_dp[::2, :]
    dW_y = dW_dp[1::2, :]

    # Строим J_p
    J_p = ca.diag(dx_dm) @ dW_x + ca.diag(dy_dm) @ dW_y

    appearance_deltas_ca = ca.DM(appearance_deltas.T)
    # Строим текущую текстуру
    texture_current = mean_texture + appearance_deltas_ca @ app_params_sym

    # Строим J_c
    J_c = ca.jacobian(texture_current, app_params_sym)

    # Финальный Якобиан
    J = ca.horzcat(J_p, J_c)

    # Остатки
    residuals = texture_current - warped_target

    JTJ = J.T @ J
    JTr = J.T @ residuals
    loss = ca.dot(residuals, residuals) / residuals.shape[0]

    functions = {
        "JTJ": ca.Function("JTJ", [all_params], [JTJ]),
        "JTr": ca.Function("JTr", [all_params], [JTr]),
        "loss": ca.Function("loss", [all_params], [loss]),
    }

    return functions
'''