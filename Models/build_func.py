import casadi as ca
import numpy as np
from warp_data_Jp import symbolic_warp
from Image_grad import image_grad
import matplotlib.pyplot as plt

def build_func(base_shape, bland_shapes, mean_texture, appearance_deltas, 
               triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel,
               target_texture, target_shape, TEXTURE_SIZE):

    grad_fn = image_grad(TEXTURE_SIZE)
    dx, dy = grad_fn(mean_texture.reshape(TEXTURE_SIZE))
    dx = dx.full().flatten()[is_valid_pixel]
    dy = dy.full().flatten()[is_valid_pixel]

    dx_dm = ca.MX(dx)
    dy_dm = ca.MX(dy)

    # Константы
    mean_texture_filtered = mean_texture[is_valid_pixel]
    appearance_deltas_filtered = appearance_deltas[:, is_valid_pixel]

    mean_texture_ca = ca.MX(mean_texture_filtered)
    appearance_deltas_ca = ca.MX(appearance_deltas_filtered.T)
    target_texture_ca = ca.MX(target_texture[is_valid_pixel])

    num_shape_params = bland_shapes.shape[0] + 3
    num_app_params = appearance_deltas.shape[0]
    shape_params_sym = ca.MX.sym('shape_params', num_shape_params)
    app_params_sym = ca.MX.sym('app_params', num_app_params)
    all_params = ca.vertcat(shape_params_sym, app_params_sym)

    warp_coords_sym = symbolic_warp(shape_params_sym, base_shape, bland_shapes,
                                     triangles, pixel_triangle_ids,
                                     pixel_bary_coords, is_valid_pixel)

    model_texture = mean_texture_ca + appearance_deltas_ca @ app_params_sym
    
    # plt.imshow((mean_texture + appearance_deltas.T @ app_params_sym.full()).reshape(TEXTURE_SIZE), cmap="gray")
    # plt.show()

    residuals_sym = model_texture - target_texture_ca

    dW_dp_sym = ca.jacobian(warp_coords_sym, shape_params_sym)
    dW_x = dW_dp_sym[::2, :]
    dW_y = dW_dp_sym[1::2, :]

    J_p_sym = ca.diag(dx_dm) @ dW_x + ca.diag(dy_dm) @ dW_y
    J_c_sym = ca.jacobian(model_texture, app_params_sym)
    J_sym = ca.horzcat(J_p_sym, J_c_sym)

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
