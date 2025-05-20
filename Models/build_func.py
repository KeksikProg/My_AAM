import casadi as ca
import numpy as np

def build_func(mean_texture, appearance_deltas, target_texture, TEXTURE_SIZE):

    mean_texture_ca = ca.MX(mean_texture)
    appearance_deltas_ca = ca.MX(appearance_deltas.T)
    target_texture_ca = ca.MX(target_texture)

    num_app_params = appearance_deltas.shape[0]
    all_params = ca.MX.sym('app_params', num_app_params)

    model_texture = mean_texture_ca + appearance_deltas_ca @ all_params
    residuals_sym = model_texture - target_texture_ca

    J_sym = ca.jacobian(model_texture, all_params)

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
