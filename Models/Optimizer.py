import casadi as ca
import numpy as np
from Visualize import plot_model_shape

def build_functions(params, linear_comb, target_flat):
    residuals = ca.vec(linear_comb - ca.MX(target_flat))
    J = ca.jacobian(residuals, params)
    JTJ = J.T @ J
    JTr = J.T @ residuals
    loss = ca.dot(residuals, residuals) / residuals.shape[0]

    return {
        "JTJ": ca.Function("JTJ", [params], [JTJ]),
        "JTr": ca.Function("JTr", [params], [JTr]),
        "loss": ca.Function("loss", [params], [loss]),
        "result": ca.Function("result", [params], [linear_comb])
    }

def optimize(functions, init_params, max_iter=100, damping=1.0, target_model=None):
    params = init_params.copy()
    for i in range(max_iter):
        if target_model is not None:
            current_flat = functions["result"](params).full().flatten()
            plot_model_shape(current_flat, f"Итерация {i}", target_model)

        JTJ_val = functions["JTJ"](params).full()
        JTr_val = functions["JTr"](params).full()
        JTJ_damped = JTJ_val + damping * np.eye(JTJ_val.shape[0])
        delta = -np.linalg.inv(JTJ_damped) @ JTr_val
        params += delta.flatten()
        err = float(functions["loss"](params))

        if err < 1e-6:
            break

        print(f"Итерация {i}, ошибка: {err:.6f}")
    return params