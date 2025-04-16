from Kabsch import kabsch
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Визуализация ---
def plot_model(flat_model, title, target_flat=None):
    model = flat_model.reshape(-1, 2)
    plt.plot(model[:, 0], model[:, 1], 'o-', label="Собранная модель")
    if target_flat is not None:
        target_model = target_flat.reshape(-1, 2)
        plt.plot(target_model[:, 0], target_model[:, 1], 'x--', label="Целевая модель")
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def normilize(shapes):
    base = shapes[0]
    normalized = []
    for shape in shapes:
        R, t = kabsch(shape, base)
        aligned = np.array(shape) @ R + t
        normalized.append(aligned)
    return normalized

def shape_model(base_shape, blend_shapes):
    base_flat = ca.MX(base_shape.flatten())
    blends_mx = [ca.MX(s.flatten()) - base_flat for s in blend_shapes]

    w = ca.MX.sym("w", len(blends_mx))
    tx = ca.MX.sym("tx")
    ty = ca.MX.sym("ty")
    translation = ca.vertcat(*([tx, ty] * (base_flat.shape[0] // 2)))
    params = ca.vertcat(w, tx, ty)

    linear_comb = base_flat + ca.mtimes(ca.horzcat(*blends_mx), w) + translation
    return params, linear_comb

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
            plot_model(current_flat, f"Итерация {i}", target_model)

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

if __name__ == "__main__":
    blandshapes = [
        [[-1, 0], [0, 0], [1, 0]],
        [[-2, 0], [0, 0], [2, 0]],
        [[-1, 0], [0, 1], [1, 0]],
    ]

    #target_model = np.array([[-1, 0], [0, 0], [1, 0]]).flatten()
    #target_model = np.array([[0, 2], [1, 2], [2, 2]]).flatten()
    target_model = np.array([[-1, 0], [0, 1], [1, 0]]).flatten()
    #target_model = np.array([[-1, 0], [0, 2], [1, 0]]).flatten()


    normalized = normilize(blandshapes)
    base_shape = normalized[0]
    blend_shapes = normalized[1:]

    params_sym, linear_comb = shape_model(base_shape, blend_shapes)
    functions = build_functions(params_sym, linear_comb, target_model)

    init_params = np.zeros(len(blend_shapes) + 2)
    final_params = optimize(functions, init_params, target_model=target_model)

    for par in range(len(final_params[:-2])):
        print(f"Коэффициент блендшейпа {par + 1} равен: {final_params[par]}")

    print(f"Трансляция по x равна: {final_params[-2]}, по y равна: {final_params[-1]}")