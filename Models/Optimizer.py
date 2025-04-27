import numpy as np

def optimize(build_fn, init_params, max_iter=100, damping=1):
    params = init_params.copy()
    for i in range(max_iter):
        funcs = build_fn(params)

        JTJ_val = funcs["JTJ"]
        JTr_val = funcs["JTr"]
        loss_val = funcs["loss"]

        delta = -np.linalg.solve(JTJ_val + damping * np.eye(JTJ_val.shape[0]), JTr_val)
        params += delta.flatten()

        print(f"Итерация {i}, ошибка: {loss_val:.8f}")

        if loss_val < 1e-6:
            break

    return params