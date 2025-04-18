import casadi as ca

def appearance_model(base_texture, texture_deltas):
    base_flat = ca.MX(base_texture.flatten())  # (D,)
    deltas_mx = [ca.MX(d.flatten()) for d in texture_deltas]

    w = ca.MX.sym("w_app", len(deltas_mx))  # Параметры модели (веса)
    params = w

    # Линейная комбинация
    linear_comb = base_flat + ca.mtimes(ca.horzcat(*deltas_mx), w)

    return params, linear_comb
