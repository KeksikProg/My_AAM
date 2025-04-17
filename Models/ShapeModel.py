import casadi as ca

def shape_model(base_shape, blend_shapes):
    base_flat = ca.MX(base_shape.flatten())
    blends_mx = [ca.MX(s.flatten()) - base_flat for s in blend_shapes]

    w = ca.MX.sym("w", len(blends_mx))
    tx = ca.MX.sym("tx")
    ty = ca.MX.sym("ty")
    translation = ca.vertcat(*([tx, ty] * (base_flat.shape[0] // 2)))
    params = ca.vertcat(w, tx, ty)

    linear_comb = (base_flat + ca.mtimes(ca.horzcat(*blends_mx), w)) + translation
    return params, linear_comb