import casadi as ca

def image_grad(texture_size):
    H, W = texture_size
    img = ca.MX.sym('img', H, W)

    # Градиент по X (по столбцам)
    pad_left  = ca.horzcat(ca.MX.zeros(H, 1), img[:, :-1])
    pad_right = ca.horzcat(img[:, 1:], ca.MX.zeros(H, 1))
    dI_dx = 0.5 * (pad_right - pad_left)

    # Градиент по Y (по строкам)
    pad_top    = ca.vertcat(ca.MX.zeros(1, W), img[:-1, :])
    pad_bottom = ca.vertcat(img[1:, :], ca.MX.zeros(1, W))
    dI_dy = 0.5 * (pad_bottom - pad_top)

    # Собрать функцию
    grad_fn = ca.Function('grad_fn', [img], [dI_dx, dI_dy])

    return grad_fn
