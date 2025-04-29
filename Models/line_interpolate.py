import casadi as ca

def bilinear_interpolate(img, coords, texture_size):
    H, W = texture_size

    x = coords[::2]
    y = coords[1::2]

    x0 = ca.floor(x)
    y0 = ca.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Ограничение в пределах изображения
    x0 = ca.fmin(ca.fmax(x0, 0), W-1)
    x1 = ca.fmin(ca.fmax(x1, 0), W-1)
    y0 = ca.fmin(ca.fmax(y0, 0), H-1)
    y1 = ca.fmin(ca.fmax(y1, 0), H-1)

    # Индексы в один номер
    def idx(x, y):
        return y * W + x

    idx_a = idx(x0, y0)
    idx_b = idx(x0, y1)
    idx_c = idx(x1, y0)
    idx_d = idx(x1, y1)

    img_flat = ca.reshape(img, -1, 1)

    # Достаем значения по индексам
    Ia = ca.if_else(idx_a < img_flat.shape[0], img_flat[idx_a], 0)
    Ib = ca.if_else(idx_b < img_flat.shape[0], img_flat[idx_b], 0)
    Ic = ca.if_else(idx_c < img_flat.shape[0], img_flat[idx_c], 0)
    Id = ca.if_else(idx_d < img_flat.shape[0], img_flat[idx_d], 0)

    # Весовые коэффициенты
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    result = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return result
