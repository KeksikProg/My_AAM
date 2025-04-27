from scipy.spatial import Delaunay
import numpy as np
import casadi as ca
from barucentr_coords import barycentric_coords

def prepare_warp_data(base_shape, texture_size):
    h, w = texture_size
    tri = Delaunay(base_shape)
    triangles = tri.simplices

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # (N, 2)

    pixel_triangle_ids = []
    pixel_bary_coords = []

    for p in pixels:
        tri_id = tri.find_simplex(p)
        if tri_id == -1:
            pixel_triangle_ids.append(None)
            pixel_bary_coords.append(None)
            continue
        tri_pts = base_shape[triangles[tri_id]]
        bary = barycentric_coords(tri_pts, p)
        pixel_triangle_ids.append(tri_id)
        pixel_bary_coords.append(bary)

    return triangles, pixel_triangle_ids, pixel_bary_coords

def symbolic_warp(params, base_shape, blendshapes, triangles, pixel_triangle_ids, pixel_bary_coords):
    num_points = base_shape.shape[0]

    # Базовая форма + линейная комбинация blendshapes + трансляция
    base_flat = ca.MX(base_shape.flatten())
    deltas = [ca.MX(s.flatten()) - base_flat for s in blendshapes]
    w = params[:-2]
    tx, ty = params[-2], params[-1]
    translation = ca.vertcat(*([tx, ty] * num_points))

    shape_vec = base_flat + ca.mtimes(ca.horzcat(*deltas), w) + translation
    shape = ca.reshape(shape_vec, num_points, 2)  # (num_points, 2)

    # Теперь проходим по всем пикселям
    pixel_coords = []

    for i in range(len(pixel_triangle_ids)):
        tri_id = pixel_triangle_ids[i]
        bary_coords = pixel_bary_coords[i]

        if tri_id is None:
            # Если пиксель вне формы — положим (0, 0)
            pixel_coords.append(0)
            pixel_coords.append(0)
            continue

        # Получаем индексы трёх вершин треугольника
        i1, i2, i3 = triangles[tri_id]

        # Берем сами вершины
        v1 = shape[i1, :]  # (2,)
        v2 = shape[i2, :]
        v3 = shape[i3, :]

        # Преобразуем barycentric coords в Casadi формат
        bary = ca.DM(bary_coords)

        # Линейная комбинация вершин по барицентрическим координатам
        p = bary[0]*v1 + bary[1]*v2 + bary[2]*v3  # (2,)

        # Сохраняем отдельно x и y координаты
        pixel_coords.append(p[0])
        pixel_coords.append(p[1])

    # Вернуть все координаты как длинный вектор (2*num_pixels,)
    return ca.vertcat(*pixel_coords)



