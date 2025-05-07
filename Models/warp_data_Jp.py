from scipy.spatial import Delaunay
import numpy as np
import casadi as ca
from barucentr_coords import barycentric_coords

def prepare_warp_data(base_shape, texture_size):
    h, w = texture_size
    tri = Delaunay(base_shape)
    triangles = tri.simplices

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    pixel_triangle_ids = []
    pixel_bary_coords = []
    is_valid_pixel = []

    for p in pixels:
        tri_id = tri.find_simplex(p)
        if tri_id == -1:
            pixel_triangle_ids.append(None)
            pixel_bary_coords.append(None)
            is_valid_pixel.append(False)
        else:
            tri_pts = base_shape[triangles[tri_id]]
            bary = barycentric_coords(tri_pts, p)
            pixel_triangle_ids.append(tri_id)
            pixel_bary_coords.append(bary)
            is_valid_pixel.append(True)

    return triangles, pixel_triangle_ids, pixel_bary_coords, np.array(is_valid_pixel)

def symbolic_warp(params, base_shape, blendshapes, triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel):
    num_points = base_shape.shape[0]

    base_flat = ca.MX(base_shape.flatten())
    deltas = [ca.MX(s.flatten()) for s in blendshapes]
    w = params[:-3]           
    theta = params[-3]
    tx, ty = params[-2], params[-1]

    shape_vec = base_flat + ca.mtimes(ca.horzcat(*deltas), w)
    shape = ca.reshape(shape_vec, num_points, 2)

    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    R = ca.vertcat(
        ca.horzcat(cos_theta, -sin_theta),
        ca.horzcat(sin_theta,  cos_theta)
    )

    shape = (R @ shape.T).T + ca.vertcat(*([tx, ty] * num_points)).reshape((num_points, 2))

    bary_coords = ca.MX(pixel_bary_coords)
    tri_ids = ca.DM(pixel_triangle_ids).T.full().astype(int).flatten()
    i1 = triangles[tri_ids, 0]
    i2 = triangles[tri_ids, 1]
    i3 = triangles[tri_ids, 2]
    
    v1 = shape[i1, :]
    v2 = shape[i2, :]
    v3 = shape[i3, :]


    coords = (bary_coords[:, 0] @ ca.DM.ones(1, 2)) * v1 + \
         (bary_coords[:, 1] @ ca.DM.ones(1, 2)) * v2 + \
         (bary_coords[:, 2] @ ca.DM.ones(1, 2)) * v3

    return ca.reshape(coords, -1, 1)
