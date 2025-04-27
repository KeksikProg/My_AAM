import numpy as np

def barycentric_coords(tri_pts, p):
    T = np.array([
        [tri_pts[0][0] - tri_pts[2][0], tri_pts[1][0] - tri_pts[2][0]],
        [tri_pts[0][1] - tri_pts[2][1], tri_pts[1][1] - tri_pts[2][1]],
    ])
    v = np.array([p[0] - tri_pts[2][0], p[1] - tri_pts[2][1]])
    w = np.linalg.solve(T, v)
    alpha, beta = w
    gamma = 1 - alpha - beta
    return [alpha, beta, gamma]