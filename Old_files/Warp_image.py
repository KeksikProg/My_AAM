import cv2
import numpy as np
from scipy.spatial import Delaunay

def warp_image_to_base(img, src_shape, dst_shape, texture_size):
    h, w = texture_size
    warped = np.zeros((h, w), dtype=np.uint8)
    tri = Delaunay(dst_shape)

    for triangle in tri.simplices:
        src_tri = np.float32([src_shape[i] for i in triangle])
        dst_tri = np.float32([dst_shape[i] for i in triangle])

        # Аффинное преобразование
        M = cv2.getAffineTransform(src_tri, dst_tri)

        # Маска для области треугольника
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri), 255)

        warped_triangle = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        warped[mask > 0] = warped_triangle[mask > 0]

    return warped
