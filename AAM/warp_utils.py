import numpy as np
import cv2
from scipy.spatial import Delaunay

def warp_piecewise_affine(image, src_points, dst_points, triangles, output_size):
    h, w = output_size[1], output_size[0]
    warped_image = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)

    for tri_indices in triangles:
        src_tri = np.float32([src_points[i] for i in tri_indices])
        dst_tri = np.float32([dst_points[i] for i in tri_indices])

        x, y, rw, rh = cv2.boundingRect(dst_tri)
        if rw == 0 or rh == 0:
            continue

        x_end = min(x + rw, w)
        y_end = min(y + rh, h)
        x = max(x, 0)
        y = max(y, 0)
        rw = x_end - x
        rh = y_end - y
        if rw <= 0 or rh <= 0:
            continue

        # локальная маска и координаты
        mask = np.zeros((rh, rw), dtype=np.float32)
        dst_tri_shifted = dst_tri - [x, y]
        cv2.fillConvexPoly(mask, np.int32(dst_tri_shifted), 1.0)

        src_tri_shifted = src_tri
        warp_mat = cv2.getAffineTransform(src_tri, dst_tri)

        # варп только региона
        warped_patch = cv2.warpAffine(image.astype(np.float32), warp_mat, (w, h), flags=cv2.INTER_LINEAR)

        # вставка с накоплением весов
        warped_crop = warped_patch[y:y+rh, x:x+rw]
        warped_image[y:y+rh, x:x+rw] += warped_crop * mask
        weight_mask[y:y+rh, x:x+rw] += mask

    # нормализация по накопленным маскам
    weight_mask[weight_mask == 0] = 1.0
    warped_image /= weight_mask

    return np.clip(warped_image, 0, 255).astype(np.uint8)