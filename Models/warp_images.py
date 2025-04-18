from scipy.spatial import Delaunay
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np

def warp_images_to_mean_shape(image_list, landmark_list, base_shape):
    mean_shape = base_shape
    warped_image_list = []
    piecewise_affine_transform = PiecewiseAffineTransform()

    # Фиксируем размер output — по первому изображению
    target_shape = image_list[0].shape

    for aligned_image, landmarks in zip(image_list, landmark_list):
        piecewise_affine_transform.estimate(mean_shape, landmarks)
        warped_image = warp(aligned_image, piecewise_affine_transform, output_shape=target_shape) * 255
        warped_image_list.append(warped_image.astype(np.uint8))  # на всякий, чтобы не float

    return warped_image_list