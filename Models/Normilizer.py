from Kabsch import kabsch
import numpy as np

def normilize(shapes):
    base = shapes[0]
    normalized = []
    for shape in shapes:
        R, t = kabsch(shape, base)
        aligned = np.array(shape) @ R + t
        normalized.append(aligned)
    return normalized

def center_shape(shape, texture_size):
    shape = shape.copy()
    center_texture = np.array(texture_size[::-1]) / 2
    center_shape = np.mean(shape, axis=0)
    shift = center_texture - center_shape
    return shape + shift
