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