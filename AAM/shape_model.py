import numpy as np
from procrust import procrustes_analysis

def build_linear_shape_model(landmarks_list):
    mean_shape, aligned = procrustes_analysis(landmarks_list)
    n_samples, n_points, _ = aligned.shape
    
    # Центрированные данные (вариации вокруг средней формы)
    centered = aligned - mean_shape.reshape(1, n_points, 2)
    basis = centered.reshape(n_samples, -1).T  # (n_points*2, n_samples)
    
    return {
        'mean_shape': mean_shape,
        'basis': basis  # Базис для линейной комбинации
    }
