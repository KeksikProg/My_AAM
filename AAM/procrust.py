import numpy as np

def procrustes_analysis(landmarks_list, max_iter=100, tolerance=1e-6):
    n_samples = len(landmarks_list)
    n_points = np.array(landmarks_list[0]).shape[0]
    aligned = np.zeros((n_samples, n_points, 2))
    
    for i in range(n_samples):
        aligned[i] = landmarks_list[i] - np.mean(landmarks_list[i], axis=0)
    
    mean_shape = aligned[0].copy()
    
    for _ in range(max_iter):
        for i in range(n_samples):
            A = aligned[i].T @ mean_shape
            U, _, Vt = np.linalg.svd(A)
            R = Vt.T @ U.T
            aligned[i] = aligned[i] @ R
        
        new_mean = np.mean(aligned, axis=0)
        new_mean -= np.mean(new_mean, axis=0)
        scale = np.linalg.norm(new_mean)
        if scale:
            new_mean /= scale
        
        if np.linalg.norm(new_mean - mean_shape) < tolerance:
            break
        mean_shape = new_mean
    
    return mean_shape, aligned

