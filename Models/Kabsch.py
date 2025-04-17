import numpy as np

def kabsch(base, target):
    # Вычисление центроид
    centroid_base = np.mean(base, axis=0)
    centroid_target = np.mean(target, axis=0)

    #Центрирование к началу координат
    target_centered = target - centroid_target
    base_centered = base - centroid_base
    
    #Ковариационная матрица
    H = base_centered.T @ target_centered
    # Синугулярное разложение
    U, S, Vt = np.linalg.svd(H)
    #Матрица поворота
    R = Vt.T @ U.T
    
    # Проверка деетерменанта
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    # Вектор смещения
    t = centroid_target - R @ centroid_base
    return R, t
