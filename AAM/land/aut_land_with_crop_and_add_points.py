import os
import face_alignment
from skimage import io
import numpy as np

# Папки
image_dir = 'dataset'
pts_dir = 'landmarks'
os.makedirs(pts_dir, exist_ok=True)

# Предиктор
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

# Функция: вставка новых точек между существующими
def densify_contour(points, n_per_segment=1):
    dense = []
    N = len(points)
    for i in range(N):
        p1 = points[i]
        p2 = points[(i + 1) % N]  # замкнутый контур
        dense.append(p1)
        for j in range(1, n_per_segment + 1):
            alpha = j / (n_per_segment + 1)
            interp = (1 - alpha) * p1 + alpha * p2
            dense.append(interp)
    return np.array(dense)

# Сохранение .pts
def save_pts(path, points):
    with open(path, 'w') as f:
        f.write('version: 1\n')
        f.write(f'n_points: {len(points)}\n')
        f.write('{\n')
        for x, y in points:
            f.write(f'{x:.6f} {y:.6f}\n')
        f.write('}\n')

# Обработка изображений
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        image = io.imread(image_path)
        landmarks = fa.get_landmarks(image)

        if landmarks:
            pts = landmarks[0]

            # Извлекаем области
            left_eye = pts[36:42]
            right_eye = pts[42:48]
            mouth_outer = pts[48:60]
            mouth_inner = pts[60:68]

            # Увеличиваем плотность (можно менять n_per_segment)
            dense_left = densify_contour(left_eye, n_per_segment=4)
            dense_right = densify_contour(right_eye, n_per_segment=4)
            dense_outer = densify_contour(mouth_outer, n_per_segment=4)
            dense_inner = densify_contour(mouth_inner, n_per_segment=4)

            # Объединяем
            final_points = np.vstack([dense_left, dense_right, dense_outer, dense_inner])

            # Сохраняем
            pts_path = os.path.join(pts_dir, os.path.splitext(filename)[0] + '.pts')
            save_pts(pts_path, final_points)
            print(f'OK: {filename}, total points: {len(final_points)}')
        else:
            print(f'NO FACE: {filename}')
