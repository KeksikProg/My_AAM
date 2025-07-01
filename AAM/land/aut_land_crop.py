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

# Функция для сохранения .pts только нужных точек
def save_filtered_pts(path, points):
    indices = list(range(36, 48)) + list(range(48, 60)) + list(range(60, 68))
    filtered_points = points[indices]
    with open(path, 'w') as f:
        f.write('version: 1\n')
        f.write(f'n_points: {len(filtered_points)}\n')
        f.write('{\n')
        for x, y in filtered_points:
            f.write(f'{x:.6f} {y:.6f}\n')
        f.write('}\n')

# Обработка
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = io.imread(image_path)
        landmarks = fa.get_landmarks(image)

        if landmarks:
            pts_path = os.path.join(pts_dir, os.path.splitext(filename)[0] + '.pts')
            save_filtered_pts(pts_path, landmarks[0])
            print(f'OK: {filename}')
        else:
            print(f'NO FACE: {filename}')
