import os
import cv2
from pathlib import Path

def read_pts(pts_path):
    """Чтение .pts файла (в формате 'version: 1\nn_points: XX\n{ x y\n... }')"""
    with open(pts_path, 'r') as f:
        lines = f.readlines()
    start = lines.index('{\n') + 1
    end = lines.index('}\n')
    points = []
    for line in lines[start:end]:
        x, y = map(float, line.strip().split())
        points.append((x, y))
    return points

def read_dataset_from_pts(image_folder):
    image_folder = Path(image_folder)
    pts_paths = list(image_folder.glob("*.pts"))

    image_list = []
    landmarks_list = []

    for pts_path in pts_paths:
        img_path_jpg = pts_path.with_suffix(".jpg")
        img_path_png = pts_path.with_suffix(".png")

        if img_path_jpg.exists():
            img_path = img_path_jpg
        elif img_path_png.exists():
            img_path = img_path_png
        else:
            print(f"[!] Нет изображения для {pts_path.name}, пропускаю.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[!] Не удалось прочитать изображение: {img_path}")
            continue

        landmarks = read_pts(pts_path)

        image_list.append(img)
        landmarks_list.append(landmarks)

    return image_list, landmarks_list
