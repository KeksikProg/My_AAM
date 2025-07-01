import os
import cv2
import numpy as np
from read_dataset import read_dataset_from_pts

input_dir = "dataset_teach64"
output_dir = "dataset_teach32"
os.makedirs(output_dir, exist_ok=True)

# Чтение
images, shapes = read_dataset_from_pts(input_dir)

for i, (img, shape) in enumerate(zip(images, shapes)):
    # Уменьшаем изображение
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # Масштабируем shape
    scaled_shape = np.array(shape) / 2.0


    # Сохраняем изображение
    img_out_path = os.path.join(output_dir, f"{i:03d}.png")
    cv2.imwrite(img_out_path, resized_img)

    # Сохраняем .pts
    pts_out_path = os.path.join(output_dir, f"{i:03d}.pts")
    with open(pts_out_path, "w") as f:
        f.write("version: 1\nn_points: {}\n{{\n".format(len(scaled_shape)))
        for x, y in scaled_shape:
            f.write(f"{x:.6f} {y:.6f}\n")
        f.write("}\n")

print(f"Сохранено {len(images)} файлов в {output_dir}")
