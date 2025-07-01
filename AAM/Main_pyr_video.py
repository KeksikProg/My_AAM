import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from read_dataset import read_dataset_from_pts
from Normilizer import normilize
from build_func import build_func
from Optimizer import optimize
from warp_utils import warp_piecewise_affine
from scipy.spatial import Delaunay

LEVELS = [32, 64, 128, 256]
EPOCHS = 10
SIZE_DATASET = -1
out_dir = "output_frames"
video_path = "output_video.avi"
fps = 8
os.makedirs(out_dir, exist_ok=True)
frames = []

# === Используем изображения из dataset (тест), оптимизируемся на dataset_teach ===
test_images, test_shapes = read_dataset_from_pts("dataset256")
test_normalized_shapes = normilize(test_shapes)

for idx, (target_image_256, target_shape_orig) in enumerate(zip(test_images, test_shapes)):
    print(f"\n===== КАДР {idx} =====")
    init_params = None

    for lvl in LEVELS:
        print(f"-- УРОВЕНЬ {lvl} --")

        teach_images, teach_shapes = read_dataset_from_pts(f"dataset_teach{lvl}")
        normalized_shapes = normilize(teach_shapes[:SIZE_DATASET])
        base_shape = np.mean(normalized_shapes, axis=0)
        bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])

        tri = Delaunay(base_shape)
        triangles = tri.simplices

        warped_images = [
            warp_piecewise_affine(img, shape, base_shape, triangles, output_size=(lvl, lvl))
            for img, shape in zip(teach_images[:SIZE_DATASET], teach_shapes[:SIZE_DATASET])
        ]
        warp_vectors = np.array([img.flatten() for img in warped_images]).astype(np.float32) / 255.0
        mean_texture = np.mean(warp_vectors, axis=0)
        appearance_deltas = warp_vectors - mean_texture

        # Тестовая цель уменьшается
        current_target_image = cv2.resize(target_image_256, (lvl, lvl), interpolation=cv2.INTER_AREA)
        if lvl == 32:
            current_target_image = cv2.GaussianBlur(current_target_image, (3, 3), sigmaX=2)
        elif lvl == 64:
            current_target_image = cv2.GaussianBlur(current_target_image, (3, 3), sigmaX=1)


        func = build_func(mean_texture, appearance_deltas, current_target_image,
                          (lvl, lvl), base_shape, bland_shapes_delt, triangles)

        num_app_params = appearance_deltas.shape[0]
        num_shape_params = len(bland_shapes_delt) + 2
        if init_params is None:
            init_params = np.zeros(num_app_params + num_shape_params)

        if lvl == 32:
            init_params = optimize(func, init_params, 15)
        elif lvl == 64:
            init_params = optimize(func, init_params, 15)
        elif lvl == 128:
            init_params = optimize(func, init_params, 10)
        elif lvl == 256:
            init_params = optimize(func, init_params, 7)

    # === Построение финальной формы (256) ===
    final_params = init_params
    rec_shape = base_shape + (bland_shapes_delt.T @ final_params[num_app_params:-2]).T
    rec_shape += np.array([final_params[-2], final_params[-1]])

    init_shape = base_shape + (bland_shapes_delt.T @ np.zeros(num_shape_params - 2)).T

    rec_tri = Delaunay(rec_shape)
    init_tri = Delaunay(init_shape)

    mean_texture_256 = mean_texture.reshape(256, 256) * 255.0
    rec_texture = warp_piecewise_affine(mean_texture_256, base_shape, rec_shape, rec_tri.simplices, (256, 256))
    init_texture = warp_piecewise_affine(mean_texture_256, base_shape, init_shape, init_tri.simplices, (256, 256))

    or_shape = normilize([target_shape_orig])[0]

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs[0].imshow(target_image_256, cmap='gray')
    axs[0].scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=5)
    axs[0].set_title("Начальная форма")

    axs[1].imshow(target_image_256, cmap='gray')
    axs[1].scatter(rec_shape[:, 0], rec_shape[:, 1], c='red', s=5)
    axs[1].set_title("Восстановленная форма")

    axs[2].imshow(target_image_256, cmap='gray')
    axs[2].scatter(or_shape[:, 0], or_shape[:, 1], c='red', s=5)
    axs[2].set_title("Оригинальная форма")

    frame_path = os.path.join(out_dir, f"frame_{idx:03d}.png")
    plt.savefig(frame_path)
    plt.close()
    frames.append(frame_path)

# === Видео ===
frame_example = cv2.imread(frames[0])
h, w = frame_example.shape[:2]
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
for f in frames:
    video_writer.write(cv2.imread(f))
video_writer.release()

print(f"\nВидео сохранено: {video_path}")
