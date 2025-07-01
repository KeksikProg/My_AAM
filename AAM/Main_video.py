import casadi as ca
import numpy as np
from Optimizer import optimize
from Normilizer import normilize
from read_dataset import read_dataset_from_pts
from build_func import build_func
import matplotlib.pyplot as plt
from warp_utils import warp_piecewise_affine
from scipy.spatial import Delaunay
import cv2
import os


TEXTURE_SIZE =  (512, 512) 
SIZE_DATASET = -1
TARGET_INDEX = 1

out_dir = "output_frames"
os.makedirs(out_dir, exist_ok=True)
video_path = "output_video.avi"
fps = 8

frames = []

if __name__ == "__main__":
    images, shapes = read_dataset_from_pts("dataset_teach")
    print("Датасет считался")

    normalized_shapes = normilize(shapes[:SIZE_DATASET])
    base_shape = np.mean(normalized_shapes, axis = 0)
    bland_shapes = np.array(normalized_shapes)
    bland_shapes_delt = np.array([s - base_shape for s in normalized_shapes])
    print("Все нормализовалось, блендшейпы формы посчитались")

    target_shape = normalized_shapes[0]

    tri = Delaunay(base_shape)
    triangles = tri.simplices
    print("Триангуляция прошла")

    # Варпинг изображений
    warped_images = []
    for img, shape in zip(images[:SIZE_DATASET], shapes[:SIZE_DATASET]):
        warped = warp_piecewise_affine(
            img,              
            shape,            
            base_shape,       
            triangles,        
            output_size=TEXTURE_SIZE
        )
        warped_images.append(warped)

    warp_vectors = np.array([img.flatten() for img in warped_images])
    warp_vectors = warp_vectors.astype(np.float32) / 255.0
    mean_texture = np.mean(warp_vectors, axis = 0)
    appearance_deltas = warp_vectors - mean_texture
    print("Блендшейпы текстуры посчитались")

    print("Целевое изображение определено")
    aam_dataset, aam_dataset_shapes = read_dataset_from_pts("dataset")
    aam_normalized_shapes = normilize(aam_dataset_shapes)
    for idx, (target_image, shape) in enumerate(zip(aam_dataset, aam_dataset_shapes)):
        print(f"\nОбработка изображения {idx}")
    
        func = build_func(mean_texture, appearance_deltas, target_image, TEXTURE_SIZE, base_shape, bland_shapes_delt, triangles)
        print("Функция построена")
    
        num_app_params = len(appearance_deltas)
        num_shape_params = len(bland_shapes_delt) + 2
    
        init_shape = base_shape + (bland_shapes_delt.T @ init_params[num_app_params:-2]).T
        init_shape += np.array([init_params[-2], init_params[-1]])
    
        print("Параметры иницилизированы")
    
        print("Пошла оптимизация")
        final_params = optimize(func, init_params, 20)
        print("Оптимизация закончена")

        rec_shape = base_shape + (bland_shapes_delt.T @ final_params[num_app_params:-2]).T
        rec_shape += np.array([final_params[-2], final_params[-1]])
    
        tri_target = Delaunay(rec_shape)
        triangles_rec = tri_target.simplices
    
        tri_init = Delaunay(init_shape)
        triangles_init = tri_init.simplices
    
        target_texture = warp_piecewise_affine((mean_texture + appearance_deltas.T @ final_params[:num_app_params]).reshape(TEXTURE_SIZE)*255.0, base_shape, rec_shape, triangles_rec, TEXTURE_SIZE)
        init_texture = warp_piecewise_affine(mean_texture.reshape(TEXTURE_SIZE)*255.0, base_shape, init_shape, triangles_init, TEXTURE_SIZE)
    
        or_shape = aam_normalized_shapes[idx]
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))

        axs[0].imshow(target_image, cmap='gray')
        axs[0].scatter(init_shape[:, 0], init_shape[:, 1], c='lime', s=5)
        axs[0].set_title("Начальная форма")
        
        axs[1].imshow(target_image, cmap='gray')
        axs[1].scatter(rec_shape[:, 0], rec_shape[:, 1], c='red', s=5)
        axs[1].set_title("Восстановленная форма")
        
        axs[2].imshow(target_image, cmap='gray')
        axs[2].scatter(or_shape[:, 0], or_shape[:, 1], c='red', s=5)
        axs[2].set_title("Оригинальная форма")
        
        frame_path = os.path.join(out_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        frames.append(frame_path)

    frame_example = cv2.imread(frames[0])
    h, w = frame_example.shape[:2]
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))


    for f in frames:
        frame = cv2.imread(f)
        video_writer.write(frame)
    video_writer.release()

    print(f"\nВидео сохранено: {video_path}")



