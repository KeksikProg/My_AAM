import cv2
import os

def extract_frames(video_path, output_dir, every_n=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frame_path = os.path.join(output_dir, f"{saved_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Готово. Сохранено {saved_idx} кадров в {output_dir}")

# Пример использования:
extract_frames("r512.mp4", "dataset", every_n=3)  # каждый 5-й кадр
