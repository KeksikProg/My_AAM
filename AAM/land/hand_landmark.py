import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from glob import glob

KEY_REGION_MAP = {
    '0': 'left_eye',
    '1': 'right_eye',
    '2': 'outer_lip',
    '3': 'inner_lip',
}

NUM_POINTS_MAP = {
    'left_eye': 20,
    'right_eye': 20,
    'outer_lip': 50,
    'inner_lip': 30,
}

class RegionDrawer:
    def __init__(self, image):
        self.image = image
        self.coords = []
        self.current_region = None
        self.result = {}
        self.ax = None
        self.fig = None
        self.lasso = None

    def onselect(self, verts):
        if self.current_region:
            self.coords = verts
            plt.close()

    def draw(self):
        while True:
            self.fig, self.ax = plt.subplots()

            # 🔄 Преобразуем BGR -> RGB
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.ax.imshow(rgb_image, interpolation='none')

            # 🔳 Открываем в полный экран
            try:
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()
            except Exception:
                pass

            plt.title("0-3 — выбрать область | Enter — сохранить | Esc — пропустить | Z — включить лупу")

            self.lasso = LassoSelector(self.ax, onselect=self.onselect)

            def key_handler(event):
                key = event.key
                if key in KEY_REGION_MAP:
                    self.current_region = KEY_REGION_MAP[key]
                    print(f"Выбранная область: {self.current_region}")
                elif key == 'enter':
                    plt.close()
                elif key == 'escape':
                    self.result = None
                    plt.close()
                elif key == '4':
                    try:
                        toolbar = plt.get_current_fig_manager().toolbar
                        if toolbar.mode == 'zoom rect':
                            toolbar.zoom()  # выключить зум
                            print("Лупа отключена")
                        else:
                            toolbar.zoom()  # включить зум
                            print("Лупа включена")
                    except Exception as e:
                        print(f"Ошибка активации лупы: {e}")

            self.fig.canvas.mpl_connect('key_press_event', key_handler)
            plt.show()
            if self.coords and self.current_region:
                points = self.resample_contour(self.coords, NUM_POINTS_MAP[self.current_region])
                self.result[self.current_region] = points
                self.coords = []
                self.current_region = None
            else:
                break

        return self.result

    def resample_contour(self, contour, num_points):
        contour = np.array(contour)
        dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        cumdist = np.insert(np.cumsum(dists), 0, 0)
        even_dists = np.linspace(0, cumdist[-1], num_points)

        new_points = []
        for d in even_dists:
            idx = np.searchsorted(cumdist, d)
            if idx == 0:
                interp = contour[0]
            elif idx >= len(contour):
                interp = contour[-1]
            else:
                p1, p2 = contour[idx - 1], contour[idx]
                ratio = (d - cumdist[idx - 1]) / (cumdist[idx] - cumdist[idx - 1] + 1e-8)
                interp = p1 + (p2 - p1) * ratio
            new_points.append(interp)
        return np.array(new_points)

def save_pts(path, points_dict):
    all_pts = np.vstack(list(points_dict.values()))
    with open(path, 'w') as f:
        f.write(f'version: 1\nn_points: {len(all_pts)}\n{{\n')
        for pt in all_pts:
            f.write(f'{pt[0]:.3f} {pt[1]:.3f}\n')
        f.write('}\n')

def main():
    folder = input("Введи путь к папке с изображениями: ").strip()
    images = sorted(glob(os.path.join(folder, "*.png")) + glob(os.path.join(folder, "*.jpg")))

    output_dir = os.path.join(folder, "pts")
    os.makedirs(output_dir, exist_ok=True)

    for img_path in images:
        print(f"\n--- Разметка: {os.path.basename(img_path)} ---")
        image = cv2.imread(img_path)
        drawer = RegionDrawer(image)
        result = drawer.draw()
        if result:
            name = os.path.splitext(os.path.basename(img_path))[0]
            save_pts(os.path.join(output_dir, f"{name}.pts"), result)
            print("Сохранено.")
        else:
            print("Пропущено.")

if __name__ == "__main__":
    main()
