import numpy as np
from Optimizer import optimize
from Image_grad import image_grad
from read_dataset import read_dataset_from_pts, read_pts
import casadi as ca
from Normilizer import normilize
from warp_data_Jp import prepare_warp_data, symbolic_warp
from build_func import build_func
import matplotlib.pyplot as plt
from barucentr_coords import barycentric_coords
import tempfile
import cv2
from pathlib import Path
from scipy.spatial import Delaunay
from warp_utils import warp_piecewise_affine


def test_newton_step():
    JTJ = np.array([[2.0, 0.0], [0.0, 4.0]])
    JTr = np.array([[-2.0], [-8.0]])
    init = np.array([0.0, 0.0])

    def build_fn(_):
        return {
            "JTJ": JTJ,
            "JTr": JTr,
            "loss": 1.0
        }

    result = optimize(build_fn, init, max_iter=1, damping=0.0)
    expected = np.linalg.solve(JTJ, -JTr).flatten()

    assert np.allclose(result, expected), f"Ожидания {expected}, результат {result}"
    print("Шаг Ньютона. Тест сдан")
def test_image_grad():
    print("Тест градиента изображения")
    texture_size = (3, 3)
    test_img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    expected_dx = np.array([
        [ 1. ,  1. , -1. ],
        [ 2.5,  1. , -2.5],
        [ 4. ,  1. , -4. ]
    ], dtype=np.float32)

    expected_dy = np.array([
        [ 2. ,  2.5,  3. ],
        [ 3. ,  3. ,  3. ],
        [-2. , -2.5, -3. ]
    ], dtype=np.float32)

    grad_fn = image_grad(texture_size)
    dx, dy = grad_fn(test_img)

    dx = dx.full()
    dy = dy.full()

    assert np.allclose(dx, expected_dx), f"dx промах\n{dx}\n!=\n{expected_dx}"
    assert np.allclose(dy, expected_dy), f"dy промах\n{dy}\n!=\n{expected_dy}"
    print("Градиент изображения. Тест сдан")
def test_prepare_warp_data():
    # Простой треугольник
    base_shape = np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [30.0, 40.0]
    ])
    texture_size = (50, 60)

    triangles, pixel_triangle_ids, pixel_bary_coords, is_valid_pixel = prepare_warp_data(base_shape, texture_size)

    assert triangles.shape[1] == 3  # должны быть треугольники
    assert len(is_valid_pixel) == texture_size[0] * texture_size[1]
    assert len(pixel_triangle_ids) == len(is_valid_pixel)
    assert len(pixel_bary_coords) == len(is_valid_pixel)

    for i, valid in enumerate(is_valid_pixel):
        if valid:
            bary = pixel_bary_coords[i]
            assert bary is not None
            assert np.isclose(sum(bary), 1.0, atol=1e-3)
            assert np.all(np.array(bary) >= -1e-3)  # допускаем численные погрешности
        else:
            assert pixel_triangle_ids[i] is None
            assert pixel_bary_coords[i] is None

    print("Подготовка данных перед варпом. Тест сдан")
def test_barycentric_coords():
    tri = [(0, 0), (1, 0), (0, 1)]

    # Центроид треугольника
    p_inside = (1/3, 1/3)
    bary = barycentric_coords(tri, p_inside)
    assert np.allclose(sum(bary), 1.0), "Сумма должна быть 1"
    assert all(0 <= b <= 1 for b in bary), "Координаты должны быть в [0, 1]"

    # Вершина A
    assert np.allclose(barycentric_coords(tri, tri[0]), [1, 0, 0])

    # Вне треугольника
    p_outside = (2, 2)
    bary_out = barycentric_coords(tri, p_outside)
    assert not all(0 <= b <= 1 for b in bary_out), "Хотя бы одна координата должна быть вне [0,1]"
    print("Проверка рассчета барицентрических координат. Тест сдан")
def test_normilize():
    base = np.array([[0.0, 0.0], [1.0, 0.0]])
    rotated = np.array([[0.0, 0.0], [0.0, 1.0]])
    translated = np.array([[1.0, 1.0], [2.0, 1.0]])

    shapes = [base, rotated, translated]
    normalized = normilize(shapes)

    for shape in normalized:
        d_base = np.linalg.norm(base[1] - base[0])
        d_shape = np.linalg.norm(shape[1] - shape[0])
        assert np.allclose(d_shape, d_base, atol=1e-6), f"Длины не совпадают: {d_shape} != {d_base}"
    
    print("Нормализация форм. Тест сдан")
def test_read_pts():
    content = """version: 1
        n_points: 3
        {
        0.0 0.0
        1.0 0.0
        1.0 1.0
        }

    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pts', delete=False) as f:
        f.write(content)
        f.flush()
        pts_path = Path(f.name)
    
    points = read_pts(pts_path)
    assert points == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    print("Чтение .pts. Тест сдан")
def test_read_dataset_from_pts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # создаём .pts
        pts_path = tmpdir / "sample.pts"
        pts_path.write_text("""version: 1
            n_points: 2
            {
            10.0 20.0
            30.0 40.0
            }

            """)
        # создаём изображение .png
        img_path = tmpdir / "sample.png"
        img = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.imwrite(str(img_path), img)

        images, landmarks = read_dataset_from_pts(tmpdir)

        assert len(images) == 1
        assert images[0].shape == (100, 100)
        assert landmarks[0] == [(10.0, 20.0), (30.0, 40.0)]
        print("Чтение датасета из .pts. Тест сдан")
def test_warp_piecewise_affine_identity():
    img = np.full((50, 50), 128, dtype=np.uint8)

    src_points = [(10, 10), (40, 10), (25, 40)]
    dst_points = [(10, 10), (40, 10), (25, 40)]
    triangles = Delaunay(np.array(src_points)).simplices

    warped = warp_piecewise_affine(img, src_points, dst_points, triangles, output_size=(50, 50))

    # Создаём маску, где применялось преобразование
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), 1)

    # Проверяем только в области маски
    diff = np.abs(warped.astype(np.int16) - img.astype(np.int16))
    assert np.max(diff[mask == 1]) <= 1, "Погрешность в области треугольника превышает допустимую"
    print("Piecewise affine warp (identity). Тест сдан")
def test_warp_piecewise_affine_small_deformation():
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(img, (25, 25), 10, 255, -1)  # круглое пятно в центре

    src_points = [(10, 10), (40, 10), (25, 40)]
    dst_points = [(10, 10), (42, 10), (25, 40)]  # сдвигаем правую вершину на 2 пикселя
    triangles = Delaunay(np.array(src_points)).simplices

    warped = warp_piecewise_affine(img, src_points, dst_points, triangles, output_size=(50, 50))

    # Изображения должны отличаться внутри треугольника
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), 1)

    diff = np.abs(warped.astype(np.int16) - img.astype(np.int16))
    changed = np.count_nonzero(diff[mask == 1])

    assert changed > 0, "Деформация не изменила изображение в области треугольника"
    print("Piecewise affine warp (деформация). Тест сдан")


if __name__ == "__main__":
    test_newton_step()
    test_image_grad()
    test_prepare_warp_data()
    test_barycentric_coords()
    test_normilize()
    test_read_pts()
    test_read_dataset_from_pts()
    test_warp_piecewise_affine_identity()
    test_warp_piecewise_affine_small_deformation()
