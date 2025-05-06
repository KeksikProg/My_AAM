import numpy as np
from Optimizer import optimize
from Image_grad import image_grad
from read_dataset import read_dataset_from_pts
import casadi as ca
from Optimizer import optimize
from Normilizer import normilize
from read_dataset import read_dataset_from_pts
from warp_data_Jp import prepare_warp_data
from build_func import build_func
import matplotlib.pyplot as plt
from line_interpolate import bilinear_interpolate

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

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("Newton step test passed")


def test_image_grad():
    print("🔍 Тест градиента изображения...")
    texture_size = (3, 3)
    test_img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    expected_dx = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.float32)

    expected_dy = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    grad_fn = image_grad(texture_size)
    dx, dy = grad_fn(test_img)

    dx = dx.full()
    dy = dy.full()

    assert np.allclose(dx, expected_dx), f"dx mismatch\n{dx}\n!=\n{expected_dx}"
    assert np.allclose(dy, expected_dy), f"dy mismatch\n{dy}\n!=\n{expected_dy}"
    print("Image gradient test passed")


def test_identity_fit():
    # Мини-датасет из 1 картинки
    images, shapes = read_dataset_from_pts("dataset")
    img = images[0]
    shape = shapes[0]

    normalized = normilize([shape])
    base_shape = center_shape(normalized[0], (16, 16))
    bland_shapes = np.array(normalized)

    warp = warp_images_to_mean_shape([img], [shape], base_shape, (16, 16))
    mean_texture = warp[0].flatten() / 255.0
    appearance_deltas = np.zeros((1, mean_texture.size))  # Только 1 текстурный параметр

    triangles, pix_ids, bary, valid = prepare_warp_data(base_shape, (16, 16))
    target_tex = warp[0]

    func = build_func(base_shape, bland_shapes, mean_texture, appearance_deltas, triangles, pix_ids, bary, valid, target_tex, shape, (16, 16))
    init_params = np.zeros(len(bland_shapes) + 2 + 1)

    final = optimize(func, init_params, max_iter=50)
    loss = func(final)["loss"]
    print(f"Final loss = {loss}")
    assert loss < 1e-6, "Too high loss on identity fit"
    print("Identity fit test passed")

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

    print("test_prepare_warp_data passed.")

def test_bilinear_interpolate():
    # Подготовка изображения 5x5 с линейно возрастающими значениями
    img_np = np.arange(25).reshape(5, 5).astype(np.float32)
    img_flat = img_np.flatten()
    img = ca.MX(img_flat)

    # Тестовые координаты: между (1,1) и (2,2)
    coords_np = np.array([1.5, 1.5])  # x=1.5, y=1.5
    coords = ca.MX(coords_np)

    # Размер изображения
    texture_size = (5, 5)

    # Функция на CasADi
    interp_val = bilinear_interpolate(img, coords, texture_size)

    # Ожидаемое значение (через NumPy)
    def numpy_bilinear(img, x, y):
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, img.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, img.shape[0] - 1)

        Ia = img[y0, x0]
        Ib = img[y1, x0]
        Ic = img[y0, x1]
        Id = img[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    expected = numpy_bilinear(img_np, 1.5, 1.5)

    # Проверка
    f = ca.Function('f', [], [interp_val], [], ['out'])
    result = f()['out'].full().item()
    
    assert np.isclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"
    print("test_bilinear_interpolate passed.")

if __name__ == "__main__":
    # test_newton_step()
    # test_identity_fit()
    # test_image_grad()
    #test_prepare_warp_data()
    test_bilinear_interpolate()
    


