from Kabsch import kabsch
import pyvista
import numpy as np
from numpy.testing import assert_allclose

def test_copy():
	base = np.array([[5, 3, 6],
					[6, 9, 1],
					[1, 2, 0]])
	target = np.copy(base)
	R, t = kabsch(base, target)
	assert_allclose(R, np.eye(3), atol=1e-7)
	assert_allclose(t, np.zeros(3), atol=1e-7)

def test_translate_only():
	base = np.array([[5, 3, 6],
					[6, 9, 1],
					[1, 2, 0]])
	translate = [8, 3, 5]
	target = base + translate
	R, t = kabsch(base, target)
	assert_allclose(R, np.eye(3), atol=1e-7)
	assert_allclose(t, translate, atol=1e-7)

def test_rotate_only():
	base = np.array([[5, 3, 6],
					[6, 9, 1],
					[1, 2, 0]])
	
	angle = np.pi / 2
	R_test = np.array([[np.cos(angle), -np.sin(angle), 0],
						[np.sin(angle), np.cos(angle), 0],
						[0, 0, 1]])

	target = base @ R_test.T
	R, t = kabsch(base, target)
	assert_allclose(R, R_test, atol=1e-7)
	assert_allclose(t, np.zeros(3), atol=1e-7)

def test_translate_and_rotate():
	base = np.array([[5, 3, 6],
					[6, 9, 1],
					[1, 2, 0]])
	angle = np.pi / 2
	R_test = np.array([[np.cos(angle), -np.sin(angle), 0],
						[np.sin(angle), np.cos(angle), 0],
						[0, 0, 1]])
	translate = [8, 3, 5]
	target = base @ R_test.T + translate
	R, t = kabsch(base, target)
	assert_allclose(R, R_test, atol=1e-7)
	assert_allclose(t, translate, atol=1e-7)

if __name__ == "__main__":
	test_copy()
	test_translate_only()
	test_rotate_only()
	test_translate_and_rotate()
	print("Все тесты успешно прошли!")

