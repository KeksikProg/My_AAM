import casadi as ca
import numpy as np
from Optimizer import build_functions, optimize
from Read_pts import read_pts
from ShapeModel import shape_model
from Normilizer import normilize




if __name__ == "__main__":

    blandshapes = read_pts("dataset")

    target_model_shape = np.array(blandshapes[-1]).flatten()
    normalized = normilize(blandshapes[:-1])
    
    base_shape = np.mean(normalized, axis = 0)
    bland_shapes = normalized

    shape_params_sym, shape_linear_comb = shape_model(base_shape, bland_shapes)
    func = build_functions(shape_params_sym, shape_linear_comb, target_model_shape)

    init_params_shape = np.zeros(len(bland_shapes) + 2)
    final_params_shape = optimize(func, init_params_shape, target_model = target_model_shape)

    for par in range(len(final_params_shape[:-2])):
        print(f"Коэффициент блендшейпа {par + 1} равен: {final_params_shape[par]}")
    print(f"Трансляция по x равна: {final_params_shape[-2]}, по y равна: {final_params_shape[-1]}")
