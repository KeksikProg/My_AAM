import casadi as ca
import pyvista as pv
import numpy as np

# Загрузка моделей
cube1 = pv.read("FrameHeadEd.obj")
cube2 = pv.read("FrameHead.obj")

m1 = cube1.points.T
m2 = cube2.points.T

# Символьные параметры трансформации
tx, ty, tz, scale, rx, ry, rz = ca.MX.sym("tx"), ca.MX.sym("ty"), ca.MX.sym("tz"), ca.MX.sym("scale"), ca.MX.sym("rx"), ca.MX.sym("ry"), ca.MX.sym("rz")
params = ca.vertcat(tx, ty, tz, scale, rx, ry, rz)

# Исходные точки
m1_mx = ca.MX(m1)
m2_mx = ca.MX(m2)

def rotation_matrix(p):
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(p[4]), -ca.sin(p[4])),
        ca.horzcat(0, ca.sin(p[4]), ca.cos(p[4])))
    Ry = ca.vertcat(
        ca.horzcat(ca.cos(p[5]), 0, ca.sin(p[5])),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(p[5]), 0, ca.cos(p[5])))
    Rz = ca.vertcat(
        ca.horzcat(ca.cos(p[6]), -ca.sin(p[6]), 0),
        ca.horzcat(ca.sin(p[6]), ca.cos(p[6]), 0),
        ca.horzcat(0, 0, 1))
    return Rx @ Ry @ Rz

# Ошибка
R = rotation_matrix(params)
transformed = (R @ m2_mx + params[:3]) * params[3]
residuals = ca.vec(transformed - m1_mx)

# Функции
J = ca.jacobian(residuals, params)
JTJ = J.T @ J
JTr = J.T @ residuals

JTJ_fun = ca.Function("JTJ", [params], [JTJ])
JTr_fun = ca.Function("JTr", [params], [JTr])
loss_fun = ca.Function("loss", [params], [ca.dot(residuals, residuals) / m1.shape[1]])

# Начальное приближение
current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Визуализация начальная
plotter = pv.Plotter()
plotter.add_mesh(cube1, color='red', opacity=0.5, show_edges=True)
plotter.add_title("Итерация: 0, Ошибка: -")
plotter.show(interactive_update=True)


dump = 100

for i in range(100):
    JTJ_val = JTJ_fun(current).full()
    JTr_val = JTr_fun(current).full()
    JTJ_dump = JTJ_val + dump * np.eye(JTJ_val.shape[0])
    JTJ_invert = np.linalg.inv(JTJ_dump)



    delta =  (-JTJ_invert @ JTr_val).flatten()
    current += delta
    err = float(loss_fun(current))

    print(f"Итерация {i}, ошибка: {err:.6f}")
    
    # Визуализация
    R_current = rotation_matrix(ca.MX(current))
    R_np = ca.Function('R', [], [R_current])()['o0'].toarray()
    new_cube = pv.read("FrameHead.obj")
    new_cube.points = ((R_np @ m2 + current[:3].reshape(3, 1)) * current[3]).T
    plotter.clear()
    plotter.add_mesh(cube1, color='red', opacity=0.5, show_edges=True)
    plotter.add_mesh(new_cube, color='blue', opacity=0.5, show_edges=True)
    plotter.add_title(f"Iter: {i}, err: {err:.4f}")
    plotter.update()
    #-----------------

    if np.linalg.norm(delta) < 1e-6 or err < 1e-5:
        print("Достигнута сходимость.")
        break

plotter.show(interactive=True)
print("Оптимальные параметры:", current)
