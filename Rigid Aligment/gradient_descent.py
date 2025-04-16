import casadi
import pyvista as pv
import numpy as np

# Создаем два куба
#cube1 = pv.Cube()
#cube2 = pv.Cube((6, 5, 7))

cube1 = pv.read("FrameHeadEd.obj")
cube2 = pv.read("FrameHead.obj")

print(cube1.points.shape)

m1 = cube1.points.T 
m2 = cube2.points.T 


tx = casadi.MX.sym("tx")
ty = casadi.MX.sym("ty")
tz = casadi.MX.sym("tz")
scale = casadi.MX.sym("scale")
rx = casadi.MX.sym("rx")
ry = casadi.MX.sym("ry")
rz = casadi.MX.sym("rz")
translate = casadi.vertcat(tx, ty, tz, scale, rx, ry, rz)

m1_mx = casadi.MX(m1)
m2_mx = casadi.MX(m2)

def rotation_matrix(translate):
    Rx = casadi.vertcat(
        casadi.horzcat(1, 0, 0),
        casadi.horzcat(0, casadi.cos(translate[4]), -casadi.sin(translate[4])),
        casadi.horzcat(0, casadi.sin(translate[4]), casadi.cos(translate[4])))
        
    Ry = casadi.vertcat(
        casadi.horzcat(casadi.cos(translate[5]), 0, casadi.sin(translate[5])),
        casadi.horzcat(0, 1, 0),
        casadi.horzcat(-casadi.sin(translate[5]), 0, casadi.cos(translate[5])))

    Rz = casadi.vertcat(
        casadi.horzcat(casadi.cos(translate[6]), -casadi.sin(translate[6]), 0),
        casadi.horzcat(casadi.sin(translate[6]), casadi.cos(translate[6]), 0),
        casadi.horzcat(0, 0, 1))

    return Rx @ Ry @ Rz

R = rotation_matrix(translate)

trans = (R @ m2_mx  + translate[:3]) * (translate[3])
err = trans - m1_mx
loss = casadi.sumsqr(err) / m1.shape[1]

f_loss = casadi.Function("loss", [translate], [loss])
grad_loss = casadi.gradient(loss, translate)
f_grad = casadi.Function("f_grad", [translate], [grad_loss])

current_trans = np.array([0.0, 0.0, 0.0, 1, 0, 0, 0])

# Для отображения моделей
plotter = pv.Plotter()
plotter.add_mesh(cube1, color='red', opacity=0.5, show_edges=True)
plotter.add_title("Итерация: 0, Ошибка: -")
plotter.show(interactive_update=True)
plotter.view_yz()


# Реализация градиентного спуска
for iterate in range(20000):
    # Вычисляем градиент и обновляем трансляцию
    grad = f_grad(current_trans).full().flatten()
    current_trans += -0.0001 * grad
    
    # Вычисляем текущую ошибку
    current_loss = float(f_loss(current_trans))
    
    R_current = rotation_matrix(casadi.MX(current_trans))
    R_current_np = casadi.Function('R_current', [], [R_current])()['o0'].toarray()

    if iterate % 5000 == 0:
        # Создаем новый куб с обновленными координатами
        new_cube = pv.read("FrameHead.obj")
        new_cube.points = ((R_current_np @ m2 + current_trans[:3].reshape(3, 1)) * current_trans[3]).T
        
        # Очищаем предыдущий куб и добавляем новый
        plotter.clear()
        plotter.add_mesh(cube1, color='red', opacity=0.5, show_edges=True)
        plotter.add_mesh(new_cube, color='blue', opacity=0.5, show_edges=True)
        plotter.add_title(f"Iter: {iterate}, err: {current_loss:.4f}")
        plotter.update()
    
    if iterate % 10 == 0:
        print(f"Итерация {iterate}, Ошибка: {current_loss}")

    if current_loss < 1e-5:
        break

plotter.show(interactive=True)
print("Оптимальный сдвиг:", current_trans)