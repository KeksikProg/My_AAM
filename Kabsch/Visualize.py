from Kabsch import kabsch
import pyvista as pv

headBase = pv.read("FrameHead.obj")
headTarget = pv.read("FrameHeadRT.obj")

# Визуализация начальная
plotter = pv.Plotter()
plotter.add_mesh(headBase, color='red', opacity=0.5, show_edges=True)
plotter.add_mesh(headTarget, color='blue', opacity=0.5, show_edges=True)
plotter.show(interactive_update=True)

x = input()

R, t = kabsch(headBase.points, headTarget.points)
headBase.points = (headBase.points @ R.T) + t

plotter.clear()
plotter.add_mesh(headBase, color='red', opacity=0.5, show_edges=True)
plotter.add_mesh(headTarget, color='blue', opacity=0.5, show_edges=True)
plotter.update()
plotter.show(interactive=True)