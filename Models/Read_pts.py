import os
import numpy as np

def read_pts(path):
    def parse_pts_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        try:
            start = lines.index('{\n') + 1
            end = lines.index('}\n')
        except ValueError:
            # В случае, если скобки не в отдельных строках
            start = next(i for i, line in enumerate(lines) if line.strip() == '{') + 1
            end = next(i for i, line in enumerate(lines) if line.strip() == '}')
        points = [list(map(float, line.strip().split())) for line in lines[start:end]]
        return np.array(points)

    if os.path.isdir(path):
        blandshapes = []
        for filename in os.listdir(path):
            if filename.endswith(".pts"):
                file_path = os.path.join(path, filename)
                points = parse_pts_file(file_path)
                if points.size > 0:
                    blandshapes.append(points)
        return blandshapes
    elif os.path.isfile(path) and path.endswith(".pts"):
        return parse_pts_file(path)
    else:
        raise ValueError("Путь должен быть к .pts файлу или папке, содержащей .pts файлы")
