import numpy as np

oneslice = np.genfromtxt('2d/points_1slice.csv', delimiter=',', skip_header=1)
file = open('3d/3d_points', 'w')
file.write('x, y, z\n')
for z in np.linspace(0.3, 30, 100):
    for point in oneslice:
        x = point[0]
        y = point[1]
        file.write(f'{x}, {y}, {z}\n')
    
file.close()