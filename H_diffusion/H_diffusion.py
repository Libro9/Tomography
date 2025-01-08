import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import jit

print("H diffusion solver")
start = time.time()

raw_grain = np.genfromtxt('H_diffusion/NodeType3.csv', delimiter=';', skip_header=True)

# Define the ranges and steps
x_start, x_end, x_step = np.min(raw_grain[:, 0]), np.max(raw_grain[:, 0]), 0.25
y_start, y_end, y_step = np.min(raw_grain[:, 1]), np.max(raw_grain[:, 1]), 0.25

# Create arrays for the x and y indices based on the start, end, and step
x_indices = np.arange(x_start, x_end + x_step, x_step)
y_indices = np.arange(y_start, y_end + y_step, y_step)

# Create an empty array with the desired shape
grain_orientation = np.full((len(x_indices), len(y_indices)), -1)  # Assuming -1 for unfilled values

# Iterate over the raw_grain to populate the new array
for entry in raw_grain:
    x_pos, y_pos, feature_id = entry
    # Find the corresponding x and y index
    x_idx = np.where(x_indices == x_pos)[0][0]  # Find index in x_indices
    y_idx = np.where(y_indices == y_pos)[0][0]  # Find index in y_indices
    
    # Assign the FeatureId to the grain_orientation at the correct position
    grain_orientation[x_idx, y_idx] = feature_id

# Now grain_orientation isttopulated with the FeatureIds at the correct (x, y) positions.

grain_orientation = grain_orientation[:, 11:]

plt.clf()
plt.title(f"Grain orientation")
plt.xlabel("x")
plt.ylabel("y")
plt.imshow(grain_orientation.T, cmap=plt.cm.jet, vmin=0, vmax=100, origin='lower', 
           extent=[x_indices[0] - 0.125, x_indices[-1] + 0.125, y_indices[11] - 0.125, y_indices[-1] + 0.125])
plt.colorbar()
plt.show()

max_iter_time = 150
alpha = 1
delta_t = (x_step ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (x_step ** 2)

# Initialize solution: the grid of u(t, i, j)
H = np.empty((max_iter_time, int(((x_end - x_start) / x_step) + 1), int(((y_end - y_start) / y_step) + 1)))
H = H[:, :, 11:]

# Initial condition everywhere inside the grid
H_initial = 0

# Boundary conditions
H_top = 0.0
H_left = 0.0
H_bottom = 1e34
H_right = 0.0

# Set the initial condition
H.fill(H_initial)

# Set the boundary conditions
H[:, 0, :] = H_left  # x=0
H[:, :, 0] = H_bottom   # y=0
H[:, -1, :] = H_right  # x=max
H[:, :, -1] = H_top  # y=max

@jit(nopython=True)
def calculate(H):

    neighbors_offset = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for t in range(0, max_iter_time-1, 1):

        H_t = H[t]
        H_t_next = H[t + 1]

        for i in range(1, len(H[0, :, 0]) - 1):
            for j in range(1, len(H[0, 0, :]) - 1):
                current_orientation = grain_orientation[i, j]

                # Check if the current point is on a grain boundary
                on_boundary = False
                if current_orientation == -1:
                    on_boundary = True

                else:
                    # Check neighbors manually
                    for di, dj in neighbors_offset:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grain_orientation.shape[0] and 0 <= nj < grain_orientation.shape[1]: 
                            if grain_orientation[ni, nj] != current_orientation:
                                on_boundary = True
                                break  # Exit the loop as we found a boundary

                if on_boundary:
                    # Perform the diffusion calculation if on boundary
                    H_t_next[i, j] = gamma * ( H_t[i+1][j] +  H_t[i-1][j] +  H_t[i][j+1] +  H_t[i][j-1] - 4* H_t[i][j]) +  H_t[i][j]

    return H

def plotheatmap(H_t, t):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Concentration at t = {t*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot H_t (H at time-step t)
    # plt.pcolormesh(H_t, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.imshow(H_t.T, cmap=plt.cm.jet, vmin=0, vmax=100, origin='lower', 
           extent=[x_indices[0] - 0.125, x_indices[-1] + 0.125, y_indices[11] - 0.125, y_indices[-1] + 0.125])
    plt.colorbar()

    return plt

# Do the calculation here
H = calculate(H)

print(f'Calculated H in {time.time() - start:.3g}s')

def animate(t):
    plotheatmap(H[t], t)

animate(max_iter_time -1)
plt.show()

fig = plt.figure(figsize=(10, 10))
anim = FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("H_diffusion/H diffusion jit.gif", dpi=200)

end = time.time()
total = end-start
print(f"Done in {total:.3g}s!")