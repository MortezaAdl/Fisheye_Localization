import numpy as np
psi = 2
delta_t = 0.1
v = 10
# Define the matrix
O_straight = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, np.cos(psi) * delta_t, -v * np.sin(psi) * delta_t, 0],
    [0, 1, np.sin(psi) * delta_t, v * np.cos(psi) * delta_t, 0],
    [1, 0, 2 * np.cos(psi) * delta_t, -2 * v * np.sin(psi) * delta_t, 0],
    [0, 1, 2 * np.sin(psi) * delta_t, 2 * v * np.cos(psi) * delta_t, 0],
    [1, 0, 3 * np.cos(psi) * delta_t, -3 * v * np.sin(psi) * delta_t, 0],
    [0, 1, 3 * np.sin(psi) * delta_t, 3 * v * np.cos(psi) * delta_t, 0],
    [1, 0, 4 * np.cos(psi) * delta_t, -4 * v * np.sin(psi) * delta_t, 0],
    [0, 1, 4 * np.sin(psi) * delta_t, 4 * v * np.cos(psi) * delta_t, 0]
])

# Compute the rank
rank = np.linalg.matrix_rank(O_straight)
print("Rank of O_straight:", rank)
