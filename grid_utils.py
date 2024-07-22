import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, depth_first_order
from utils import plot_binary_matrix

# Create a 10x10 grid adjacency matrix
def create_grid(n):
    size = n * n
    grid = np.zeros((size, size))
    
    for row in range(n):
        for col in range(n):
            node = row * n + col
            if row > 0:  # Connect to the node above
                grid[node, node - n] = 1
                grid[node - n, node] = 1
            if row < n - 1:  # Connect to the node below
                grid[node, node + n] = 1
                grid[node + n, node] = 1
            if col > 0:  # Connect to the node on the left
                grid[node, node - 1] = 1
                grid[node - 1, node] = 1
            if col < n - 1:  # Connect to the node on the right
                grid[node, node + 1] = 1
                grid[node + 1, node] = 1
                
    return grid

# Create a 10x10 grid
n = 10
grid = create_grid(n)
print("Grid created")

# Convert the grid to a sparse matrix
sparse_grid = csr_matrix(grid)
print("Sparse Grid created")

# Covert grid to DFS ordering
dfs_order = depth_first_order(sparse_grid, 0)[0]
print("DFS complete")

# Apply the reverse Cuthill-McKee ordering
rcm_order = reverse_cuthill_mckee(sparse_grid)
print("RCM complete")

# Reorder the grid according to RCM ordering
reordered_grid_RCM = sparse_grid[rcm_order][:, rcm_order].toarray()
reordered_grid_DFS = sparse_grid[dfs_order][:, dfs_order].toarray()

plot_binary_matrix(sparse_grid.toarray())
plot_binary_matrix(reordered_grid_RCM)

