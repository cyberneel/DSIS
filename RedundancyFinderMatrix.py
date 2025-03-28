import numpy as np

# Define the matrix
matrix = np.array([[0,2,0,0,-1,0,0],
                   [0,10,0,0,0,0,0],
                   [0,0,2,-1,-3,-2,0],
                   [1,0,0,-1,0,0,0],
                   [0,35,4,-4,0,-1,0],
                   [1,0,1,0,0,0,0],
                   [1,0,1,0,0,0,0]])

# Perform rank check to identify redundancy
rank = np.linalg.matrix_rank(matrix)
redundant_rows = len(matrix) - rank
redundant_rows

print(redundant_rows)