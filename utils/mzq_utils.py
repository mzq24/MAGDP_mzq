import numpy as np

def is_block_diagonal(matrix):
    n = len(matrix)
    i = 0
    start_point = 0
    pointer = start_point
    submatrix_size = []
    
    while pointer < n:
        tmp_size = 0
        while matrix[start_point][pointer]:
            tmp_size += 1
            pointer += 1
            if pointer >= n:
                break
        submatrix_size.append(tmp_size)
        if np.all(matrix[start_point:start_point+tmp_size, start_point:start_point+tmp_size].flatten()):
            pass
        else:
            print(f'Block [{start_point}, {pointer}] is not diagonal')
            return False
        start_point = pointer
    return True, submatrix_size
