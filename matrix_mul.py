from __future__ import division
from numba import cuda
from timeit import default_timer as timer
import numpy
import math


# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


def matmul2(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp


# Host code

# Initialize the data arrays
A = numpy.full((240, 120), 3, numpy.float)  # matrix containing all 3's
B = numpy.full((120, 220), 4, numpy.float)  # matrix containing all 4's

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((240, 220))

# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel
start = timer()
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
C = C_global_mem.copy_to_host()
print("main part - with GPU:", timer()-start)

# Copy the result back to the host


# print(C)
# print()
C = numpy.zeros((240, 220))
start = timer()

matmul2(A, B, C)
print("main part - without GPU:", timer()-start)
# print(C)
