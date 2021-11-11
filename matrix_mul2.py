
from numba import jit
import numpy
from numba import cuda
from timeit import default_timer as timer


def matmul(A, B, C):
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp


@jit
def matmulCUDA(A, B, C):
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp
    return (A, B, C)



A = numpy.full((240, 120), 3, numpy.float)  # matrix containing all 3's
B = numpy.full((120, 220), 4, numpy.float)  # matrix containing all 4's



C = numpy.zeros((240, 220))

start = timer()
matmulCUDA(A, B, C)
print("with GPU:", timer()-start)
cuda.profile_stop()



C = numpy.zeros((240, 220))

start = timer()
matmul(A, B, C)
print("without GPU:", timer()-start)