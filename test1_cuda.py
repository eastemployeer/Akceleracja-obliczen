import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 
from numba import jit
from numba import cuda
from timeit import default_timer as timer

image = cv.imread("images/example.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


def edgeDetector(image):
    kernel = [ [-1, -1, -1], [-1, 8 , -1], [-1, -1, -1]]
    kernel = np.asarray(kernel)
    filtered_image = cv.filter2D(image, -1, kernel=kernel)
    return filtered_image


@jit
def edgeDetectorCUDA(image):
    kernel = [ [-1, -1, -1], [-1, 8 , -1], [-1, -1, -1]]
    kernel = np.asarray(kernel)
    filtered_image = cv.filter2D(image, -1, kernel=kernel)
    return  (filtered_image, image)





start = timer()
(filtered_image, image) = edgeDetectorCUDA(image)
print("with GPU:", timer()-start)
cuda.profile_stop()

plt.rcParams['figure.figsize'] = (18, 10)
plt.imshow(filtered_image)
plt.show()



start = timer()
filtered_image = edgeDetector(image)
print("without GPU:", timer()-start)

plt.rcParams['figure.figsize'] = (18, 10)
plt.imshow(filtered_image)
plt.show()