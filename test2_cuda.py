import cv2 as cv
import numpy as np
from numba import jit
from numba import cuda
from timeit import default_timer as timer
from matplotlib import pyplot as plt 

image = cv.imread("images/example.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = image * 1./255
image_copy = image.copy()


def rgbToRgb(image, image_copy):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x,y,0] = image_copy[x,y,0] * 0.393 + image_copy[x,y,1] * 0.769 + image_copy[x,y,2] * 0.189
            image[x,y,1] = image_copy[x,y,0] * 0.349 + image_copy[x,y,1] * 0.689 + image_copy[x,y,2] * 0.168
            image[x,y,2] = image_copy[x,y,0] * 0.272 + image_copy[x,y,1] * 0.534 + image_copy[x,y,2] * 0.131
    image = np.clip(image, 0, 1)

def clipImage(image):
    np.clip(image,0,1)

@jit
def rgbToRgbCUDA(image, image_copy):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x,y,0] = image_copy[x,y,0] * 0.393 + image_copy[x,y,1] * 0.769 + image_copy[x,y,2] * 0.189
            image[x,y,1] = image_copy[x,y,0] * 0.349 + image_copy[x,y,1] * 0.689 + image_copy[x,y,2] * 0.168
            image[x,y,2] = image_copy[x,y,0] * 0.272 + image_copy[x,y,1] * 0.534 + image_copy[x,y,2] * 0.131
    return (image, image_copy)

@jit
def clipImageCUDA(image):
    return np.clip(image,0,1)

start = timer()
(image1, image_copy1) = rgbToRgbCUDA(image,image_copy)
print("main part - with GPU:", timer()-start)

start = timer()
image1 = clipImageCUDA(image1)
print("clipping - with GPU:", timer()-start)


start = timer()
rgbToRgb(image,image_copy)
print("main part - without GPU:", timer()-start)

start = timer()
image = clipImage(image)
print("clipping - without GPU:", timer()-start)

cuda.profile_stop()

# plt.rcParams['figure.figsize'] = (18, 10)
# plt.imshow(image1)
# plt.show()