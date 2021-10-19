import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 
from numba import jit
from numba import cuda
from timeit import default_timer as timer

fig, ax = plt.subplots(1, 5)
image = cv.imread("images/example.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_copy = image.copy()

def rgbToYCbCr(image, image_copy):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x,y,0] = image_copy[x,y,0] * 0.229 + image_copy[x,y,1] * 0.587 + image_copy[x,y,2] * 0.114
            image[x,y,1] = image_copy[x,y,0] * 0.500 + image_copy[x,y,1] * (-0.418) + image_copy[x,y,2] * (-0.082) + 128
            image[x,y,2] = image_copy[x,y,0] * (-0.168) + image_copy[x,y,1] * (-0.331) + image_copy[x,y,2] * 0.500 + 128
    

@jit
def rgbToYCbCrCUDA(image, image_copy):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x,y,0] = image_copy[x,y,0] * 0.229 + image_copy[x,y,1] * 0.587 + image_copy[x,y,2] * 0.114
            image[x,y,1] = image_copy[x,y,0] * 0.500 + image_copy[x,y,1] * (-0.418) + image_copy[x,y,2] * (-0.082) + 128
            image[x,y,2] = image_copy[x,y,0] * (-0.168) + image_copy[x,y,1] * (-0.331) + image_copy[x,y,2] * 0.500 + 128
    return (image, image_copy)
        

start = timer()
(image1, image_copy1) = rgbToYCbCrCUDA(image, image_copy)
print("with GPU:", timer()-start)
cuda.profile_stop()

start = timer()
rgbToYCbCr(image, image_copy)
print("without GPU:", timer()-start)

image1 = np.clip(image1, 0, 255)
imageRGB = cv.cvtColor(image1, cv.COLOR_YCrCb2RGB)
ax[0].imshow(image_copy1)
ax[1].imshow(imageRGB)
ax[2].imshow(image1[:,:,0], cmap="Greys_r")
ax[3].imshow(image1[:,:,1], cmap="Greys_r")
ax[4].imshow(image1[:,:,2], cmap="Greys_r")

plt.show()