import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 
from numba import jit
from numba import cuda
from timeit import default_timer as timer

fig, ax = plt.subplots(1, 4)

image = cv.imread("images/example.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
result_image = image.copy()

result_image_CUDA = image.copy()

downsampled_image_cr_CUDA = np.zeros(shape=(400,600))
downsampled_image_cb_CUDA = np.zeros(shape=(400,600))

upsampled_image_cr_CUDA = np.zeros(shape=(800,1200))
upsampled_image_cb_CUDA = np.zeros(shape=(800,1200))

downsampled_image_cr = []
downsampled_image_cb = []

upsampled_image_cr = []
upsampled_image_cb = []



def upsample_fragment(up_image, down_image, x, y):
        up_image[x*2].append(down_image[x][y])
        up_image[x*2].append(down_image[x][y])
        up_image[x*2+1].append(down_image[x][y])
        up_image[x*2+1].append(down_image[x][y])

def downsample(downsampled_image_cr, downsampled_image_cb, image):
    for x in range(0,image.shape[0],2):
        downsampled_image_cr.append([])
        downsampled_image_cb.append([])
        for y in range(0,image.shape[1],2):
            if x != 0:
                downsampled_image_cr[int(x/2)].append(image[x,y,1])
                downsampled_image_cb[int(x/2)].append(image[x,y,2])
            else:
                downsampled_image_cr[x].append(image[x,y,1])
                downsampled_image_cb[x].append(image[x,y,2])


def upsample(downsampled_image_cb, downsampled_image_cr, upsampled_image_cb, upsampled_image_cr, result_image):
    for x in range(len(downsampled_image_cb)):
        #CR
        upsampled_image_cr.append([])
        upsampled_image_cr.append([])
        #CB
        upsampled_image_cb.append([])
        upsampled_image_cb.append([])

        # iterating over downsampled image width
        for y in range(len(downsampled_image_cb[0])):
            #CR
            upsample_fragment(upsampled_image_cr,downsampled_image_cr,x,y)
            #CB
            upsample_fragment(upsampled_image_cb,downsampled_image_cb,x,y)
    
    for x in range(result_image.shape[0]):
        for y in range(result_image.shape[1]):
            result_image[x,y,1] = upsampled_image_cr[x][y]
            result_image[x,y,2] = upsampled_image_cb[x][y]

@jit
def downsampleCUDA(downsampled_image_cr, downsampled_image_cb, image):
    for x in range(0,image.shape[0],2):
        iterator = 0
        for y in range(0,image.shape[1],2):
            if x != 0:
                downsampled_image_cr[int(x/2), iterator] = image[x,y,1]
                downsampled_image_cb[int(x/2), iterator] = image[x,y,2]
            else:
                downsampled_image_cr[x, iterator] = image[x,y,1]
                downsampled_image_cb[x, iterator] = image[x,y,2]

            iterator = iterator + 1
    return (downsampled_image_cr, downsampled_image_cb)


@jit
def upsampleCUDA(downsampled_image_cb, downsampled_image_cr, upsampled_image_cb, upsampled_image_cr, result_image_CUDA):
    # iterating over downsampled image height
    for x in range(len(downsampled_image_cb)):

        # iterating over downsampled image width
        for y in range(len(downsampled_image_cb[0])):
            iterator = y*2
  
            upsampled_image_cr[x*2, iterator] = downsampled_image_cr[x, y]
            upsampled_image_cr[x*2, iterator+1] = downsampled_image_cr[x, y]
            upsampled_image_cr[x*2+1, iterator] = downsampled_image_cr[x, y]
            upsampled_image_cr[x*2+1, iterator+1] = downsampled_image_cr[x, y]
            upsampled_image_cb[x*2, iterator] = downsampled_image_cb[x, y]
            upsampled_image_cb[x*2, iterator+1] = downsampled_image_cb[x, y]
            upsampled_image_cb[x*2+1, iterator] = downsampled_image_cb[x, y]
            upsampled_image_cb[x*2+1, iterator+1] = downsampled_image_cb[x, y]

    
    for x in range(result_image_CUDA.shape[0]):
        for y in range(result_image_CUDA.shape[1]):
            result_image_CUDA[x,y,1] = upsampled_image_cr[x, y]
            result_image_CUDA[x,y,2] = upsampled_image_cb[x, y]
    return result_image_CUDA



# CUDA
start = timer()
(downsampled_image_cr_CUDA, downsampled_image_cb_CUDA) = downsampleCUDA(downsampled_image_cr_CUDA, downsampled_image_cb_CUDA, image)
print("downsampling with GPU:", timer()-start)

start = timer()
result_image_CUDA = upsampleCUDA(downsampled_image_cb_CUDA, downsampled_image_cr_CUDA, upsampled_image_cb_CUDA, upsampled_image_cr_CUDA, result_image_CUDA)
print("upsampling with GPU:", timer()-start)

# No CUDA
start = timer()
downsample(downsampled_image_cr, downsampled_image_cb, image)
print("downsampling without GPU:", timer()-start)

start = timer()
upsample(downsampled_image_cb, downsampled_image_cr, upsampled_image_cb, upsampled_image_cr, result_image)
print("upsampling without GPU:", timer()-start)

result_image_CUDA = cv.cvtColor(result_image_CUDA, cv.COLOR_YCrCb2RGB)
result_image = cv.cvtColor(result_image, cv.COLOR_YCrCb2RGB)

ax[0].imshow(result_image)
ax[1].imshow(result_image[:,:,0])
ax[2].imshow(result_image[:,:,1])
ax[3].imshow(result_image[:,:,2])

plt.show() 