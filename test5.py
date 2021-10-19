import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

image = cv.imread("images/example.jpg")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
result_image = image.copy()

def upsample(up_image, down_image, x, y):
        up_image[x*2].append(down_image[x][y])
        up_image[x*2].append(down_image[x][y])
        up_image[x*2+1].append(down_image[x][y])
        up_image[x*2+1].append(down_image[x][y])

#DOWNSAMPLING
downsampled_image_cr = []
downsampled_image_cb = []
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


#UPSAMPLING
upsampled_image_cr = []
upsampled_image_cb = []

# iterating over downsampled image height
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
        upsample(upsampled_image_cr,downsampled_image_cr,x,y)
        #CB
        upsample(upsampled_image_cb,downsampled_image_cb,x,y)
 
for x in range(result_image.shape[0]):
    for y in range(result_image.shape[1]):
        result_image[x,y,1] = upsampled_image_cr[x][y]
        result_image[x,y,2] = upsampled_image_cb[x][y] 

        
result_image = cv.cvtColor(result_image, cv.COLOR_YCrCb2RGB)

def countMSE(image_in, image_out):
    sum = 0
    for x in range(image_in.shape[0]):
        for y in range(image_in.shape[1]):
            for z in range(3):
                sum += pow(int(image_in[x,y,z]) - int(image_out[x,y,z]), 2)
    MSE = (1/(int(image_in.shape[0])*int(image_in.shape[1])))*(1/3)*sum
    print("Błąd: {}".format(MSE))

countMSE(image_rgb,result_image)




