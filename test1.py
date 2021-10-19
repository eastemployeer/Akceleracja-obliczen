import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

image = cv.imread("images/example.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

kernel = [ [-1, -1, -1], [-1, 8 , -1], [-1, -1, -1]]
kernel = np.asarray(kernel)
filtered_image = cv.filter2D(image, -1, kernel=kernel)

plt.rcParams['figure.figsize'] = (18, 10)
plt.imshow(filtered_image)
plt.show()