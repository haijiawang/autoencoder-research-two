'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io, viewer
import skimage as sk
img = io.imread('image.jpg', as_grey=True)
print 'image matrix size: ', img.shape
print '\n First 5 columns and rows of the image matrix: \n', img[:5, :5]* 255
viewer = sk.ImageViewer(img)
viewer.show()
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pylab
from skimage import io, color

def convolve2d(image, kernel):
    #This function takes an iamge and a kernel and returns the convolution of them
    #Takes an image and kernel
    #Returns a numpy array of size image height by image width (convolution output)

    kernel = np.flipud(np.fliplr(kernel)) #flip the kernel
    output = np.zeros_like(image)

    #add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] =  image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            #element-wise multiplication of the kernel and the image
            output[y, x] = (kernel*image_padded[y:y+3,x:x+3]).sum()

    return output

img = io.imread('image.jpg')
img = color.rgb2gray(img)

#adjust the contrast of the image by applying Histogram Equalization
image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
plt.imshow(image_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

#convolve the sharpen kernel and the image
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
image_sharpen = convolve2d(img, kernel)
print'\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5, :5]*255

#plot the filtered image
plt.imshow(image_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

#adjust the contrast of the filiter image by applying histogram equalization
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)))
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()


