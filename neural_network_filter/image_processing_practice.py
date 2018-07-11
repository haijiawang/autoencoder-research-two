#Opening and writing to image files
from PIL import Image
from scipy import misc
import scipy.ndimage as ndimage
import numpy as np
f = misc.face()
misc.imsave('face.png', f)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#plt.imshow(f)
#plt.show()

#create a numpy array from an image file
face = misc.imread('face.png')
print(face)

#Display Images
f = misc.face(gray=True) #retrieve a grayscale image
#plt.imshow(f, cmap=plt.cm.gray)
#increase contrast by setting min and max values
#plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)
plt.axis('off')
#plt.show()

'''
Image Filtering: 
local filters: replace the value of pixels by a functions of the values of neighbouring pixels
link: http://www.scipy-lectures.org/advanced/image_processing/
'''
face = misc.face(gray=True)
blurred_face = ndimage.gaussian_filter(face, sigma=3)
very_blurred = ndimage.gaussian_filter(face, sigma=5)

face = misc.face(gray=True).astype(float)

