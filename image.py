import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import ImageTool as imtl
from statistical import *
from parameters import *

i = 1

pwd = 'examples/Normal_quad_scan_L2/'
filename = pwd + 'X111_YAG_'+str(i)+'_bkg_0.png'
filenamebg = pwd + 'X111_YAG_'+str(i)+'_img_0.png'

data = imtl.Load(filename)
bg = imtl.Load(filenamebg)
data = data - bg
data = data[0:2000,0:2000]
data = imtl.Normalize(data)

#experimental
data_rot = data.copy()
data_rot = rotate(data,skew,reshape=False)

x,y = imtl.ImageFit(data_rot,1.0,True)
print "Image center is at: ", int(x[0]),int(y[0])
print "Gaussian fit sizes (px): ",int(x[2]),int(y[2])
print "Gaussian fit sizes (m): ",cal*x[2],cal*y[2]
sigma_max = int(np.max((x[2],y[2])))
sigma_min = int(np.min((x[2],y[2])))

#initial crop
box = int(4.7*sigma_max)
data = imtl.AutoCrop(data,box,box)
data = imtl.Denoise(data)

#x,y = imtl.ImageFit(data,cal)
#print "Gaussian fit: ",x,y

if (quiet != True):
    plt.figure()
    extent = (0, len(data[:,0]), 0, len(data[:,1]))
    plt.imshow(data,extent=extent)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.tight_layout()
    plt.show()

xx, yy, xy = image_moments(data,int(x[2]),int(y[2]))
print "Calculated image moments are (m):\n"
print "sigma_x: "+ str(cal*np.sqrt(xx))
print " sigma_y: "+str(cal*np.sqrt(yy)) 
print "<xy>: " + str(cal*cal*xy)
#print "Skew angle suggestion (experimental): ",np.rad2deg(-np.arctan(xy/np.sqrt(xx*yy)))

