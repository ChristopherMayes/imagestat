# A. Halavanau. DeKalb, IL (2017) / Menlo Park, CA (2018)
import numpy as np
import matplotlib.pyplot as plt
import ImageTool as imtl
import matplotlib.patches as patches
#importing global variables
from parameters import *

#Central moments calculation procedure based on a scanning elliptical mask method


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
#Create meshgrid for fast central moments (iord,jord) calculation
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def image_raw_moments(data):
#Calculate <x>,<y>,<xx>,<xy> and <xy>
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    print "Statistical moments sigma_x,sigma_y: "
    print np.sqrt(u20), np.sqrt(u02)
    return x_bar, y_bar, cov

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the center of the image
	center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image borders
	radius = min(center[0], center[1], w-center[0], h-center[1])
#Create a meshgrid of a given area for fast calculation
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#Create a circular mask
    mask = dist_from_center <= radius
    return mask

def create_elliptical_mask(h, w, center=None, a = None, b = None, theta = None):
    if center is None: # use the center of the image
	center = [int(w/2), int(h/2)]
    if a is None: 
	a = min(center[0], center[1], w-center[0], h-center[1])
    if b is None: 
	b = min(center[0], center[1], w-center[0], h-center[1])
    if theta is None: 
	theta = 0.0
#Create a meshgrid of a given area for fast calculation
    Y, X = np.ogrid[:h, :w]
    X = X - center[0]
    Y = Y - center[1]
    rtheta = np.radians(theta)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])
    Y, X = np.dot(R, np.array([Y, X]))
    dist_from_center = np.sqrt(b*b*(X)**2 + a*a*(Y)**2)
#Create a circular mask
    mask = dist_from_center <= np.sqrt(a*a*b*b)
    return mask


def covR(data):
#Calculation of image covariance matrix as a function of mask radius R
#Find the image center-of-mass
    xbar, ybar, cov = image_raw_moments(data)
    center = [xbar, ybar]
    p2x,p2y = imtl.ImageFit(data,1.0)
    print "Image center is at: ", p2x[0], p2y[0]
    center = [p2x[0], p2y[0]]
#Define image size
    h, w = data.shape[:2]
    cov_array = np.zeros((m,4))
#Loop over radius size (program core loop)
    for i in range (0,m):
        factor = int( max(semi_a0,semi_b0) / min(semi_a0,semi_b0) )
        semi_a = semi_a0 + i*step
        semi_b = semi_b0 + i*step
        mask = create_elliptical_mask(h, w,center,semi_a,semi_b,skew)
        temp = data.copy()
        #zero all the points outside of region of interest
        temp[~mask] = 0.0
        xbart, ybart, covt = image_raw_moments(temp)
        cov_array[i,:] = np.ndarray.flatten(covt)
        print semi_a,semi_b,i
        #store calculated moments as a function of radius size for analysis
    return cov_array

def plot_covarray(cov_array):
    plt.figure()
    #   plt.yscale('log')
    plt.plot(cov_array[:,0],'-',linewidth=3)
    plt.plot(np.abs(cov_array[:,1]),'-',linewidth=3)
    plt.plot(cov_array[:,3],'-',linewidth=3)
    plt.yscale('log')
    plt.ylabel('Central moment value (px*px)')
    plt.xlabel(r'Mask size + $d_0$ (px)')
    plt.legend((r'$\langle xx\rangle$',r'$\langle xy\rangle$',r'$\langle yy\rangle$'),loc=4)
    plt.tight_layout()

    plt.figure()
    plt.plot(np.gradient(cov_array[:,0]),'-',linewidth=3)
    plt.plot(np.gradient(cov_array[:,1]),'-',linewidth=3)
    plt.plot(np.gradient(cov_array[:,3]),'-',linewidth=3)
    plt.ylabel('Moment derivative (px)')
    plt.xlabel(r'Mask size + $d_0$ (px)')
    plt.legend((r'$\langle xx\rangle$',r'$\langle xy\rangle$',r'$\langle yy\rangle$'),loc=1)

    plt.tight_layout()
    #print "minimum at: ", np.argmin(np.abs(np.gradient(cov_array[3:,1])))
    plt.show()

def dcov_array_max(cov_array):
    m = len(cov_array)
   # nx = ny = nxy = int(0.3*m)
    nx = np.argmax(np.gradient(cov_array[0:m/2,0]))
    ny = np.argmax(np.gradient(cov_array[0:m/2,3]))
    nxy = np.argmax(np.abs(np.gradient(cov_array[0:m/2,1])))
    return nx, ny, nxy

def dcov_array_min(cov_array):
#Calculate the correct radii for <xx>,<yy> and <xy> moments. The condition is min(d covR(R) / dR)
#calculating derivatives
    divx = np.gradient(cov_array[:,0])
    divy = np.gradient(cov_array[:,3])
    divxy = np.abs(np.gradient(cov_array[:,1]))
#calculating local minima for a not too noisy function
    minx = (np.diff(np.sign(np.diff(divx))) > 0).nonzero()[0] + 1
    miny = (np.diff(np.sign(np.diff(divy))) > 0).nonzero()[0] + 1
    minxy = (np.diff(np.sign(np.diff(divxy))) > 0).nonzero()[0] + 1
    if (len(minx)==0):
       minx = [0]
    if (len(miny)==0):
       miny = [0]
    if (len(minxy)==0):
       minxy = [0]
    if (quiet != True):
       print "Resulting derivative masks:"
       print minx, miny, minxy

#Picking the global minimum out of local minima
    nx = np.argmin(divx[minx])
    ny = np.argmin(divy[miny])
#<xy> moment mask can not be less in size than the max(Rx,Ry). This is needed for non homogeneous beams
    maxr = np.max((minx[nx],miny[ny]))
 #   print "max(Rx,Ry): ", maxr
    if (maxr>np.max(minxy)): 
       maxr = np.max(minxy)
       print "Warning: <xy> may not be calculated correctly"
    w = np.where(minxy>=maxr)
    minxy = minxy[w]
#    print 'Filtered minxy', minxy
    nxy = np.argmin(divxy[minxy])

    print minx[nx],miny[ny],minxy[nxy]
    return minx[nx],miny[ny],minxy[nxy]

def image_moments(data,semiaxis_a,semiaxis_b):
   
    global radius0 
    radius0 = (semiaxis_a + semiaxis_b) / 2
    global semi_a0, semi_b0
    semi_a0 = int(2*semiaxis_a)
    semi_b0 = int(2*semiaxis_b)
    print "Inital mask radius: ", radius0
    h,v = data.shape[:2]
    global step 
    step = int((h/2 - max(semi_a0,semi_b0))/m)
    print "Step size is: ", step

#Retrieve cov(R) data
    cov_array = covR(data)
#Find the image center-of-mass
    xbar, ybar, cov = image_raw_moments(data)
    center = [xbar, ybar]

    p2x,p2y = imtl.ImageFit(data,1.0)
    print "Image center is at: ", p2x[0], p2y[0]
    center = [p2x[0], p2y[0]]
#    ellipse = data.copy()
#    mask_ellipse = create_elliptical_mask(h,v,center,radius0,3*radius0,45.0)
#    ellipse[~mask_ellipse] = 0
#    plt.figure()
#    plt.imshow(ellipse)
#    plt.show()
    
#Calculate the correct radii for <xx>,<yy> and <xy> moments. The condition is min(d covR(R) / dR)
#    m = len(cov_array)
#Retrieve the positions of max(d covR(R) / dR)
#    nx, ny, nxy = dcov_array_max(cov_array)
#Radius is calculated as the location of derivative minimum
#    radiusX = radius0 + step*np.argmin(np.gradient(cov_array[nx:m-1,0])) + step*nx
#    radiusY = radius0 + step*np.argmin(np.gradient(cov_array[ny:m-1,3])) + step*ny
#    radiusXY = radius0 + step*np.argmin(np.abs(np.gradient(cov_array[nxy:m-1,1]))) + step*nxy

    minx,miny,minxy = dcov_array_min(cov_array)

    semi_aX = semi_a0 + step*minx
    semi_bX = semi_b0 + step*minx
    semi_aY = semi_a0 + step*miny
    semi_bY = semi_b0 + step*miny
    semi_aXY = semi_a0 + step*minxy
    semi_bXY = semi_b0 + step*minxy


    print "Mask ellipse X: ", semi_aX, semi_bX
    print "Mask ellipse Y: ", semi_aY, semi_bY
    print "Mask ellipse XY: ",semi_aXY, semi_bXY
#Plotting covariance matrix elements for debugging
    plot_covarray(cov_array)
#Create masks for <xx>,<yy> and <xy> moments    
    maskX = create_elliptical_mask(h,v,center,semi_aX,semi_bX,skew)
    maskY = create_elliptical_mask(h,v,center,semi_aY,semi_bY,skew)
    maskXY = create_elliptical_mask(h,v,center,semi_aXY,semi_bXY,skew)
#Prepare three copies of masked data for final moments calculation    
    tempX = data.copy()
    tempY = data.copy()
    tempXY = data.copy()
    tempX[~maskX] = 0
    tempY[~maskY] = 0
    tempXY[~maskXY] = 0
#    plt.figure()
    
    fig, ax = plt.subplots(1)
    extent = (0, len(tempXY[:,0]), 0, len(tempXY[:,1]))
    tempXY = np.flip(tempXY,0)
    ax.imshow(imtl.Normalize(tempXY),extent=extent)
    cx = patches.Ellipse(center,2*semi_aX,2*semi_bX,skew,color='y',linewidth=3,fill=False)
    cy = patches.Ellipse(center,2*semi_aY,2*semi_bY,skew,color='g',linewidth=3,fill=False)
    cxy = patches.Ellipse(center,2*semi_aXY,2*semi_bXY,skew,color='m',linewidth=3,fill=False)
    ax.add_patch(cx)
    ax.add_patch(cy)
    ax.add_patch(cxy)
    plt.legend((cx,cy,cxy), (r'Mask $\langle xx \rangle$',r'Mask $\langle yy\rangle$',r'Mask $\langle xy\rangle$'))
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.tight_layout()
    plt.show()


#Final calculation
    xxt,yyt,covx = image_raw_moments(tempX)
    xxt,yyt,covy = image_raw_moments(tempY)
    xxt,yyt,covxy = image_raw_moments(tempXY)

    return covx[0,0],covy[1,1],covxy[0,1]

