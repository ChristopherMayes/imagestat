#See https://github.com/NIUaard/ImageTool for the newest version of this toolbox
#Written by P. Piot at NIU (2016)

import numpy as np
import math 
import pylab as pyl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize 
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap
from numpy import mean, sqrt, square, arange

from plotting import *
# beam density friendly color map
from parameters import *

cdict = cm.get_cmap('spectral')._segmentdata

cdict['red'][0]   = (0, 0.0, 1)  
cdict['blue'][0]  = (0, 0.0, 1)  
cdict['green'][0] = (0, 0.0, 1)  

del cdict['red'][17:19]
del cdict['blue'][17:19]
del cdict['green'][17:19]

cdict['red'][-1]   = (1, 1.00, 1)  
cdict['blue'][-1]  = (1, 0.0, 0)  
cdict['green'][-1] = (1, 0.0, 0)  


#for i in range(17):
#  cdict['red'][i][0]   = cdict['red'][i][0]*1.0/0.8
#  cdict['red'][i][1]   = cdict['red'][i][0]*1.0/0.8 
#  cdict['blue'][i]  = (0, 0.0, 1)  
#  cdict['green'][i] = (0, 0.0, 1)  


#print cdict['red'][:]
#print cdict['blue'][:]
#print cdict['green'][:]

beamcmap = LinearSegmentedColormap('name', cdict)

plt.register_cmap(name='beamcmap', cmap=beamcmap)


def Normalize(MyImage):
     w = np.where(MyImage<0.0)
     MyImage[w] = 0.0
     maxv=np.amax(MyImage)
     MyImage=MyImage/maxv
     return(MyImage)

def Denoise(MyImage):
     img = ndimage.gaussian_filter(MyImage, sigma=(5), order=0)
     return img


def MonteCarloXY(MyImage,N,cal):
     x,y = np.shape(MyImage)
    # print x,y
     dist=np.zeros((N,2))
     i=0
     while i<N:
        rand_x=np.random.random_integers(x-2)+np.random.uniform(-1, 1)
        rand_y=np.random.random_integers(y-2)+np.random.uniform(-1, 1)
        value=np.random.rand()
        if value<MyImage[int(round(rand_x)),int(round(rand_y))]:
#add randomized dx,dy
            dist[i,0]=rand_x
            dist[i,1]=rand_y
            i=i+1  
     meanx,meany = dist.mean(axis=0)
     dist[:,0]=dist[:,0]-meanx
     dist[:,1]=dist[:,1]-meany
     dist=dist*cal*1.0e-6
     
     xrms=sqrt(mean(square(dist[:,0])))
     yrms=sqrt(mean(square(dist[:,1])))
     
     print "RMS values:"
     print xrms*1000.0, yrms*1000.0
     
     return(dist)


def dg(x,p0):
    rv=np.zeros(len(x))
    for i in range(len(x)):
        rv[i]=p0[3]+p0[1]*math.exp(-(x[i]-p0[0])*(x[i]-p0[0])/2/p0[2]/p0[2])
    return rv


def fitprofile(projection, axiscoord):
    
     xhist = projection
     xaxis = axiscoord
     indexXmax=xaxis[np.argmax(xhist)]
     bkg = np.mean(xhist[0:40])
     Xmax = np.max(xhist)
     p0x  = [indexXmax,Xmax, 1.,bkg]
     if (quiet==False): print Xmax, indexXmax, bkg
     ErrorFunc = lambda p0x,xaxis,xhist: dg(xaxis,p0x)-xhist
     p2,success = scipy.optimize.leastsq(ErrorFunc, p0x[:], args=(xaxis,xhist))
     
     return(p2)


def Load(filename):
     return(pyl.imread(filename))

def AutoCrop(MyImage, xbox,ybox):
#padding image with (0,0) on each side to avoid cropping error
     m = len(MyImage)
     MyImage = np.pad(MyImage,((m,m),(m,m)), 'constant')
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))

     return(MyImage[indexYmax-ybox:indexYmax+ybox,indexXmax-xbox:indexXmax+xbox])
     
def Threshold(MyImage, thres):
     MyImage = Normalize(MyImage)
     index = np.where(MyImage<thres)
     MyImage[index]=0.0
     return(MyImage)
     
def DisplayCalibrated(MyImage, cal):
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     ImShape=np.shape(MyImage)
     calx=cal
     caly=cal
     
     xmin=calx*(0.-indexXmax)
     xmax=calx*(ImShape[0]-indexXmax)
     ymin=caly*(0.-indexYmax)
     ymax=caly*(ImShape[1]-indexYmax)
     
     plt.imshow(MyImage, aspect='auto', cmap='spectral',origin='lower',extent=[xmin, xmax, ymin, ymax])
     plt.colorbar()
     
def ImageFit(MyImage,cal,plot=None):
    sumx=np.sum(MyImage,0)
    sumy=np.sum(MyImage,1)
    axisX=np.arange(len(sumx))
    axisY=np.arange(len(sumy))
    
    p2X= fitprofile(sumx, axisX)
    p2Y= fitprofile(sumy, axisY)
    
    if plot is None:
        plot = False
    if (plot != False):  
        print "Gaussian sigma x: ", cal*p2X[2]
        print "Gaussian sigma y: ", cal*p2Y[2]
    
        plt.figure()
        plt.title('Gaussian fit')
        plt.plot(axisX, sumx,'ob',alpha=0.45)
        plt.plot(axisY, sumy,'or',alpha=0.45)
        plt.legend(('X proj.','Y proj.'))
        plt.plot(axisX, dg(axisX,p2X),'--b',linewidth=3)
        plt.plot(axisY, dg(axisY,p2Y),'--r',linewidth=3)

        plt.xlabel('Size (px)')
        plt.ylabel('Axis projection')
        plt.tight_layout()
        plt.show()
    
    return p2X, p2Y
    
    
def DisplayCalibratedProj(MyImage, cal, fudge):
     indexXmax=np.argmax(np.sum(MyImage,0))
     indexYmax=np.argmax(np.sum(MyImage,1))
     ImShape=np.shape(MyImage)
     calx=cal
     caly=cal
     
     xmin=calx*(0.-indexXmax)
     xmax=calx*(ImShape[0]-indexXmax)
     ymin=caly*(0.-indexYmax)
     ymax=caly*(ImShape[1]-indexYmax)
     
     xhist = np.sum(MyImage,0)/np.max(np.sum(MyImage,0))
     yhist = np.sum(MyImage,1)/np.max(np.sum(MyImage,1))
     
     xcoord = xmin+np.linspace(0,1,len(xhist))*(xmax-xmin)
     xhist  = xmin+ fudge*(xmax-xmin)*xhist
     ycoord = ymin+np.linspace(0,1,len(yhist))*(ymax-ymin)
     yhist  = ymin+ fudge*(ymax-ymin)*yhist

     plt.imshow(MyImage, aspect='auto', cmap='beamcmap',origin='lower',extent=[xmin, xmax, ymin, ymax])
     plt.plot(xcoord,xhist,color='r',linewidth=3) 
     plt.plot(yhist, ycoord,color='r', linewidth=3) 
     plt.ylim(ymin, ymax)
     plt.xlim(xmin, xmax)
     plt.colorbar()
     
    
