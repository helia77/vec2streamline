import numpy as np
import scipy as sp
from skimage.color import rgb2gray
import scipy.ndimage

def structure2d(I, sigma):

    if(len(I.shape) > 2):
        img = rgb2gray(I)
    else:
        img = I

    # calculate the image gradient
    dIdy, dIdx = np.gradient(img)
    
    

    # create the structure tensor
    T = np.zeros((img.shape[0], img.shape[1], 2, 2))
    T[:, :, 0, 0] = dIdy * dIdy
    T[:, :, 1, 1] = dIdx * dIdx
    T[:, :, 0, 1] = dIdx * dIdy
    T[:, :, 1, 0] = T[:, :, 0, 1]

    #if the sigma value is 0, don't do any blurring or resampling
    if sigma == 0:
        return T
        
    # otherwise blur and resample the image
    else:    
        # blur the structure tensor
        T_blur = sp.ndimage.gaussian_filter(T, [sigma, sigma, 0, 0])

    return T_blur

def structure3d(V, sigma, z_scale = 1):
    
    vol_gradient = np.gradient(V)
    dIdz, dIdy, dIdx = vol_gradient[0]/z_scale, vol_gradient[1], vol_gradient[2]
    # create the structure tensor
    T = np.zeros((dIdx.shape[0], dIdx.shape[1], dIdx.shape[2], 3, 3))
    T[:, :, :, 0, 0] = dIdz * dIdz
    T[:, :, :, 1, 1] = dIdy * dIdy
    T[:, :, :, 2, 2] = dIdx * dIdx
    T[:, :, :, 0, 1] = dIdz * dIdy
    T[:, :, :, 1, 0] = T[:, :, :, 0, 1]
    T[:, :, :, 0, 2] = dIdz * dIdx
    T[:, :, :, 2, 0] = T[:, :, :, 0, 2]
    T[:, :, :, 1, 2] = dIdy * dIdx
    T[:, :, :, 2, 1] = T[:, :, :, 1, 2]

    # if the sigma value is 0, don't do any blurring or resampling
    if sigma == 0:
        return T
        
    # otherwise blur and resample the volume
    else:
        # blur the structure tensor
        T_blur = sp.ndimage.gaussian_filter(T, [sigma/z_scale, sigma, sigma, 0, 0])

    return T_blur
        