# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:57:24 2023

@author: helioum

This code is to test the vector2streamline function for a tensor field of an image
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.ndimage as sp
from skimage.color import rgb2gray
from skimage.io import imread
import threading
#%%
def structure2d(I, sigma):

    if(len(I.shape) > 2):
        img = rgb2gray(I)
    else:
        img = I

    # calculate the image gradient
    dIdy, dIdx = np.gradient(img)

    # create the structure tensor
    T = np.zeros((img.shape[0], img.shape[1], 2, 2))
    T[:, :, 0, 0] = dIdx * dIdx
    T[:, :, 1, 1] = dIdy * dIdy
    T[:, :, 0, 1] = dIdx * dIdy
    T[:, :, 1, 0] = T[:, :, 0, 1]

    #if the sigma value is 0, don't do any blurring or resampling
    if sigma == 0:
        return T
        
    # otherwise blur and resample the image
    else:    
        # blur the structure tensor
        T_blur = sp.gaussian_filter(T, [sigma, sigma, 0, 0])

    return T_blur

#%%
def vec2streamline_2d(vec_field, seed_pts, iters = 10000, epsilon = 0.05):
    all_lines = []
    delta_t = 4

    # the Euler integration function takes a starting point (x, y) and returns a line as list of points
    def euler_intg(x, y, start_vector, line):
        prev_vector = [1, 1]
        first = True
        for j in range(iters):
            vector = vec_field[int(x) - 1, int(y) - 1]
            
            # check for any irregular flipped vectors in the way, so all the sequencial vectors are in the same direction
            if not first and np.dot(vector, prev_vector) < 0:
                vector = [-v for v in vector]
            
            # the starting vector should get assigned outside of the function and leaved unchanged
            if first:
                vector = start_vector
                first = False
                
            # make sure the vector size is not too small
            vec_size = np.sqrt(vector[0]**2 + vector[1]**2)
            if vec_size < epsilon:
                break
            
            # the next point
            x_n = x + delta_t * vector[0]
            y_n = y + delta_t * vector[1]
            
            # stops if the next point is outside the image bounds
            if x_n > img_range[0] or x_n < 0:
                break
            if y_n > img_range[1] or y_n < 0:
                break
            
            line.append([x_n, y_n])                                             # new point added to line list
            
            x = x_n                                                             # current point updated
            y = y_n                                                             # current point updated
            prev_vector = vector                                                # previous vector updated
            
        return line
        
    for i in range(len(seed_pts)):
        line = []
        (x, y) = seed_pts[i]
        line.append([x, y])                                                 # first point added to line list
        
        start_vector = vec_field[int(x) - 1, int(y) - 1]
        # run the Euler integration from each seed points twice; one for each direction
        t1 = threading.Thread(target=euler_intg, args=(x, y, start_vector, line))
        
        (x, y) = seed_pts[i]                                                # start again from the seed point
        new_vector = [-v for v in start_vector]                             # changes to the other direction
        t2 = threading.Thread(target=euler_intg, args=(x, y, new_vector, line))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        all_lines.append(line)                                              # full line added to the list
        
    return all_lines

#%%
'''img = imread('sample.png')
tensor = structure2d(img, 3)

vec_field = np.empty((tensor.shape[0], tensor.shape[1], 2))
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        tensor_field = tensor[i, j]
        eigvals, eigvecs = lin.eigh(tensor_field)                           # sorted
        vec_field[i, j] = eigvecs[:, 0]                                     # smallest eigenvector
'''      
#%%
img = np.ones((453, 806))
#%%
# create random points on the plane
num_points = 200
seed_pts = [(np.random.randint(10, img.shape[0]), np.random.randint(10, img.shape[1])) for _ in range(num_points)]

plt.figure(figsize=(10, 10))

plt.scatter(*zip(*seed_pts), color='red', label='Seed Points')
plt.legend()
plt.grid()
plt.xlim(0, img.shape[0])
plt.ylim(0, img.shape[1])
#plt.show()

#%%
vec_field = np.load('vector_field.npy')
img_range = [img.shape[0], img.shape[1]]
#%%
#all_lines = vec2streamline_2d(vec_field, seed_pts, img_range)
#%%
#np.save('vector_field.npy', vec_field)
#%%
iters = 10000
epsilon = 0.05

all_lines = []
delta_t = 10

# the Euler integration function takes a starting point (x, y) and returns a line as list of points
def euler_intg(x, y, start_vector, line):
    prev_vector = [1, 1]
    first = True
    for j in range(iters):
        vector = vec_field[int(x)-1, int(y)-1]
        
        # check for any irregular flipped vectors in the way, so all the sequencial vectors are in the same direction
        if not first and np.dot(vector, prev_vector) < 0:
            vector = [-v for v in vector]
        
        # the starting vector should get assigned outside of the function and leaved unchanged
        if first:
            vector = start_vector
            first = False
            
        # make sure the vector size is not too small
        vec_size = np.sqrt(vector[0]**2 + vector[1]**2)
        if vec_size < epsilon:
            break
        
        # the next point
        x_n = x + delta_t * vector[0]
        y_n = y + delta_t * vector[1]
        
        # stops if the next point is outside the image bounds
        if x_n > img_range[0] or x_n < 0:
            break
        if y_n > img_range[1] or y_n < 0:
            break
        
        line.append([x_n, y_n])                                             # new point added to line list
        
        x = x_n                                                             # current point updated
        y = y_n                                                             # current point updated
        prev_vector = vector                                                # previous vector updated
        
    return line
    
for i in range(len(seed_pts)):
    line = []
    (x, y) = seed_pts[i]
    line.append([x, y])                                                 # first point added to line list
    start_vector = vec_field[x, y]
    # run the Euler integration from each seed points twice; one for each direction
    t1 = threading.Thread(target=euler_intg, args=(x, y, start_vector, line))
    #euler_intg(x, y, start_vector, line)
    (x, y) = seed_pts[i]                                                # start again from the seed point
    new_vector = [-v for v in start_vector]                             # changes to the other direction
    t2 = threading.Thread(target=euler_intg, args=(x, y, new_vector, line))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    all_lines.append(line)                                              # full line added to the list


#%%

for i in range(num_points):
    plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()

    
    
    
    
    
    
    
    