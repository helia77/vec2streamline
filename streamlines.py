# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:35:14 2023

@author: helioum

This code is a function that gets a vector field for input and produces streamline as the output. 
The vector field is basically the two (three for 3D volume) eigenvectors calculated from the tensor field.

"""

#%%
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import threading


# def vec2streamlines(V, seeds = 100, dt = 1.0, iters = 10000, epsilon = 0.05)

# if seeds is a scalar, generate that many random seed points
# otherwise use the provided list of seed points

# return a list of lines

def vec2streamlines(V, seeds, dt = 4.0, iters = 10000, epsilon = 0.05):
    all_lines = []
    x_bound, y_bound, z_bound = V.shape
    
    # if seeds is a scalar, generate that many random seed points
    if isinstance(seeds, int):
        if   len(V.shape) == 3:                      # for 2D images
            seed_pts = [(np.random.randint(1, x_bound), np.random.randint(1, y_bound)) for _ in range(seeds)]
        elif len(V.shape) == 4:                      # for 3D volumes
            seed_pts = [(np.random.randint(1, x_bound), np.random.randint(1, y_bound), np.random.randint(1, z_bound)) for _ in range(seeds)]
            
    elif isinstance((seeds), list):
        seed_pts = seeds
    else:
        print("Wrong data type.")
        
    # the Euler integration function takes a starting point (x, y), returns a line (a list of points)
    def euler_intg(x, y, z, start_vector, line):
        prev_vector = [1, 1] if z is None else [1, 1, 1]
        first = True
        
        for j in range(iters):
            vector = V[int(x) - 1, int(y) - 1] if z is None else V[int(x) - 1, int(y) - 1, int(z) - 1]
            
            # check for any irregular flipped vectors in the way, so all the sequencial vectors stay in the same direction
            if not first and np.dot(vector, prev_vector) < 0:
                vector = [-v for v in vector]
            
            # the starting vector is assigned outside of the function and should left unchanged
            if first:
                vector = start_vector
                first = False
                
            # make sure the vector size is not too small
            vec_size = np.linalg.norm(vector)
            if vec_size < epsilon:
                break
            
            # the next point
            x_n = x + dt * vector[0]
            y_n = y + dt * vector[1]
            z_n = z + dt * vector[2] if z is not None else 0
                
            # stops if the next point is outside the image bounds
            if x_n > x_bound or x_n < 0:
                break
            if y_n > y_bound or y_n < 0:
                break
            if z_n > z_bound or z_n < 0:
                break
            
            if len(V.shape) == 4:
                line.append([z_n, y_n, x_n])
            else:
                line.append([y_n, x_n])                                         # new point added to line list
            
            x = x_n                                                             # current point updated
            y = y_n                                                             # current point updated
            z = z_n if z is not None else None
            prev_vector = vector                                                # previous vector updated
            
        return line
        
    for i in range(len(seed_pts)):
        line = []
        if   len(V.shape) == 3:
            (x, y) = seed_pts[i]
            line.append([y, x])                                                     # first point added to line list
            start_vector = V[int(x) - 1, int(y) - 1]
            z = None
            
        elif len(V.shape) == 4:
            (x, y, z) = seed_pts[i]
            line.append([z, y, x])                                                  # first point added to line list
            start_vector = V[int(x) - 1, int(y) - 1, int(z) - 1]
        
        # run the Euler integration from each seed points twice; one for each direction
        t1 = threading.Thread(target=euler_intg, args=(x, y, z, start_vector, line))
        
        if   len(V.shape) == 3:
            (x, y) = seed_pts[i]                                                    # start over from the seed point
            z = None
        elif len(V.shape) == 4:
            (x, y, z) = seed_pts[i]
            
        new_vector = [-v for v in start_vector]                                     # changes to the other direction
        t2 = threading.Thread(target=euler_intg, args=(x, y, z, new_vector, line))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        all_lines.append(line)                                              # full line added to the list
        
    return all_lines




#%%
# inputs:
    # vec_field:    a XxYx2 array containing the tensor field
    # seed_pts:     a list of seed points 
    # img_range:    the ranges for x and y axis of the iamge as two variables
    # iters:        number of iterations for the Euler integration
    # epsilon:      smallest value to compare the smallest possible vector size with

    
def vec2streamline_2d(vec_field, seed_pts, img_range, iters = 10000, epsilon = 0.05):
    all_lines = []
    delta_t = 0.5

    # Create a RectBivariateSpline for both dimensions (x and y)
    x_range =  np.linspace(-10, 10, 100)                                    # coordinates in ascending orver
    x_spline = RectBivariateSpline(x_range, x_range, vec_field[:,:,0])      # bivariate spline approximation over a rectangular mesh
    y_spline = RectBivariateSpline(x_range, x_range, vec_field[:,:,1])
    
    # the Euler integration function takes a starting point (x, y) and returns a line as list of points
    def euler_intg(x, y, start_vector, line):
        prev_vector = [1, 1]
        first = True
        for j in range(iters):
            # finding the new vector at this point
            interpolated_x = np.squeeze(x_spline(x, y))
            interpolated_y = np.squeeze(y_spline(x, y))
            
            vector = [interpolated_x, interpolated_y]
            
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
            if x_n > img_range[0][-1] or x_n < img_range[0][0]:
                break
            if y_n > img_range[1][-1] or y_n < img_range[1][0]:
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
        # finding the vector at the starting point
        interpolated_x = np.squeeze(x_spline(x, y))
        interpolated_y = np.squeeze(y_spline(x, y))
        
        start_vector = [interpolated_x, interpolated_y]
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

