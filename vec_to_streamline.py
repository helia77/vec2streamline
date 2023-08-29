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

#%%
if __name__ == "__main__":
    # create a 2D vector field
    N = 100
    x_range = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(x_range, x_range)
    R = np.sqrt(X**2 + Y**2)
    dy, dx = np.gradient(R)
    
    # create structure tensor
    T = np.zeros((dx.shape[0], dy.shape[0], 2, 2))
    T[:, :, 0, 0] = dx * dx
    T[:, :, 1, 1] = dy * dy
    T[:, :, 0, 1] = dx * dy
    T[:, :, 1, 0] = T[:, :, 0, 1]
    
    # calculate the eigenvectors
    vec_field = np.empty((T.shape[0], T.shape[1], 2))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            tensor_field = T[i, j]
            eigvals, eigvecs = lin.eigh(tensor_field)                           # sorted
            vec_field[i, j] = eigvecs[:, 0]                                     # smallest eigenvector
    
    #%%
    # Create a streamplot
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, dx, dy, density=1.5, linewidth=1, cmap='viridis')
    
    # Add labels and a title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Streamplot Example')
    
    # Show the plot
    plt.show()
    
    #%%
    # create random points on a circle
    '''num_points = 50
    
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    
    radius = 5  # Adjust this radius as needed
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)
    
    seed_pts = []
    for x, y in zip(x_coordinates, y_coordinates):
        if -5 <= x <= 5 and -5 <= y <= 5:
            seed_pts.append((x, y))'''
    
    
    #%%
    # create random points on the plane
    num_points = 50
    seed_pts = [(np.random.uniform(-8.0, 8.0), np.random.uniform(-8.0, 8.0)) for _ in range(num_points)]
    
    #%%
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    major = np.arange(-10, 101, 10)
    minor = np.arange(0, 101, 5)
    ax.set_xticks(major)
    ax.set_xticks(minor, minor=True)
    ax.set_yticks(major)
    ax.set_yticks(minor, minor=True)
    
    plt.imshow(np.zeros_like(X), extent=(-10, 10, -10, 10), cmap='gray')
    plt.scatter(*zip(*seed_pts), color='red', label='Seed Points')
    plt.legend()
    plt.grid()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    #plt.show()
    
    #%%
    img_range = [[-10, 10], [-10, 10]]
    all_lines = vec2streamline_2d(vec_field, seed_pts, img_range)
    
    for i in range(num_points):
        plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
        plt.show()
    #%%
    '''point = [(5, 5)]
    line = vec2streamline_2d(vec_field, point, img_range)
    plt.plot(point[0][0], point[0][1], color='red', marker='o')
    plt.scatter(*zip(*line[0]), color='blue', marker='.', label='Seed Points')'''





