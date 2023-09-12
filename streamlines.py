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
import scipy as sp
import threading
import scipy.interpolate
#%%
# 1) seed (x0, y0)
# 2) Euler step (xn, yn)
# 2a) Evaluate tensor T(xn, yn) interpolate
# 2b) Eigendecomposition V(xn, yn)
# 2c) Dot product to get V_hat(xn, yn)

# returns: list of numpy arrays containing points for each streamline
#          seed points
def tensor2streamlines(T, S, evec = 0, dt = 4.0, epsilon = 0.5):
    
    # for this function, we use X to represent the 0 index and Y to represent the 1 index
    # start with an empty list of lines
    all_lines = []

    # create splines representing the vector fields
    xx_spline = sp.interpolate.RectBivariateSpline(range(T.shape[0]), range(T.shape[1]), T[:, :, 0, 0])
    yy_spline = sp.interpolate.RectBivariateSpline(range(T.shape[0]), range(T.shape[1]), T[:, :, 1, 1])    
    xy_spline = sp.interpolate.RectBivariateSpline(range(T.shape[0]), range(T.shape[1]), T[:, :, 0, 1])
    
    # generate seed points
    seeds = np.zeros((S, 2))
    seeds[:, 0] = np.random.randint(1, T.shape[0] - 1, S)
    seeds[:, 1] = np.random.randint(1, T.shape[1] - 1, S)
    
    # for each seed point
    for si in range(S):

        # create a line tracing the vector field forward
        line_forward = []
        
        # initialize the seed point
        x0 = seeds[si, 0]
        y0 = seeds[si, 1]
        
        # evaluate tensor from interpolation
        T0 = np.zeros((2, 2))
        T0[0, 0] = np.squeeze(xx_spline(x0, y0))
        T0[1, 1] = np.squeeze(yy_spline(x0, y0))
        T0[0, 1] = np.squeeze(xy_spline(x0, y0))        
        T0[1, 0] = T0[0, 1]
        
        # eigendecompostion
        evals, evecs = np.linalg.eigh(T0)       # ascending order
        V = evecs[:, evec]                         # largest eigenvector
        
        
        # trace the streamlines until you reach the edge of the vector field
        while x0 < T.shape[0] and x0 > 0 and y0 < T.shape[1] and y0 > 0:
            line_forward.append((x0, y0))
            
            # Euler step
            xn = x0 + dt * V[0]
            yn = y0 + dt * V[1]
            
            # evaluate tensor from interpolation of T(xn, yn)
            Tn = np.zeros((2,2))
            Tn[0, 0] = np.squeeze(xx_spline(xn, yn))
            Tn[1, 1] = np.squeeze(yy_spline(xn, yn))
            Tn[0, 1] = np.squeeze(xy_spline(xn, yn))        
            Tn[1, 0] = Tn[0, 1]
            
            # eigendecompostion
            evals, evecs = np.linalg.eigh(Tn)
            Vn = evecs[:, evec]
            
            # check for dot product
            #prev_vector = np.array(xn - x0, yn - y0)
            dot = np.dot(Vn, np.array([xn - x0, yn - y0]))
            if (dot < 0):
                break #V = [-v for v in Vn]
            else:
                V = Vn
            
            if abs(evals[evec]) < epsilon:
                print('small vector')
                break
            
            x0 = xn
            y0 = yn
            
        # do the same thing tracing the vector field backwards
        line_backward = []
        
        # initialize the seed point
        x0 = seeds[si, 0]
        y0 = seeds[si, 1]
        
        # eigendecompostion
        evals, evecs = np.linalg.eigh(T0)                   # ascending order
        V = np.asarray([-e for e in evecs[:, evec]])           # flips the initial vector
        
        while x0 < T.shape[0] and x0 > 0 and y0 < T.shape[1] and y0 > 0:
            line_backward.append((x0, y0))
            
            # Euler step
            xn = x0 + dt * V[0]
            yn = y0 + dt * V[1]
            
            # evaluate tensor from interpolation of T(xn, yn)
            Tn = np.zeros((2,2))
            Tn[0, 0] = np.squeeze(xx_spline(xn, yn))
            Tn[1, 1] = np.squeeze(yy_spline(xn, yn))
            Tn[0, 1] = np.squeeze(xy_spline(xn, yn))        
            Tn[1, 0] = Tn[0, 1]
            
            # eigendecompostion
            evals, evecs = np.linalg.eigh(Tn)
            Vn = evecs[:, evec]
            
            # check for dot product
            dot = np.dot(Vn, np.array([xn - x0, yn - y0]))
            # flip the vector if dot product is negatives
            if (dot < 0):
                V = np.asarray([-v for v in Vn])
            else:
                V = Vn
            
            if abs(evals[evec]) < epsilon:
                print('small vector')
                break
            
            x0 = xn
            y0 = yn
        # backward lines are reverses, minus the initial seed point because is already added to the forward line
        all_lines.append(np.asarray(line_backward[::-1][:-1] + line_forward))
        
    return all_lines, seeds
#%%
# def vec2streamlines(V, seeds = 100, dt = 1.0, iters = 10000, epsilon = 0.05)

# if seeds is a scalar, generate that many random seed points
# otherwise use the provided list of seed points

# return a list of lines

def vec2streamlines_old(V, seeds, dt = 4.0, iters = 10000, epsilon = 0.05):
    all_lines = []
    x_bound, y_bound, z_bound = V.shape[0], V.shape[1], V.shape[2]
    
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
            
            # check for any irregular flipped vectors in the way, so all vectors stay in the same direction
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
    x_spline = sp.RectBivariateSpline(x_range, x_range, vec_field[:,:,0])      # bivariate spline approximation over a rectangular mesh
    y_spline = sp.RectBivariateSpline(x_range, x_range, vec_field[:,:,1])
    
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
def vec2streamlines_new(V, S, dt = 3.0, iters = 10000, epsilon = 0.05):
    
    # start with an empty list of lines
    all_lines = []
    
    # create splines representing the vector fields
    x_spline = sp.interpolate.RectBivariateSpline(range(V.shape[0]), range(V.shape[1]), V[:, :, 0])
    y_spline = sp.interpolate.RectBivariateSpline(range(V.shape[0]), range(V.shape[1]), V[:, :, 1])
    
    # generate seed points
    seeds = np.zeros((S, 2))
    seeds[:, 0] = np.random.randint(1, V.shape[0], S)
    seeds[:, 1] = np.random.randint(1, V.shape[1], S)
    
    # for each seed point
    for si in range(S):

        # create a line tracing the vector field forward
        line_forward = []
        
        # initialize the seed point
        x0 = seeds[si, 0]
        y0 = seeds[si, 1]
        dx0dt = np.squeeze(x_spline(x0, y0))
        dy0dt = np.squeeze(y_spline(x0, y0))
        
        # trace the streamlines until you reach the edge of the vector field
        while x0 < V.shape[0] and x0 > 0 and y0 < V.shape[1] and y0 > 0:
            line_forward.append((x0, y0))
            
            dxdt = np.squeeze(x_spline(x0, y0))
            dydt = np.squeeze(y_spline(x0, y0))
            
            if dx0dt * dxdt + dy0dt * dydt < 0:
                dxdt = -dxdt
                dydt = -dydt
                
            # Euler step
            x0 = x0 + dt * dxdt
            y0 = y0 + dt * dydt
            
            dx0dt = dxdt
            dy0dt = dydt
            
        # do the same thing tracing the vector field backwards
        line_backward = []
        
        # initialize the seed point
        x0 = seeds[si, 0]
        y0 = seeds[si, 1]
        line_backward.append((x0, y0))
        
        dx0dt = np.squeeze(x_spline(x0, y0))
        dy0dt = np.squeeze(y_spline(x0, y0))
        
        # Euler step
        x0 = x0 - dt * dxdt
        y0 = y0 - dt * dydt
        
        while x0 < V.shape[0] and x0 > 0 and y0 < V.shape[1] and y0 > 0:
            line_backward.append((x0, y0))
            
            dxdt = np.squeeze(x_spline(x0, y0))
            dydt = np.squeeze(y_spline(x0, y0))
            
            if dx0dt * dxdt + dy0dt * dydt < 0:
                dxdt = -dxdt
                dydt = -dydt
            
            # Euler step
            x0 = x0 + dt * dxdt
            y0 = y0 + dt * dydt
            
            dx0dt = dxdt
            dy0dt = dydt
                  
        all_lines.append(np.asarray(line_backward[::-1][:-1] + line_forward))
    return all_lines