# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:35:14 2023

@author: helioum

This code is a function that gets a vector field for input and produces streamline as the output. 
The vector field is basically the two (three for 3D volume) eigenvectors calculated from the tensor field.

"""

#%%
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
#%%
# inputs:
    # vec_field:    a XxYx2 array containing the tensor field
    # seed_pts:     a list of seed points 
    # img_range:    the ranges for x and y axis of the iamge as two variables
    # iters:        number of iterations for the Euler integration
    # epsilon:      smallest value to compare the smallest possible vector size with
    
def vec2streamline_2d(vec_field, seed_pts, img_range, iters = 10000, epsilon = 0.1):
    all_lines = []
    for i in range(len(seed_pts)):
        line = []
        (x, y) = seed_pts[i]
        
        for _ in range(iters):
            line.append([x, y])                         # first point and the updated points after each iteration added
            
            # mapping the meshgrid range to integer variable to access the vector field as an array
            xi = int((x + 10) * (9/ 20))
            yi = int((y + 10) * (9/ 20))
            vector = vec_field[xi, yi]
            
            vec_size = np.sqrt(vector[0]**2 + vector[1]**2)
            
            if vec_size < epsilon:
                break
            
            x_tmp = vector[0] + x
            y_tmp = vector[1] + y
            
            # stops if the point leaves the image
            if x_tmp > img_range[0][-1] or x_tmp < img_range[0][0]: 
                break
            if y_tmp > img_range[1][-1] or y_tmp < img_range[1][0]: 
                break
            
            x = x_tmp                                   # current point updates
            y = y_tmp                                   # current point updates
        
        line.append([x, y])                             # last point to be updated added
        all_lines.append(line)                          # full line added to the list
        
    return all_lines

#%%

# create a 2D vector field
N = 100
x_range = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x_range, x_range)
R = np.sqrt(X**2 + Y**2)
dx, dy = np.gradient(R)

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
        eigvals, eigvecs = lin.eigh(tensor_field)                   # sorted
        vec_field[i, j] = eigvecs[:, 0]                             # smallest eigenvector
        
#%%

plt.imshow(R, cmap='gray')
plt.show()

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

#num_points = 20
#seed_pts = [[-2.5, -2.5], [-2.5, 0], [-2.5, 2.5], [0, -2.5], [0, 2.5], [2.5, -2.5], [2.5, 0], [2.5, 2.5]]
num_points = 20

angles = np.random.uniform(0, 2 * np.pi, num_points)

radius = 5  # Adjust this radius as needed
x_coordinates = radius * np.cos(angles)
y_coordinates = radius * np.sin(angles)

seed_pts = []
for x, y in zip(x_coordinates, y_coordinates):
    if -5 <= x <= 5 and -5 <= y <= 5:
        seed_pts.append((x, y))

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
#plt.scatter(*zip(*all_lines[1]), color='blue', marker='.', label='Seed Points')
#plt.show()

for i in range(num_points):
    plt.scatter(*zip(*all_lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()





