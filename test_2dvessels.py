# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:57:24 2023

@author: helioum

This code is to test the vector2streamline function for a tensor field of an image
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import structure as st
import streamlines as sl


I = skimage.io.imread('sample.png')[:,:, 0]
T = st.structure2d(I, 5)

num_points = 200
lines, seeds = sl.tensor2streamlines(T, num_points, 0, 10)

plt.imshow(np.transpose(I))
for i in range(num_points):
    #plt.scatter(*zip(*lines[i]), color='blue', marker='.', label='Seed Points')
    plt.plot(lines[i][:, 0], lines[i][:, 1])

#plt.scatter(*zip(*seeds), color='red', marker='.', label='Seed Points')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Streamplot Example')

plt.show()
#%%

vol = np.load('cube.npy')
T2 = st.structure3d(vol, 3, 1)
evals, evecs = np.linalg.eigh(T2)

V = evecs[:, :, :, 0]                                  # the vector field is all the smallest vectors
num_points = 100
lines = sl.vec2streamlines(V, num_points, 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(num_points):
    
    x = [point[0] for point in lines[i]]
    y = [point[1] for point in lines[i]]
    z = [point[2] for point in lines[i]]

    
    ax.scatter3D(x, y, z, c='b', marker='.');
    #plt.scatter(*zip(*lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()