# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:57:24 2023

@author: helioum

This code is to test the vector2streamline function for a tensor field of an image
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import structure2d as st
import streamlines as sl


I = skimage.io.imread('sample.png')[:,:, 0]
T = st.structure2d(I, 3)
evals, evecs = np.linalg.eigh(T)

V = evecs[:, :, 0]                                  # the vector field is all the smallest vectors
num_points = 200
lines = sl.vec2streamlines(V, num_points, 5)
plt.imshow(I)
for i in range(num_points):
    plt.scatter(*zip(*lines[i]), color='blue', marker='.', label='Seed Points')
    plt.show()
