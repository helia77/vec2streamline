# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:57:24 2023

@author: helioum

This code is to test the vector2streamline function for a tensor field of an image
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import skimage
import structure2d as st
import streamlines as sl


I = skimage.io.imread('sample.png')[:,:, 0]
T = st.structure2d(I, 3)
evals, evecs = np.linalg.eigh(T)

V = evecs[:, :, 0]
sl.vec2streamline_2d(V, seed_pts, img_range)
