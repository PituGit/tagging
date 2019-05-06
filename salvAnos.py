import KMeans as km
import numpy as np
from scipy.spatial import distance_matrix
from skimage import io
from skimage import color
import ColorNaming as cn

X = np.array([[239, 44, 35], [179, 267, 254], [175, 158, 1], [204, 21, 32], [284, 205, 239], [265, 58, 279]])
C = np.array([[211, 116, 22], [202, 13, 50]])
D = km.distance(X, C)
F = distance_matrix(X, C)

print(D)
print("--------")
print(F) 
print("--------")
""""
km = km.KMeans(X,2)
km.centroids = C
km._cluster_points()
J = km.clusters[:100]

print(J)
"""
print("--------")
"""
km._get_centroids()
print(km.centroids)
"""
print("--------")

im = io.imread('Images/0065.jpg')
options = {'verbose': False, 'km_init': 'custom'}
km2 = km.KMeans(im,3, options)
km2.custom(3)
print(km2.centroids)

print("--------")

im1 = color.rgb2lab(im)
print(im1[41][31])

print("--------")

im2 = cn.ImColorNamingTSELabDescriptor(im)
print(im2[17][65])

print("--------")

