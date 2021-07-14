import numpy as np
import _neighbour_array
import _image_processing
import curveshortening
import _image_curve
import os
import matplotlib.pyplot as plt

inputx = np.hstack((np.arange(10), 10 * np.ones(10), np.arange(1, 11)[::-1], np.zeros(10))).astype(int) + 1
inputy = np.hstack((np.zeros(10), np.arange(10), 10 * np.ones(10), np.arange(1, 11)[::-1])).astype(int) + 1

input = np.vstack((inputx, inputy))

output = np.zeros((13, 13))
output[tuple(p for p in zip(*input.transpose()))] = 1

im_recovered = _image_curve.curve_to_image_matrix(input.transpose(), (13, 13))


plt.imshow(output)
plt.figure()
plt.imshow(im_recovered)
plt.show()