from __future__ import division, print_function

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scip.io as sio
import os


fdir = '10282017_SSPF Smoothing DL'
files = os.listdir(fdir)
print(files)

img = sio.loadmat(files[0])['img']

