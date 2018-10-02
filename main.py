from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data import DataSet
import model 

# Import Dataset
modes = DataSet.learningModes;
data = DataSet(modes[2])
data.print()
