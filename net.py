
'''
Tensorflow Code for a color segmentation network
'''

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
data = DataSet(True)
data.print()

# Training Parameters
learning_rate = 1e-4
num_steps = 10000
batch_size = 16
display_step = 100

# Network Parameters 
WIDTH = data.WIDTH 
HEIGHT = data.HEIGHT
CHANNELS = data.CHANNELS_IN
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = data.CHANNELS_OUT

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output

# Define loss and optimizer
prediction = model.unet(X)
loss = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

# Train network
_step = []
_loss = []
for step in range(num_steps):
    batch_xs, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={ X: batch_xs, Y: batch_ys } )

    if(step % display_step == 0):
      test_loss = sess.run(loss, feed_dict={ X: data.x_test, Y: data.y_test })
      print("Step: " + str(step) + " LOSS: " + str(test_loss))
      _step.append(step); _loss.append(test_loss) 

# Show results
prediction = sess.run(prediction, feed_dict={ X: data.x_test, Y: data.y_test })
data.plot(data.x_test[0], data.y_test[0], prediction[0])

# Plot loss
plt.plot(_step, np.log10(_loss))
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Epoches')
plt.ylabel('ln(MSE)')
plt.show()
