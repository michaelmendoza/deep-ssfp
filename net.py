
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data import DataSet
data = DataSet()
data.print()

# Training Parameters
learning_rate = 0.0001
num_steps = 10000
batch_size = 32
display_step = 100

# Network Parameters 
WIDTH = 128; HEIGHT = 128; CHANNELS = 3
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 2
NUM_C1 = 32
NUM_C2 = 32

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output

# Define loss and optimizer
logits, prediction = unet(X) #simple_unet(X) # simple_net(X) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model
segmentation = tf.argmax(prediction, 3)
correct_pred = tf.equal(tf.argmax(prediction, 3), tf.argmax(Y, 3))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={ X: batch_xs, Y: batch_ys } )

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={ X: data.x_test, Y: data.y_test })
      _step.append(step)
      _acc.append(acc)

      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for Color Segmentation")
#plt.show()
plt.savefig('results/segmentation-accuracy.png')

# Show results
segmentation = sess.run(segmentation, feed_dict={ X: data.x_test, Y: data.y_test })
print(segmentation.shape)
index = 0;
matplotlib.image.imsave('results/real-img.png', data.unwhiten_img(data.x_test[index]), cmap='gray') 
matplotlib.image.imsave('results/real-test.png', data.y_test[index][:,:,1], cmap='gray') 
matplotlib.image.imsave('results/real-results.png', segmentation[index], cmap='gray') 