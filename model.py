import tensorflow as tf 
import numpy as np

# Create model
def conv_net_simple(x):    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv2 = tf.layers.conv2d(conv1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    out =   tf.layers.conv2d(conv2, OUT_LABELS, [1, 1], padding="SAME", activation=None,       kernel_initializer=he_init, name='Output')
    return out

# Create model
def conv_net(x):    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    up1 = tf.layers.conv2d_transpose(conv4, 64, [3, 3], strides=2, padding="SAME", name='Up1')

    conv5 = tf.layers.conv2d(up1,   32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv6 = tf.layers.conv2d(conv5, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    out =   tf.layers.conv2d(conv6, OUT_LABELS, [1, 1], padding="SAME", activation=None,       kernel_initializer=he_init, name='Output')

    return out
