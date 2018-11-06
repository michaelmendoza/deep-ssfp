
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
    import tensorflow as tf
    from data_loader import DataSet
    import model

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time

def runNetwork(modeindex, doRestore = False):

    # Import Dataset
    modes = DataSet.learningModes;
    data = DataSet(modes[modeindex])
    data.print()

    # Training Parameters
    learning_rate = 1e-4
    num_steps = 50000
    batch_size = 16
    display_step = 1000
    save_step = 10000
    
    # Network Parameters
    WIDTH = data.WIDTH
    HEIGHT = data.HEIGHT
    CHANNELS = data.CHANNELS_IN
    NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
    NUM_OUTPUTS = data.CHANNELS_OUT

    # Network Varibles and placeholders
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
    Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Define loss and optimizer
    prediction = model.unet(X, NUM_OUTPUTS)
    loss = tf.reduce_mean(tf.square(prediction - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer = optimizer.minimize(loss, global_step=global_step)

    # Setup Saver
    saver = tf.train.Saver()

    # Initalize varibles, and run network
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if(doRestore):
        ckpt = ckpt = tf.train.get_checkpoint_state('./checkpoints/' + modes[modeindex])
        if(ckpt and ckpt.model_checkpoint_path):
            print('Restoring Prev. Model ....')
            saver.restore(sess,  ckpt.model_checkpoint_path)
            print('Model Loaded....')

    print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

    # Train network
    _step = []
    _loss_train = []
    _loss_test = []

    t0 = time()
    for _ in range(num_steps):
        batch_xs, batch_ys = data.next_batch(batch_size)
        sess.run( trainer, feed_dict={ X: batch_xs, Y: batch_ys } )

        step = sess.run(global_step)

        if(step % display_step == 0):
            train_loss = sess.run(loss, feed_dict={ X:batch_xs, Y:batch_ys })
            test_loss = sess.run(loss, feed_dict={ X: data.x_test, Y: data.y_test })
            print("Step: " + str(step) + " Train Loss: %.4e" % train_loss + " Test Loss: %.4e" % test_loss + " TIME: %g" % (time() - t0))
            _step.append(step); _loss_test.append(test_loss); _loss_train.append(train_loss)

        if(step % save_step == 0):
            saver.save(sess, './checkpoints/' + modes[modeindex] + '/' + modes[modeindex], global_step=global_step)

    # Show results
    prediction = sess.run(prediction, feed_dict={ X: data.x_test, Y: data.y_test })

    plot(data, prediction, modeindex, 0)

    # Plot loss
    plt.plot(_step, np.log10(_loss_train), label='training loss')
    plt.plot(_step, np.log10(_loss_test), label='test loss')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epoches')
    plt.ylabel('ln(MSE)')
    plt.legend()
    plt.show()

def plotSavedModel(modeindex):
    
    # Import Dataset
    modes = DataSet.learningModes;
    data = DataSet(modes[modeindex])
    data.print()

    # Network Parameters
    WIDTH = data.WIDTH
    HEIGHT = data.HEIGHT
    CHANNELS = data.CHANNELS_IN
    NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
    NUM_OUTPUTS = data.CHANNELS_OUT

    # Network Varibles and placeholders
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
    Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Define loss and optimizer
    prediction = model.unet(X, NUM_OUTPUTS)

    # Setup Saver
    saver = tf.train.Saver()

    # Initalize varibles, and run network
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ckpt = ckpt = tf.train.get_checkpoint_state('./checkpoints/' + modes[modeindex])
    if(ckpt and ckpt.model_checkpoint_path):
        print('Restoring Prev. Model ....')
        saver.restore(sess,  ckpt.model_checkpoint_path)
        print('Model Loaded....')

        # Show results
        prediction = sess.run(prediction, feed_dict={ X: data.x_test, Y: data.y_test })

        index = np.random.randint(data.x_test.shape[0])
        print('Selecting Test Image #', index)
        plot(data, prediction, modeindex, index)

def plot(data, prediction, modeindex, index = 0):
    if modeindex == 3:
        data.plot_evenodd(data.x_test[index], data.y_test[index], prediction[index])
    elif modeindex == 2:
        data.plot_synthetic_banding(data.x_test[index], data.y_test[index], prediction[index])
    else:
        data.plot(data.x_test[index], data.y_test[index], prediction[index])