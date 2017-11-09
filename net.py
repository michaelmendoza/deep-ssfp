import tensorflow as tf 
import numpy as np
import matplotlib
from matplotlib.pyplot import cm

import model

# Network Parameters
INPUT_SIZE = 196 #512 #128 #32
INPUT_CHANNELS = 3
INPUT_NUM = INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS
OUT_LABELS = 2

# Training Parameters
learning_rate = 0.0001
num_steps = 10000
batch_size = 4 #16 #128
display_step = 100
scale = 0.001

# tf Graph input
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS])
Y = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, OUT_LABELS])

# Construct model
#logits = conv_net(X)
logits = model.conv_net_simple(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
loss_op = tf.add_n([loss_op] + reg_loss, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
prediction_state = tf.argmax(prediction, 3)
correct_pred = tf.equal(tf.argmax(prediction, 3), tf.argmax(Y, 3))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Init
init = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

# Start training
with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(init)
    
    _step = []; _train_acc = []; _test_acc = []
    for step in range(1, num_steps + 1):
        batch_x, batch_y = data.train_batch(batch_size)
        test_x, test_y = data.test_batch()

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_x, Y: test_y})

            _step.append(step) 
            _train_acc.append(train_acc)
            _test_acc.append(acc)
            print("Step " + str(step) + 
                ", Loss= " + "{:.3f}".format(loss) + \
                ", Train Accuracy= " + "{:.5f}".format(train_acc) + \
                ", Test Accuracy= " + "{:.5f}".format(acc))

    print("Training Finished!")

