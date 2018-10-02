
import tensorflow as tf 

def simple_net(x, NUM_OUTPUTS = 2):

    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    prediction = tf.layers.conv2d(conv2, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None, kernel_initializer=he_init, name='output')
    return prediction

def simple_unet(x, NUM_OUTPUTS = 2):

    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    up1 = tf.layers.conv2d_transpose(conv4, 64, [3, 3], strides=2, padding="SAME", name='Up1')

    conv5 = tf.layers.conv2d(up1,   32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv6 = tf.layers.conv2d(conv5, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    prediction = tf.layers.conv2d(conv6, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None,       kernel_initializer=he_init, name='Output')
    return prediction

def unet(x, NUM_OUTPUTS = 2):
    print('Unet: Output ', NUM_OUTPUTS);

    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv1 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1-2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    conv2 = tf.layers.conv2d(conv2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv3 = tf.layers.conv2d(conv3, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)    

    conv4 = tf.layers.conv2d(pool3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    conv4 = tf.layers.conv2d(conv4, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)    

    conv5 = tf.layers.conv2d(pool4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2')
 
    up6 = tf.layers.conv2d_transpose(conv5, 256, [3, 3], strides=2, padding="SAME", name='Up6')
    up6 = tf.concat([up6, conv4], 3)
    conv6 = tf.layers.conv2d(up6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    conv6 = tf.layers.conv2d(conv6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6-2')
    
    up7 = tf.layers.conv2d_transpose(conv6, 128, [3, 3], strides=2, padding="SAME", name='Up7')
    up7 = tf.concat([up7, conv3], 3)
    conv7 = tf.layers.conv2d(up7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7')
    conv7 = tf.layers.conv2d(conv7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7-2')

    up8 = tf.layers.conv2d_transpose(conv7, 64, [3, 3], strides=2, padding="SAME", name='Up8')
    up8 = tf.concat([up8, conv2], 3)
    conv8 = tf.layers.conv2d(up8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8')
    conv8 = tf.layers.conv2d(conv8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8-2')
 
    up9 = tf.layers.conv2d_transpose(conv8, 32, [3, 3], strides=2, padding="SAME", name='Up9')
    up9 = tf.concat([up9, conv1], 3)
    conv9 = tf.layers.conv2d(up9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9')
    conv9 = tf.layers.conv2d(conv9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9-2')
    
    prediction = tf.layers.conv2d(conv9, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None, kernel_initializer=he_init, name='Output')
    return prediction