import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from deepssfp import dataset, models

def train():

    # Training Parameters
    epochs = 100
    batch_size = 16
    test_batch_size = 8
    validation_split = 0.2
    shuffle = True

    mode = dataset.modes[0]
    ds = dataset.Dataset.load()

    x_train = ds.x_train
    y_train = ds.y_train
    x_test = ds.x_test
    y_test = ds.y_test

    print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
    print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(50)
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(50)
    valid_dataset = valid_dataset.repeat()

    # Network Parameters
    WIDTH = ds.WIDTH
    HEIGHT = ds.HEIGHT
    CHANNELS = 8
    NUM_OUTPUTS = 2

    model = models.unet_model_0(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    #model = models.simple_conv(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)

    model.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()

    start = time.time()
    #history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)
    history = model.fit(train_dataset, 
            epochs=epochs, 
            steps_per_epoch=20,
            validation_data=valid_dataset,
            validation_steps = 10)    
    
    evaluation = model.evaluate(x_test, y_test, verbose=1)
    predictions = model.predict(x_test)
    end = time.time()

    print("Training Complete.")
    print('Summary: Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
    
    return model, history, evaluation, predictions
