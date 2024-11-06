import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from deepssfp import dataset, models

def train(mode=dataset.modes[0], epochs=200, model_dir='saved_models', 
          continue_training=False, input_data=None, output_data=None):
    """Train the DeepSSFP model with support for saving and loading weights.
    
    Parameters
    ----------
    mode : str
        Training mode from dataset.modes
    epochs : int
        Number of epochs to train
    model_dir : str
        Directory to save/load model weights
    continue_training : bool
        If True, load existing weights when available
    input_data : ndarray, optional
        Custom input data of shape [slices, height, width, phase_cycles]
    output_data : ndarray, optional
        Custom output/target data of shape [slices, height, width, channels]
    """
    
    # Training Parameters
    batch_size = 16
    test_batch_size = 8
    validation_split = 0.2
    shuffle = True

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate a model name based on the mode and parameters
    model_name = f"deepssfp_{mode.lower().replace(':', '_')}"
    model_path = os.path.join(model_dir, model_name)

    ds = dataset.Dataset(mode, input_data, output_data)

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
    CHANNELS = ds.CHANNELS_IN
    NUM_OUTPUTS = ds.CHANNELS_OUT

    # Create model
    model = models.unet_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    print(f'DL Model: {HEIGHT}, {WIDTH}, {CHANNELS}, {NUM_OUTPUTS}')

    model.compile(optimizer='adam', 
                 loss=tf.keras.losses.MeanSquaredError(), 
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    # Load weights if continuing training and weights exist
    initial_epoch = 0
    if continue_training and os.path.exists(f"{model_path}.index"):
        print(f"Loading existing model weights from {model_path}")
        model.load_weights(model_path)
        
        # Load training history if it exists
        history_path = f"{model_path}_history.npy"
        if os.path.exists(history_path):
            print("Loading training history")
            history_dict = np.load(history_path, allow_pickle=True).item()
            initial_epoch = len(history_dict['loss'])
            print(f"Continuing training from epoch {initial_epoch}")
    
    model.summary()

    # Create ModelCheckpoint callback to save best weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    start = time.time()
    history = model.fit(
        train_dataset, 
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=20,
        validation_data=valid_dataset,
        validation_steps=10,
        verbose=2,
        callbacks=[checkpoint_callback]
    )
    
    # Save training history
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mean_absolute_error': history.history['mean_absolute_error'],
        'val_mean_absolute_error': history.history['val_mean_absolute_error']
    }
    np.save(f"{model_path}_history.npy", history_dict)
    
    evaluation = model.evaluate(x_test, y_test, verbose=1)
    predictions = model.predict(x_test)
    end = time.time()

    print("Training Complete.")
    print('Summary: Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
    
    return model, history, ds, predictions


def load_model(mode=dataset.modes[0], model_dir='saved_models', data_shape=None):
    """Load a trained DeepSSFP model for inference.
    
    Parameters
    ----------
    mode : str
        Model mode from dataset.modes
    model_dir : str
        Directory where model weights are saved
    custom_shape : tuple, optional
        Custom shape tuple (HEIGHT, WIDTH, CHANNELS_IN, CHANNELS_OUT) 
        If None, uses default shape from Dataset class
        
    Returns
    -------
    model : tf.keras.Model
        Loaded model ready for inference
    """
    # Generate the model name based on the mode
    model_name = f"deepssfp_{mode.lower().replace(':', '_')}"
    model_path = os.path.join(model_dir, model_name)
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.index"):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    
    # Get model parameters either from custom_shape or dataset
    if data_shape is not None:
        HEIGHT, WIDTH, CHANNELS_IN, CHANNELS_OUT = data_shape
    else:
        # Create dummy dataset to get shapes
        ds = dataset.Dataset(mode)
        HEIGHT = ds.HEIGHT
        WIDTH = ds.WIDTH
        CHANNELS_IN = ds.CHANNELS_IN
        CHANNELS_OUT = ds.CHANNELS_OUT
    
    # Create and compile model
    model = models.unet_model(HEIGHT, WIDTH, CHANNELS_IN, CHANNELS_OUT)
    model.compile(optimizer='adam', 
                 loss=tf.keras.losses.MeanSquaredError(), 
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    # Load weights
    print(f"Loading model weights from {model_path}")
    model.load_weights(model_path)
    
    return model