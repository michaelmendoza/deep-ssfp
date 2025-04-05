import os
import time
import logging
from typing import Tuple, Dict, Optional, Union, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard, 
    Callback
)

import deepssfp
from deepssfp import dataset, models
from .dataset import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeepSSFP')

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


class HistorySaver(Callback):
    """Custom callback to save training history after each epoch."""
    
    def __init__(self, history_dict: Dict[str, List], save_path: str):
        """Initialize the callback.
        
        Parameters
        ----------
        history_dict : Dict[str, List]
            Dictionary to store history values
        save_path : str
            Path to save the history
        """
        super().__init__()
        self.history_dict = history_dict
        self.save_path = save_path
        
    def on_epoch_end(self, epoch, logs=None):
        """Save history at the end of each epoch."""
        logs = logs or {}
        for key in self.history_dict.keys():
            if key in logs:
                self.history_dict[key].append(logs[key])
        
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history_dict['lr'].append(lr)
        print(f"Learning rate: {lr:.4f}")
        
        # Save history to disk
        np.savez(self.save_path, **self.history_dict)


def create_callbacks(
    model_path: str,
    history_dict: Dict[str, List],
    patience: int = 20,
    min_delta: float = 0.001,
    use_early_stopping: bool = True,
    use_reduce_lr: bool = True,
    use_tensorboard: bool = True
) -> List[Callback]:
    """Create a list of callbacks for model training.
    
    Parameters
    ----------
    model_path : str
        Path to save the model (without extension)
    history_dict : Dict[str, List]
        Dictionary to store history values
    patience : int
        Patience for early stopping and learning rate reduction
    min_delta : float
        Minimum change to qualify as improvement
    use_early_stopping : bool
        Whether to use early stopping
    use_reduce_lr : bool
        Whether to use learning rate reduction
    use_tensorboard : bool
        Whether to use TensorBoard logging
        
    Returns
    -------
    List[Callback]
        List of Keras callbacks
    """
    callbacks = []
    
    # Model checkpoint to save best model
    keras_model_path = f"{model_path}.keras"
    callbacks.append(ModelCheckpoint(
        keras_model_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ))
    
    # History saver
    history_path = f"{model_path}_history.npz"
    callbacks.append(HistorySaver(history_dict, history_path))
    
    # Early stopping
    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            restore_best_weights=True
        ))
    
    # Reduce learning rate on plateau
    if use_reduce_lr:
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_delta=min_delta,
            verbose=1,
            min_lr=1e-6
        ))
    
    # TensorBoard logging
    if use_tensorboard:
        log_dir = f"{model_path}_logs"
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
    
    return callbacks


def train(
    mode: str = dataset.modes[0],
    epochs: int = 200,
    lr: float = 1e-3,
    model_name: str = "deepssfp",
    model_dir: str = 'saved_models',
    continue_training: bool = False,
    input_data: Optional[np.ndarray] = None,
    output_data: Optional[np.ndarray] = None,
    custom_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    validation_batch_size: int = 8,
    steps_per_epoch: int = 20,
    validation_steps: int = 10,
    use_early_stopping: bool = True,
    patience: int = 20,
    use_tensorboard: bool = False,
    fine_tune: bool = False,
    fine_tune_model_path: str = '',
    fine_tune_suffix: str = "finetuned",
    fine_tune_lr: float = 1e-5,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dataset, np.ndarray]:
    """Train the DeepSSFP model with support for saving and loading with .keras format.
    
    Parameters
    ----------
    mode : str
        Training mode from dataset.modes
    epochs : int
        Number of epochs to train
    model_name : str
        Base name for the model (will be combined with mode)
    model_dir : str
        Directory to save/load model weights
    continue_training : bool
        If True, load existing weights when available
    input_data : np.ndarray, optional
        Custom input data of shape [slices, height, width, phase_cycles]
    output_data : np.ndarray, optional
        Custom output/target data of shape [slices, height, width, channels]
    custom_dataset : Dataset, optional
        Dataset object to use instead of creating a new one
    batch_size : int
        Batch size for training
    validation_batch_size : int
        Batch size for validation
    steps_per_epoch : int
        Number of steps (batches) per epoch
    validation_steps : int
        Number of validation steps per epoch
    use_early_stopping : bool
        Whether to use early stopping
    patience : int
        Patience for early stopping and learning rate reduction
    use_tensorboard : bool
        Whether to use TensorBoard logging
    fine_tune : bool
        If True, fine-tune an existing model instead of training from scratch
    fine_tune_model_path : str, optional
        Path of the source model to fine-tune (path + filename without extension)
    fine_tune_suffix : str
        Suffix to add to the model name for the fine-tuned model
    fine_tune_lr : float
        Learning rate to use for fine-tuning, typically lower than for initial training

    Returns
    -------
    model : tf.keras.Model
        Trained model
    history : tf.keras.callbacks.History
        Training history
    ds : Dataset
        Dataset used for training
    predictions : np.ndarray
        Predictions on test data
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model path based on model_name and mode
    mode_str = mode.lower().replace(':', '_')

    # For fine-tuning, use a different model name to avoid overwriting the original
    if fine_tune:                    
        model_name = f"{model_name}_{fine_tune_suffix}"
        fine_tune_history_path = f"{fine_tune_model_path}_history.npz"
        fine_tune_model_path = f"{fine_tune_model_path}.keras"

    model_path = os.path.join(model_dir, f"{model_name}_{mode_str}")
    logger.info(f"Model will be saved to: {model_path}")
    
    # Define paths for model and history
    keras_model_path = f"{model_path}.keras"
    history_path = f"{model_path}_history.npz"
    
    # Prepare dataset
    if custom_dataset is not None:
        ds = custom_dataset
    else:
        ds = Dataset(mode, input_data, output_data)

    x_train = ds.x_train
    y_train = ds.y_train
    x_test = ds.x_test
    y_test = ds.y_test

    logger.info(f"Training DataSet: {x_train.shape} | {y_train.shape}")
    logger.info(f"Test DataSet: {x_test.shape} | {y_test.shape}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).shuffle(50).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    valid_dataset = valid_dataset.batch(validation_batch_size).shuffle(50).repeat()

    # Network Parameters
    WIDTH = ds.WIDTH
    HEIGHT = ds.HEIGHT
    CHANNELS = ds.CHANNELS_IN
    NUM_OUTPUTS = ds.CHANNELS_OUT

    # Initialize history tracking
    initial_epoch = 0
    history_dict = {
        'loss': [],
        'val_loss': [],
        'mean_absolute_error': [],
        'val_mean_absolute_error': [],
        'lr': []
    }
    
    # Load existing model or create new one
    if continue_training and os.path.exists(keras_model_path):
        logger.info(f"Loading existing model from {keras_model_path}")
        model = tf.keras.models.load_model(keras_model_path)
        
        # Load training history if it exists
        if os.path.exists(history_path):
            logger.info(f"Loading training history from {history_path}")
            history_data = np.load(history_path)
            for key in history_dict.keys():
                if key in history_data:
                    history_dict[key] = history_data[key].tolist()
            
            initial_epoch = len(history_dict['loss'])
            logger.info(f"Continuing training from epoch {initial_epoch}")

    # Load source model for fine-tuning
    elif fine_tune and os.path.exists(fine_tune_model_path):
        logger.info(f"Loading source model for fine-tuning from {fine_tune_model_path}")
        model = tf.keras.models.load_model(fine_tune_model_path)

        # Use a lower learning rate for fine-tuning
        logger.info(f"Recompiling model with fine-tuning learning rate: {fine_tune_lr}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        # Load training history if it exists
        if os.path.exists(fine_tune_history_path):
            logger.info(f"Loading training history from {fine_tune_history_path}")
            history_data = np.load(fine_tune_history_path)
            for key in history_dict.keys():
                if key in history_data:
                    history_dict[key] = history_data[key].tolist()
            
            initial_epoch = len(history_dict['loss'])
            logger.info(f"Continuing training from epoch {initial_epoch}")
    
    else:
        # Create new model
        logger.info(f"Creating new model with dimensions: {HEIGHT}x{WIDTH}x{CHANNELS}→{NUM_OUTPUTS}")
        model = models.unet_model(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
        logger.info(f"Model created: {model.name}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    # Print model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(
        model_path=model_path,
        history_dict=history_dict,
        patience=patience,
        use_early_stopping=use_early_stopping,
        use_tensorboard=use_tensorboard
    )

    # Train the model
    start_time = time.time()
    
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            verbose=2,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model state...")
        model.save(keras_model_path)
        np.savez(history_path, **history_dict)
        logger.info(f"Model saved to {keras_model_path}")
    
    # Save final model if training completed
    else:
        # Ensure final history is saved
        np.savez(history_path, **history_dict)
        
        # Save final model
        model.save(keras_model_path)
        logger.info(f"Final model saved to {keras_model_path}")
    
    # Evaluate and predict
    try:
        evaluation = model.evaluate(x_test, y_test, verbose=1)
        predictions = model.predict(x_test)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if isinstance(evaluation, list) and len(evaluation) >= 2:
            logger.info(f"Training Complete. Loss: {evaluation[0]:.4f}, MAE: {evaluation[1]:.4f}")
            logger.info(f"Time Elapsed: {training_time:.2f} seconds")
        else:
            logger.info(f"Training Complete. Loss: {evaluation:.4f}")
            logger.info(f"Time Elapsed: {training_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        predictions = None
    
    return model, history_dict, ds, predictions


def get_path(mode: str, model_name: str, model_dir: str) -> str:
    """Get the path to a saved model.
    
    Parameters
    ----------
    mode : str
        Model mode from dataset.modes
    model_name : str
        Base name for the model (will be combined with mode)
    model_dir : str
        Directory where model weights are saved
        
    Returns
    -------
    str
        Path to saved model
    """
    return os.path.join(model_dir, f"{model_name}_{mode.lower().replace(':', '_')}")

def load_model(
    mode: str = dataset.modes[0],
    model_name: str = "deepssfp",
    model_dir: str = 'saved_models',
    custom_shape: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[tf.keras.Model, Optional[Dict]]:
    """Load a trained DeepSSFP model for inference.
    
    Parameters
    ----------
    mode : str
        Model mode from dataset.modes
    model_name : str
        Base name for the model (will be combined with mode)
    model_dir : str
        Directory where model weights are saved
    custom_shape : tuple, optional
        Custom shape tuple (HEIGHT, WIDTH, CHANNELS_IN, CHANNELS_OUT) 
        If None, uses default shape from Dataset class
        
    Returns
    -------
    model : tf.keras.Model
        Loaded model ready for inference
    history_dict : dict, optional
        Training history if available
    """
    # Generate the model path based on model_name and mode
    mode_str = mode.lower().replace(':', '_')
    model_path = os.path.join(model_dir, f"{model_name}_{mode_str}")
    logger.info(f"Loading model from path: {model_path}")
    
    # Define paths for model and history
    keras_model_path = f"{model_path}.keras"
    history_path = f"{model_path}_history.npz"
    
    # Check if model exists
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"No saved model found at {keras_model_path}")
    
    # Load the model
    model = tf.keras.models.load_model(keras_model_path)
    logger.info(f"Model loaded successfully from {keras_model_path}")
    
    # Load history if it exists
    history_dict = None
    if os.path.exists(history_path):
        logger.info(f"Loading training history from {history_path}")
        history_data = np.load(history_path)
        history_dict = {}
        for key in history_data.files:
            history_dict[key] = history_data[key].tolist()
    
    return model, history_dict


def predict(
    model: tf.keras.Model,
    input_data: np.ndarray,
    batch_size: int = 16
) -> np.ndarray:
    """Make predictions using a trained model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained DeepSSFP model
    input_data : np.ndarray
        Input data to make predictions on
    batch_size : int
        Batch size for prediction
        
    Returns
    -------
    np.ndarray
        Model predictions
    """
    logger.info(f"Making predictions on data with shape {input_data.shape}")
    predictions = model.predict(input_data, batch_size=batch_size)
    logger.info(f"Predictions complete. Output shape: {predictions.shape}")
    return predictions


def plot_training_history(
    history_dict: Dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """Plot the training history from a history dictionary.
    
    Parameters
    ----------
    history_dict : Dict
        Dictionary containing training history
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    if not history_dict:
        logger.warning("No history data to plot")
        return
    
    if 'lr' in history_dict and history_dict['lr']:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axs[0].semilogy(history_dict['loss'], label='Training Loss')
    axs[0].semilogy(history_dict['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Log Loss')
    axs[0].legend()
    
    # Plot MAE
    axs[1].semilogy(history_dict['mean_absolute_error'], label='Training MAE')
    axs[1].semilogy(history_dict['val_mean_absolute_error'], label='Validation MAE')
    axs[1].set_title('Mean Absolute Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Log MAE')
    axs[1].legend()
    
    # Plot learning rate if available
    if 'lr' in history_dict and history_dict['lr']:
        axs[2].semilogy(history_dict['lr'], label='Learning Rate')
        axs[2].set_title('Learning Rate')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Learning Rate')
        axs[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()
