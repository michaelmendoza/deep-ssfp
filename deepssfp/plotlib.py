import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union, List
import deepssfp

def plot_model_history(history):
    ''' Plot Loss History ''' 
    plt.plot(np.log10(history.history['loss']), label='loss')
    plt.plot(np.log10(history.history['val_loss']), label='val_loss')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")
    plt.show()

def is_float(x):
    return x.dtype == float or x.dtype == np.float16 or x.dtype == np.float32 or x.dtype == np.float64

def visualize_images(
        index : Optional[List[int]] = None,
        input_images: Optional[np.ndarray] = None, 
        target_images: Optional[np.ndarray] = None, 
        predicted_images: Optional[np.ndarray] = None, 
        num_samples: int = 1,
        kspace: bool = False, 
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None):
    
    ''' Visualize input images, Plots input, target and predicted image in a row of images. '''

    # Get image indices to visualize
    if isinstance(index, int):
        index = [index]
    elif index is None:
        # Randomly select indices
        Nimages = input_images.shape[0] if input_images is not None else target_images.shape[0] if target_images is not None else predicted_images.shape[0] if predicted_images is not None else 0
        index = np.random.choice(range(Nimages), size=min(num_samples, Nimages), replace=False)
        print(f"Selected indices: {index}")

    num_samples = len(index) if index is not None else num_samples

    # Sample images
    input_images = input_images[index] if input_images is not None else None
    target_images = target_images[index] if target_images is not None else None
    predicted_images = predicted_images[index] if predicted_images is not None else None

    # Convert complex data to real/imag pairs
    if (input_images is not None and not is_float(input_images)):
        input_images = deepssfp.from_complex_to_pairs(input_images)
    if (target_images is not None and not is_float(target_images)):
        target_images = deepssfp.from_complex_to_pairs(target_images)
    if (predicted_images is not None and not is_float(predicted_images)):
        predicted_images = deepssfp.from_complex_to_pairs(predicted_images)

    # Get shapes
    Ninput = input_images.shape[-1] // 2 if input_images is not None else 0
    Ntarget = target_images.shape[-1] // 2 if target_images is not None else 0
    Npredict = predicted_images.shape[-1] // 2 if predicted_images is not None else 0
    num_cols = math.floor(Ninput) + math.floor(Ntarget) + math.floor(Npredict)

    # Calculate figure size
    if figsize is None:
        figsize = (num_cols * 2, num_samples * 2)
    else:
        figsize = (num_cols * figsize[0], num_samples * figsize[1])
    print(f"Figure size: {figsize}")

    # Convert to k-space (using ifft)
    if(kspace):
        if (input_images is not None):
            input_images = np.fft.ifft2(np.fft.fftshift(input_images, axes=(1,2)), axes=(1,2))
        if (target_images is not None):
            target_images = np.fft.ifft2(np.fft.fftshift(target_images, axes=(1,2)), axes=(1,2))
        if (predicted_images is not None):
            predicted_images = np.fft.ifft2(np.fft.fftshift(predicted_images, axes=(1,2)), axes=(1,2))

    fig, axs = plt.subplots(num_samples, num_cols, sharey=True, tight_layout=True, figsize=figsize)

    # Handle case of a single sample or single column
    if num_samples == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_samples == 1:
        axs = axs.reshape(1, -1)
    elif num_cols == 1:
        axs = axs.reshape(-1, 1)

    for row, idx in enumerate(index):
        count = 0
        if (input_images is not None):
            for ii in range(Ninput):
                v = input_images[row,:,:,2*ii] + 1j * input_images[row,:,:,2*ii+1]
                axs[row, count].imshow(np.abs(v), cmap='gray')
                axs[row, count].set_title(f'Input {ii+1}')
                axs[row, count].axis('off')
                count = count + 1

        if (target_images is not None):
            for ii in range(Ntarget):
                v = target_images[row,:,:,2*ii] + 1j * target_images[row,:,:,2*ii+1]
                axs[row, count].imshow(np.abs(v), cmap='gray')
                axs[row, count].set_title(f'Target {ii+1}')
                axs[row, count].axis('off')
                count = count + 1

        if (predicted_images is not None):
            for ii in range(Npredict):
                v = predicted_images[row,:,:,2*ii] + 1j * predicted_images[row,:,:,2*ii+1]
                axs[row, count].imshow(np.abs(v), cmap='gray')
                axs[row, count].set_title(f'Prediction {ii+1}')
                axs[row, count].axis('off')
                count = count + 1

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image visualization saved to {save_path}")

    plt.show()