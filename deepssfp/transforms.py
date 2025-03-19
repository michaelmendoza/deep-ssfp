import numpy as np

def ifft(x):
    return np.fft.ifft2(np.fft.fftshift(x, axes=(1,2)), axes=(1,2))

def from_pairs_to_complex(image):
    """Convert real/imaginary channels to complex array.
    
    Parameters
    ----------
    image : np.ndarray
        Image with real and imaginary channels
        
    Returns
    -------
    np.ndarray
        Complex array
    """
    if image.shape[-1] >= 2:
        # If there are multiple channel pairs, use the first pair
        return image[..., 0] + 1j * image[..., 1]
    else:
        # If there's only one channel, use it as magnitude
        return image[..., 0]
