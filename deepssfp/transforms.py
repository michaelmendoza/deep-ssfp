import numpy as np

def ifft(x):
    return np.fft.ifft2(np.fft.fftshift(x, axes=(1,2)), axes=(1,2))

import numpy as np

def from_complex_to_pairs(complex_image):
    """Convert complex array to real/imaginary channels.
    
    Parameters
    ----------
    complex_image : np.ndarray
        Complex array with shape [..., n]
        
    Returns
    -------
    np.ndarray
        Image with real and imaginary channels with shape [..., 2*n]
    """
    
    if len(complex_image.shape) == 3:
        complex_image = complex_image[..., np.newaxis]

    # Get the shape of the input array
    input_shape = complex_image.shape
    
    # Calculate the output shape by doubling the last dimension
    output_shape = input_shape[:-1] + (input_shape[-1] * 2,)
    
    # Create a new array with doubled last dimension
    result = np.zeros(output_shape, dtype=float)
    
    # Fill in the real and imaginary components for each channel
    for i in range(input_shape[-1]):
        result[..., 2*i] = np.real(complex_image[..., i])
        result[..., 2*i+1] = np.imag(complex_image[..., i])
    
    return result

def from_pairs_to_complex(image):
    """Convert real/imaginary channels to complex array.
    
    Parameters
    ----------
    image : np.ndarray
        Image with real and imaginary channels with shape [..., 2*n]
        
    Returns
    -------
    np.ndarray
        Complex array with shape [..., n]
    """

    if image.shape[-1] < 2:
        # If there's only one channel, use it as magnitude
        return image[..., 0]
    
    if image.shape[-1] == 2:
        # If there are one channel pairs, use the first pair
        return image[..., 0] + 1j * image[..., 1]

    # Ensure the last dimension is even
    if image.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even to represent pairs of real/imaginary components")
    
    # Calculate the output shape by halving the last dimension
    output_shape = image.shape[:-1] + (image.shape[-1] // 2,)
    
    # Create the complex array
    result = np.zeros(output_shape, dtype=complex)
    
    # Fill in the complex values by combining real and imaginary parts
    for i in range(output_shape[-1]):
        result[..., i] = image[..., 2*i] + 1j * image[..., 2*i+1]
    
    return result

def combine_synthetic_banding_datasets(x, y):
    """Combines datasets from SyntheticBanding experiments.
    
    Parameters
    ----------
    x : np.ndarray
        Input data with shape [slices, height, width, channels]
    y : np.ndarray
        Target data with shape [slices, height, width, channels]
        
    Returns
    -------
    np.ndarray
        Combined input data with shape [slices, height, width, channels * 2]
    """
    # Get the shape of the input array
    input_shape = x.shape
    
    # Calculate the output shape by doubling the last dimension
    output_shape = input_shape[:-1] + (input_shape[-1] * 2,)
    
    if (x.dtype == float or x.dtype == np.float32 or x.dtype == np.float64): # Using real/imag pairs
        # Create a new array with doubled last dimension
        result = np.zeros(output_shape, dtype=float)

        # Fill in the real and imaginary components for each channel
        result[:,:,:,0] = x[:,:,:,0]
        result[:,:,:,1] = x[:,:,:,1]
        result[:,:,:,2] = y[:,:,:,0]
        result[:,:,:,3] = y[:,:,:,1]
        result[:,:,:,4] = x[:,:,:,2]
        result[:,:,:,5] = x[:,:,:,3]
        result[:,:,:,6] = y[:,:,:,2]
        result[:,:,:,7] = y[:,:,:,3]
    else: # Using complex data
        result = np.zeros(output_shape, dtype=complex)
        result[:,:,:,0] = x[:,:,:,0]
        result[:,:,:,1] = x[:,:,:,1]
        result[:,:,:,2] = y[:,:,:,0]
        result[:,:,:,3] = y[:,:,:,1]

    return result
