import numpy as np

modes = ['BandRemoval:4', 'BandRemoval:2', 'SyntheticBanding', 'SuperFOV']

def format_and_prepare_data(x, y, mode):
    ''' Formats and prepares data for deepssfp experiments:
        1) BandRemoval:4 - Transforms complex data into real/img components
        2) BandRemoval:2 - Takes a subset of x data and transforms complex data into real/img components
        3) SyntheticBanding - Takes alternating subsets of data and transforms complex data into real/img components
        4) SuperFOV - Takes alternating even/odd lines of k-space taken from 2 phase cycled acquisitions (k-space), output vector also in k-space. 
        5) SuperFOV - Takes alternating even/odd lines of k-space taken from 2 phase cycled acquisitions (image-space), output vector also in image-space. 
    '''
    if mode == 'BandRemoval:4':
        pass
    elif mode == 'BandRemoval:2':
        x = x[:,:,:,::2]
    elif mode == 'SyntheticBanding':
        y = x[:,:,:,1::2]
        x = x[:,:,:,::2]
    elif mode == 'SuperFOV' or mode == 'SuperFOVi':
        x = x[:,:,:,::2]
        x = np.fft.fftshift(np.fft.fft2(x, axes=(1,2)), axes=(1,2))
        y = np.fft.fftshift(np.fft.fft2(y, axes=(1,2)), axes=(1,2))
    else:
        raise Exception('Invalid data mode')

    x = complex_to_real_img(x)
    y = complex_to_real_img(y)

    if mode == 'SuperFOV' or mode == 'SuperFOVi':
        sx = x.shape
        _x = np.zeros((sx[0], sx[1], sx[2], 2))
        _x[:,::2,:,0] = x[:,::2,:,0]
        _x[:,1::2,:,0] = x[:,1::2,:,2]
        _x[:,::2,:,1] = x[:,::2,:,1]
        _x[:,1::2,:,1] = x[:,1::2,:,3]
        x = _x

    if mode == 'SuperFOVi':
        x = real_imag_to_complex(x)
        y = real_imag_to_complex(y)
        x = np.fft.ifft2(np.fft.fftshift(x, axes=(1,2)), axes=(1,2))
        y = np.fft.ifft2(np.fft.fftshift(y, axes=(1,2)), axes=(1,2))
        x = complex_to_real_img(x)
        y = complex_to_real_img(y)
    return x, y

def complex_to_real_img(x):
    ''' Transforms complex numpy array into arrays with real/img component channels '''

    sx = x.shape
    if (len(sx) == 3):
        x = x.reshape(x.shape + (1,))
        _sx = (sx[0], sx[1], sx[2], 2)
    elif(len(sx) == 4):
        _sx = (sx[0], sx[1], sx[2], int(2 * sx[3]))
    else:
        raise Exception('Invalid shape for x')

    _x = np.zeros(_sx)

    for n in range( int(_sx[3] / 2) ):
        _x[:,:,:,2*n] = x[:,:,:,n].real
        _x[:,:,:,2*n+1] = x[:,:,:,n].imag
    return _x

def real_imag_to_complex(x):
    ''' Converts real and imaginary dims for a tensor with a complex dim to a complex tensor '''

    s = x.shape
    out = np.zeros((s[0], s[1], s[2], s[3] // 2), dtype=complex)
    for ii in range(s[3] // 2):
        out[:,:,:,ii] = x[:,:,:,2*ii] + 1j * x[:,:,:,2*ii+1]
    return out

def format_synthetic_banding_for_realimag_data(data):
    # Data shape: (Slices, Height, Width, 2*num_phase_cycles)
    
    # Get shape info
    sx = data.shape
    total_channels = sx[3]
    
    # Each phase cycle takes 2 channels (real and imaginary)
    num_phase_cycles = total_channels // 2
    channels_per_output = num_phase_cycles  # Half go to x, half to y
    
    # Create output arrays
    x = np.zeros((sx[0], sx[1], sx[2], channels_per_output), dtype=np.float16)
    y = np.zeros((sx[0], sx[1], sx[2], channels_per_output), dtype=np.float16)
    
    # Assign phase cycles to x and y
    # Formatting data for SyntheticBanding
    print(f"Formatting data for SyntheticBanding mode...")
    for i in range(num_phase_cycles // 2):
        # Phase cycles 0,2,4,... go to x
        x_pc_idx = i*2  # 0,2,4,...
        x[:,:,:,i*2] = data[:,:,:,x_pc_idx*2]      # Real part
        x[:,:,:,i*2+1] = data[:,:,:,x_pc_idx*2+1]  # Imaginary part
        
        # Phase cycles 1,3,5,... go to y
        y_pc_idx = i*2+1  # 1,3,5,...
        y[:,:,:,i*2] = data[:,:,:,y_pc_idx*2]      # Real part
        y[:,:,:,i*2+1] = data[:,:,:,y_pc_idx*2+1]  # Imaginary part
    print('Data format complete.')
    return x, y