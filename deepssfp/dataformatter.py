import numpy as np

modes = ['BandRemoval:4', 'BandRemoval:2', 'SyntheticBanding:1_3->2_4', 'SuperFOV']

def format_and_prepare_data(x, y, mode):
    ''' Formats and prepares data for deepssfp experiments:
        1) BandRemoval:4 - Transforms complex data into real/img components
        2) BandRemoval:2 - Takes a subset of x data and transforms complex data into real/img components
        3) SyntheticBanding:1_3->2_4 - Takes alternating subsets of data and transforms complex data into real/img components
        4) SuperFOV - Takes alternating even/odd lines of k-space taken from 2 phase cycled acquisitions (k-space), output vector also in k-space. 
    '''
    if mode == 'BandRemoval:4':
        pass
    elif mode == 'BandRemoval:2':
        x = x[:,:,:,::2]
    elif mode == 'SyntheticBanding:1_3->2_4':
        y = x[:,:,:,1::2]
        x = x[:,:,:,::2]
    elif mode == 'SuperFOV':
        x = x[:,:,:,::2]
        x = np.fft.fftshift(np.fft.fft2(x, axes=(1,2)), axes=(1,2))
        y = np.fft.fftshift(np.fft.fft2(y, axes=(1,2)), axes=(1,2))
    else:
        raise Exception('Invalid data mode')

    x = complex_to_real_img(x)
    y = complex_to_real_img(y)

    if mode == 'SuperFOV':
        sx = x.shape
        _x = np.zeros((sx[0], sx[1], sx[2], 2))
        _x[:,::2,:,0] = x[:,::2,:,0]
        _x[:,1::2,:,0] = x[:,1::2,:,2]
        _x[:,::2,:,1] = x[:,::2,:,1]
        _x[:,1::2,:,1] = x[:,1::2,:,3]
        x = _x

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
