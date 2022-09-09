from typing import Any
import numpy as np

def phantom_generator(size = 256, type = 'shepp_logan'):
    ''' Generates a phantom with a given shape for a number of coils.
    Args:
        size: size of image.
        coil: number of coils (Dtype): data type.
        type: type of phantom
    Returns:
        array.
    '''
    size = int(size)

    if type == 'shepp_logan':
        rawdata = shepp_logan_phantom([size,size]) 
    elif type == 'circle':
        rawdata = circle_phantom([size,size])
    elif type == 'circles':
        rawdata = circle_array_phantom([size,size])
    elif type == 'blocks':
        rawdata = block_phantom()
    else:
        raise ValueError('Incorrect phantom type')

    return rawdata
    
def block_phantom(shape = 256, blocks = 8, padding = 8):
    width = 64
    height = 64

    s = (width - 2 * padding, height - 2 * padding)
    patches = [[],[]]
    for i in range(blocks+1):
        if i > 0:
            patch = np.ones(s) * i
            patch = np.pad(patch, (padding, padding))
            patches[int((i - 1) / 4)].append(patch)
        
    mask : Any = np.block(patches)
    mask = mask.astype(int)
    mask = np.pad(mask, ((64,64),(0,0)))

    return mask

def shepp_logan_phantom(fov):
    ''' Generates a Shepp Logan phantom with a given shape.
    Args:
        fov: size
    Returns:
        array.
    ''' 

    sl_amps = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    sl_scales = [[.6900, .920, .810],  # white big
                [.6624, .874, .780],  # gray big
                [.1100, .310, .220],  # right black
                [.1600, .410, .280],  # left black
                [.2100, .250, .410],  # gray center blob
                [.0460, .046, .050],
                [.0460, .046, .050],
                [.0460, .046, .050],  # left small dot
                [.0230, .023, .020],  # mid small dot
                [.0230, .023, .020]]

    sl_offsets = [[0., 0., 0],
                [0., -.0184, 0],
                [.22, 0., 0],
                [-.22, 0., 0],
                [0., .35, -.15],
                [0., .1, .25],
                [0., -.1, .25],
                [-.08, -.605, 0],
                [0., -.606, 0],
                [.06, -.605, 0]]

    sl_angles = [[0, 0, 0],
                [0, 0, 0],
                [-18, 0, 10],
                [18, 0, 10],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

    image = phantom(fov, sl_amps, sl_scales, sl_offsets, sl_angles, dtype = float)
    return image

def circle_array_phantom(fov):
    ''' Generates an array of circle phantoms with a given shape and dtype.
    Args:
        fov: size
    Returns:
        array.
    ''' 
    amps = [ 1, 2, 3, 4 ]
    scales = [[ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ]]
    offsets = [[ 0.5, 0.5, 0 ], [ -0.5, 0.5, 0 ], [ 0.5, -0.5, 0 ], [ -0.5, -0.5, 0 ]]
    angles = [[ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ]]

    image = phantom(fov, amps, scales, offsets, angles, dtype = float)
    return image

def circle_phantom(fov):
    ''' Generates a circle phantom with a given size.
    Args:
        fov: size
    Returns:
        array.
    ''' 
    amps = [1]
    scales = [[ 0.75, 0.75, 0.75 ]]
    offsets = [[ 0, 0, 0 ]]
    angles = [[ 0, 0, 0 ]]

    image = phantom(fov, amps, scales, offsets, angles, dtype = float)
    return image

def phantom(shape, amps, scales, offsets, angles, dtype):
    '''
    Generate a cube of given shape using a list of ellipsoid
    parameters.
    '''
    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])

    elif len(shape) == 3:
        ndim = 3

    else:
        raise ValueError('Incorrect dimension')

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[-(shape[-3] // 2):((shape[-3] + 1) // 2),
                       -(shape[-2] // 2):((shape[-2] + 1) // 2),
                       -(shape[-1] // 2):((shape[-1] + 1) // 2)]

    coords = np.stack((x.ravel() / shape[-1] * 2,
                       y.ravel() / shape[-2] * 2,
                       z.ravel() / shape[-3] * 2))

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):

        ellipsoid(amp, scale, offset, angle, coords, out)

    if ndim == 2:

        return out[0, :, :]

    else:

        return out


def ellipsoid(amp, scale, offset, angle, coords, out):
    '''
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new
    one.
    '''
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / \
        np.reshape(scale, (3, 1))

    r2 = np.sum(coords ** 2, axis=0).reshape(out.shape)

    out[r2 <= 1] += amp


def rotation_matrix(angle):
    ''' Generate rotation matrix '''
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
              -spsi * sphi + ctheta * cphi * cpsi,
              cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.array(alpha)
