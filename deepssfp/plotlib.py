import imageio; 
from IPython.display import Video; 
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(data, slice = None):
    ''' Plots a slice of dataset of form: [Slice, Height, Width, Channel] '''

    slice = 0 if slice is None else slice
    npcs = data.shape[3]
    nx, ny = 2, int(npcs / 2)
    plt.figure()
    for ii in range(nx * ny):
        _data = np.abs(data[slice, :, :, ii])
        plt.subplot(nx, ny, ii+1)
        plt.imshow(_data, cmap='gray')
        plt.title('%d deg PC' % (ii * (360/npcs)))
    plt.show()

def combine_channels(data):
    ''' Combines width and channel data i.e. [B, H, W, C] => [B, H, W * C] '''

    _data = np.transpose(data, (3,1,2,0))
    _data = np.transpose(_data, (0,2,1,3))
    _data = np.concatenate(_data, axis=0)
    return np.transpose(_data, (2,1,0))

def show_dataset_channel(data, channel = 4):
    ''' Shows a channel of dataset in a video  (time is the slice dimension) '''

    _data = (np.abs(data[:,:,:,channel]) * 255).astype(np.uint8)
    imageio.mimwrite('_.mp4', _data, fps=30); 
    return Video('_.mp4', width=480, height=360) 

def show_dataset(data, min = None, max = None):
    ''' Shows dataset in a video (time is the slice dimension) '''

    if len(data.shape) == 4:
        _data = combine_channels(data)
    elif len(data.shape) == 3:
        _data = data
    else:
        return None

    resolution = 255
    _data = np.abs(_data)
    if min == None:
        min = float(np.nanmin(_data))
    if max == None:
        max = float(np.nanmax(_data))
        
    _data = (_data - min) * resolution / (max - min)
    _data = (_data).astype(np.uint8)
    imageio.mimwrite('_.mp4', _data, fps=30); 
    return Video('_.mp4', width=_data.shape[2], height=_data.shape[1])