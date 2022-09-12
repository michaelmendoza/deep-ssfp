import os
import numpy as np
import mapvbvd
from pathlib import Path

from deepssfp import recon

data_folderpath = '../../data/2017_DeepSSFP/11062017_SSFP_Smoothing_DL_Phantom';
cache_filename = 'deep_ssfp_phantom_dataset_cache'

def load():
    ''' Loads and processes raw data into input (x) data tensor and output (y) data tensor.
        Loads data from cache if cached data is available. Output data is generated using
        the elliptical signal model band reduction method. 
    '''

    # Load cached data 
    if(os.path.isfile(f'{cache_filename}.npy')):
        cache = np.load(f'./{cache_filename}.npy', allow_pickle=True)[0]
        x = cache['x']
        y = cache['y']
        return x, y

    # Load filepath from folderpath and organze in filesets
    filesets = load_filepaths()

    # Load and process rawdata into rawdata tensor
    M = []
    for fileset in filesets:
        M.append(load_data_and_prepare(fileset))
    x = np.concatenate(tuple(M), axis=0)

    # Generate truth dataset from data using elliptical signal model band reduction method
    y = [] 
    for slice in range(x.shape[0]):
        y.append(recon.gs_recon(x[slice,:,:,:], pc_axis=2))
    y = np.stack(y, axis = 0)
    
    # Cache rawdata into npy file
    cache = { 'x':x, 'y':y }
    np.save(cache_filename, [cache])

    # Return input / output dataset 
    return x, y

def load_filepaths():
    ''' Retrieves valid filepaths from a folderpath. Filepath are organized to sets of four files '''

    # Get file paths
    path = Path(data_folderpath)
    valid_filetypes=['.dat']
    files = []
    for item in path.iterdir():
        if os.path.isfile(item) and item.suffix in valid_filetypes:
            files.append(item)
    files.sort()

    filesets = [files[0:4], files[4:8]]
    return filesets

def load_data_and_prepare(files):
    ''' Loads and prepares raw data from a list of filepaths '''

    # Load data from file
    M = []
    for file in files:
        data = read_rawdata(file, is3D=True, doChaAverage = True, doAveAverage = True)
        M.append(data['data'])
    
    # Prepare data 
    m = np.stack(M[0:4], axis=-1)

    # Crop to [64:,128,128,:]
    x0 = 32; dx = 128; y0 = 12; dy = 128; z0=32; dz = 64
    m = m[z0:z0+dz, y0:y0+dy, x0:x0+dx, :]

    return m
    
def read_rawdata(filepath, datatype='image', is3D=False, doChaAverage = True, doChaSOSAverage = False, doAveAverage = True):
    ''' Reads rawdata files and returns NodeDataset '''

    twixObj = mapvbvd.mapVBVD(filepath)
    sqzDims = twixObj.image.sqzDims    
    twixObj.image.squeeze = True

    data = twixObj.image['']
    # Move Lin be first index
    linIndex = sqzDims.index('Lin')
    data = np.moveaxis(data, linIndex, 0)
    sqzDims.insert(0, sqzDims.pop(linIndex))

    if doAveAverage and 'Ave' in sqzDims:
        chaIndex = sqzDims.index('Ave')
        data = np.mean(data, axis=chaIndex)
        sqzDims.pop(chaIndex)
                
    if is3D:
        if 'Par' in sqzDims:
            sliceIndex = sqzDims.index('Par')
            data = np.moveaxis(data, sliceIndex, 0)
            sqzDims.insert(0, sqzDims.pop(sliceIndex))

    if datatype == 'image':
        if is3D:
            data = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes=(0,1,2))))
        else:
            data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    else: # datatype is kspace
        pass

    if (doChaAverage or doChaSOSAverage) and 'Cha' in sqzDims:
        chaIndex = sqzDims.index('Cha')

        if doChaAverage:
            data = np.mean(data, axis=chaIndex)
        elif doChaSOSAverage:
            data = np.sqrt(np.sum(data**2, axis=(chaIndex)))

        sqzDims.pop(chaIndex)

    if 'Sli' in sqzDims:
        sliceIndex = sqzDims.index('Sli')
        data = np.moveaxis(data, sliceIndex, 0)
        sqzDims.insert(0, sqzDims.pop(sliceIndex))

    return { 'data':data, 'dims':sqzDims, 'shape':data.shape } 
