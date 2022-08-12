import os
import numpy as np
import mapvbvd
from pathlib import Path

from deepssfp import recon

def load():
    filesets = load_filepaths()

    M = []
    for fileset in filesets:
        M.append(load_data_and_prepare(fileset))

    m = np.concatenate(tuple(M), axis=0)

    em = []
    for slice in range(m.shape[0]):
        em.append(recon.gs_recon(m[slice,:,:,:], pc_axis=2))
    em = np.stack(em, axis = 0)

    return m, em

def load_filepaths():
    # Get file paths
    path = Path('../../data/2017_DeepSSFP/11062017_SSFP_Smoothing_DL_Phantom')
    valid_filetypes=['.dat']
    files = []
    for item in path.iterdir():
        if os.path.isfile(item) and item.suffix in valid_filetypes:
            files.append(item)
    files.sort()

    filesets = [files[0:4], files[4:8]]
    return filesets

def load_data_and_prepare(files):

    # Load data from file
    M = []
    for file in files:
        data = read_rawdata(file, is3D=True, doChaAverage = True, doAveAverage = True);
        M.append(data['data'])
    
    # Prepare data 
    m = np.stack(M[0:4], axis=-1)
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
