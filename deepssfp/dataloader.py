import re
import os
import numpy as np
import mapvbvd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pydicom import dcmread
from pathlib import Path
from skimage.filters import threshold_li

from deepssfp import recon, dataloader, transforms

def load_raw_datasets(datapath, cachepath = './', cache_filename = 'phantom_dataset_cache', filter = None, indices = None, save_dataset=True):
    ''' Loads raw data from a folderpath and caches it into a npy file.
        If cached data is available, it will be loaded instead of loading from the folderpath.
        
        Parameters
        ----------
        datapath : str
            Path to folder containing raw data
        cachepath : str, optional
            Path to folder to save cached data, by default './'
        cache_filename : str, optional
            Name of cached data file, by default 'phantom_dataset_cache'
        filter : str, optional
            Filter to apply to files, by default None
        indices : list, optional
            List of indices to load, by default None
        save_dataset : bool, optional
            Whether to save the dataset to disk, by default True
        
        Returns
        -------
        np.ndarray
            Loaded dataset
    '''
    datapath = os.path.normpath(datapath)
    cachepath = os.path.normpath(cachepath)

    # Load cached data 
    if(os.path.isfile(f'{cache_filename}.npy')):
        print('Cached data found. Loading...')
        cache = np.load(f'./{cache_filename}.npy', allow_pickle=True)[0]
        print('Dataset loaded:', cache.shape)
        return cache
    
    files = os.listdir(datapath)
    files.sort()

    if filter:
        files = [file for file in files if filter in file]

    if indices:
        files = [files[i] for i in indices]

    print(f'Path: {datapath}')
    print(f'Loading files: {files}')

    dataset = [dataloader.read_rawdata(os.path.join(datapath, file)) for file in files]
    dataset = np.stack([data['data'] for data in dataset], axis=-1)
    print('Dataset loaded:', dataset.shape)
    print(f"Memory size: {dataset.nbytes / 1000000000} GB")

    # Cache rawdata into npy file
    np.save(os.path.join(cachepath, cache_filename), [dataset])
    print(f'Dataset cached as {os.path.join(cachepath, cache_filename)}.npy')

    # Create segmentation mask
    seg = create_segmentation_mask(dataset)
    dataset = { 'M': dataset, 'seg': seg }
    return dataset

data_folderpath = '../../../data/2017_DeepSSFP/11062017_SSFP_Smoothing_DL_Phantom'
cache_filename = 'deep_ssfp_phantom_dataset_cache'

def create_segmentation_mask(M):
    # Create mask of phantom
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=3))
    _ = abs(_)
    thresh = threshold_li(_)
    mask = np.abs(_) > thresh
    seg = mask * 1
    return seg

def load():
    ''' Loads and processes raw data into input (x) data tensor and output (y) data tensor.
        Loads data from cache if cached data is available. Output data is generated using
        the elliptical signal model band reduction method.

        Note: This function is deprecated and will be removed in a future release.
        Use load_datasets() instead.
    '''

    # Load cached data 
    if(os.path.isfile(f'{cache_filename}.npy')):
        print('Cached data found. Loading...')
        cache = np.load(f'./{cache_filename}.npy', allow_pickle=True)[0]
        x = cache['x']
        y = cache['y']
        return x, y

    # Load filepath from folderpath and organze in filesets
    filesets = load_filepaths()
    filesets = [filesets[0:4], filesets[4:8]]

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

def load_filepaths(datapath = data_folderpath):
    ''' Retrieves valid filepaths from a folderpath. Filepath are organized to sets of four files '''

    # Get file paths
    path = Path(datapath)
    valid_filetypes=['.dat']
    files = []
    for item in path.iterdir():
        if os.path.isfile(item) and item.suffix in valid_filetypes:
            files.append(item)
    files.sort()

    return files

def load_data_and_prepare(files):
    ''' Loads and prepares raw data from a list of filepaths '''

    # Load data from file
    M = []
    for file in files:
        data = read_rawdata(file, doChaAverage = True, doAveAverage = True)
        M.append(data['data'])
    
    # Prepare data 
    m = np.stack(M[0:4], axis=-1)

    # Crop to [64:,128,128,:]
    x0 = 32; dx = 128; y0 = 12; dy = 128; z0=32; dz = 64
    m = m[z0:z0+dz, y0:y0+dy, x0:x0+dx, :]

    return m
    
def read_rawdata(filepath: str, datatype: str = 'image', doChaAverage: bool = True, doChaSOSAverage: bool = False, doAveAverage: bool = True) -> Dict[str, Any]:
    """Read raw data file and return data dictionary."""
    twixObj = mapvbvd.mapVBVD(filepath)
    twixObj.image.squeeze = True
    data = twixObj.image['']
    sqzDims = twixObj.image.sqzDims.copy()  # Create a copy to modify

    # Move Lin to be first index
    linIndex = sqzDims.index('Lin')
    data = np.moveaxis(data, linIndex, 0)
    sqzDims.insert(0, sqzDims.pop(linIndex))

    # Process data based on options
    if doAveAverage and 'Ave' in sqzDims:
        ave_index = sqzDims.index('Ave')
        data = np.mean(data, axis=ave_index)
        sqzDims.pop(ave_index)

    if 'Par' in sqzDims:
        slice_index = sqzDims.index('Par')
        data = np.moveaxis(data, slice_index, 0)
        sqzDims.insert(0, sqzDims.pop(slice_index))

    if datatype == 'image':
        if 'Par' in sqzDims:
            data = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes=(0,1,2))))
        else:
            data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    if (doChaAverage or doChaSOSAverage) and 'Cha' in sqzDims:
        cha_index = sqzDims.index('Cha')
        if doChaAverage:
            data = np.mean(data, axis=cha_index)
        elif doChaSOSAverage:
            data = np.sqrt(np.sum(data**2, axis=cha_index))
        sqzDims.pop(cha_index)

    if 'Sli' in sqzDims:
        slice_index = sqzDims.index('Sli')
        data = np.moveaxis(data, slice_index, 0)
        sqzDims.insert(0, sqzDims.pop(slice_index))

    return {
        'data': data,
        'dims': sqzDims,
        'shape': data.shape,
        'min': float(np.nanmin(np.abs(data))),
        'max': float(np.nanmax(np.abs(data))),
        'isComplex': np.iscomplexobj(data)
    }

def read_complex_dicom_datasets(base_filepath, cache_filename = 'complex_images', filters = None, data_format='RealImag'):
    base_filepath = os.path.normpath(base_filepath)
    save_filepath = os.path.join(base_filepath, cache_filename)
    folders_list = os.listdir(base_filepath) #gives you the list of folders within the Michael_data_for_ML_model folder

    if cache_filename in folders_list:
        folders_list.remove(cache_filename)

    sorted_folders = sorted(folders_list, key=lambda x: int(x[2:].split("_")[0])) #sorting the folders based on the number after HV
    # Note that the folders are now sorted based on the HV - I have not sorted them based on the knee and repetition as that is
    # not relevant for training or testing and the files will anyway be saved as npy files with the appropriate knee and rep
    # later
    
    if filters:
        sorted_folders = list(filter(lambda x: any(f in x for f in filters), sorted_folders))  

    folder_names = ['pc_0', 'pc_90', 'pc_180', 'pc_270']

    datasets = []
    for i in range(len(sorted_folders)): #take the 1st 20 sorted folders for the training data
        combined_filepath = os.path.join(base_filepath, sorted_folders[i])
        complex_images = load_dicom_dataset(combined_filepath, folder_names)

        if (data_format == 'RealImag'):
            complex_images = transforms.from_complex_to_pairs(complex_images)
            complex_images = complex_images.astype(np.float16)
            datasets.append(complex_images)
        else:
            datasets.append(complex_images)

    # Prepare data 
    m = np.stack(datasets, axis=0)

    # Reshape to combine slices and datasets into single dimension
    new_shape = (m.shape[0] * m.shape[1],) + m.shape[2:]
    m = m.reshape(new_shape)

    # Create segmentation mask (for complex data only)
    if (data_format != 'RealImag'):
        seg = create_segmentation_mask(m).astype(np.uint8)
        dataset = { 'M': m, 'seg': seg }
    else:
        dataset = { 'M': m }

    print('Dataset loaded:', m.shape)
    print(f"Memory size: {m.nbytes / 1000000000} GB")
    return dataset

def extract_numbers(files):
    match = re.search(r'\.(\d+)\.(\d+)\.', files)
    if match:
        return int(match.group(1)), int(match.group(2))  # Extract second and third numbers
    return (0, 0)  # Default in case of no match

def mag_phase_to_complex(mag, phase):
    """
    Function that takes in magnitude and phase images and returns a complex image.

    Parameters
    ----------
    mag : array_like
        Magnitude images of shape (n, l, w), where n is the number of slices (images), 
        l is the number of rows, and w is the number of columns.

    phase : array_like
        Phase images of shape (n, l, w).

    Returns
    -------
    image : array_like
        Complex image of shape (n, l, w) calculated using the magnitude and phase data.
    """
    # Compute the real and imaginary parts using vectorized operations  
    real_part = mag * np.cos(phase)  # Shape: (n, l, w)
    imag_part = mag * np.sin(phase)  # Shape: (n, l, w)
    
    # Combine real and imaginary parts to create a complex array of shape (n, l, w)
    complex_image = real_part + 1j * imag_part  # Shape: (n, l, w)

    return complex_image #np.transpose(complex_image, (1, 2, 0))  # Shape: (l, w, n)

def load_dicom_dataset(base_filepath, folder_names):

    """
    Function that loads in the DICOM phase-cycled bSSFP data and returns the
    images in a single array.
    
    Arguments:
    ----------
    - base_filepath: string, directory containing the folders for each phase
                     cycle
    - folder_names: list of strings, folder names, with each folder 
                    containing mag/phase images acquired with diff phase 
                    cycling increments.
    
    
    Returns:
    --------
    - pc_bSSFP_imgs: numpy array, phase-cycled bSSFP images of size 
                     (80, 416, 416, 4), with each slice being of size 80 slices, 
                     416x416 length and width, and 4 phase-cycled images
    """
    
    
    # Initialize an empty array for complex images
    pc_bSSFP_imgs = np.empty((80, 416, 416, 4), dtype=np.complex64)
    idx_mg = np.arange(0, 80)  # Magnitude indices
    idx_ph = np.arange(80, 160)  # Phase indices

    for f in tqdm(range(len(folder_names))):
        filepath = os.path.join(base_filepath, folder_names[f])

        ds_path_list = os.listdir(filepath)
        ds_path_list_sorted = sorted(ds_path_list, key=lambda x: extract_numbers(x))
        ds = [dcmread(os.path.join(filepath ,ds_path)) for ds_path in ds_path_list_sorted]  # List comprehension to read DICOMs

        # Load magnitude and phase arrays
        arr_mg_list = np.array([ds[i].pixel_array for i in idx_mg])  # Collecting magnitude images
        arr_ph_list = np.array([ds[i].pixel_array for i in idx_ph])  # Collecting phase images

        # Normalize phase images
        arr_ph_normalized = np.pi * ((arr_ph_list - (np.max(arr_ph_list, axis=(1, 2), keepdims=True) / 2)) / 
                                      (np.max(arr_ph_list, axis=(1, 2), keepdims=True) / 2))

        # Calculate complex images
        pc_bSSFP_imgs[:, :, :, f] = mag_phase_to_complex(arr_mg_list, arr_ph_normalized)

    return pc_bSSFP_imgs