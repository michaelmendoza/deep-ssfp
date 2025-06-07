import os.path
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from deepssfp import dataloader, dataformatter, recon

# 'SyntheticBanding:1_3->2_4': 'SyntheticBanding'
modes = ['BandRemoval:4', 'BandRemoval:2', 'SyntheticBanding', 'SuperFOV', 'SuperFOVi']

class DataMode(Enum):
    BandRemoval4 = 'BandRemoval:4'
    BandRemoval2 = 'BandRemoval:2'
    SyntheticBanding = 'SyntheticBanding'
    SyntheticBanding2 = 'SyntheticBanding:2'
    SyntheticBandingSuperFOVi = 'SyntheticBanding:SuperFOVi' 
    SuperFOV = 'SuperFOV'
    SuperFOVi = 'SuperFOVi'

class Dataset:

    def __init__(self, mode, input_data=None, output_data=None, ratio = 0.8, stats_faction : float = 1.0, scaler = None, verbose=True):
        """Initialize Dataset with either provided data or loaded data.
            
            Parameters
            ----------
            mode : str
                One of the supported modes from modes list
            input_data : ndarray, optional
                Input data of shape [slices, height, width, phase_cycles]
            output_data : ndarray, optional
                Output/target data of shape [slices, height, width, channels]
        """

        self.mode = mode
        self.x, self.y = self.load_data(input_data, output_data)

        self.SIZE = self.x.shape[0]
        self.HEIGHT = self.x.shape[1]
        self.WIDTH = self.x.shape[2]
        self.CHANNELS_IN = self.x.shape[3]
        self.CHANNELS_OUT = self.y.shape[3]
        self.ratio = ratio
        self.scaler = scaler
        self.stats_faction = stats_faction
        self.dtype = self.x.dtype

        if verbose:
            print(f"Dataset: mode:{self.mode}, size:{self.SIZE} height:{self.HEIGHT} width:{self.WIDTH} cin:{self.CHANNELS_IN} cout:{self.CHANNELS_OUT} ratio:{self.ratio} dtype: {self.x.dtype}, {self.y.dtype}")

        self.generate()

    def __str__(self):
        return f'Dataset: mode:{self.mode}, size:{self.SIZE} height:{self.HEIGHT} width:{self.WIDTH} cin:{self.CHANNELS_IN} cout:{self.CHANNELS_OUT} ratio:{self.ratio}'

    def __repr__(self) -> str:
        return f'dataset.Dataset({self.mode})'

    def load_data(self, input_data=None, output_data=None):
        ''' Load, format and prepare data for dataset '''

        # Check if custom data is provided
        if input_data is not None:
            x = input_data

            if output_data is None and 'SyntheticBanding' not in self.mode:
                y = [] 
                for slice in range(x.shape[0]):
                    y.append(recon.gs_recon(x[slice,:,:,:], pc_axis=2))
                y = np.stack(y, axis = 0)
            else:
                y = output_data

        else:
            # Load default data if no custom data provided
            x, y = dataloader.load()

        if input_data is None or output_data is None:
            # Format data
            #print('Formatting data...')
            x, y = dataformatter.format_and_prepare_data(x, y, self.mode)
        else:
            #print('Info: Data all ready formated.')
            x, y = (input_data, output_data)

        return x, y

    def generate(self):
        ''' Generates training/test dataset '''
        
        if self.scaler is not None:
            #print("Using provided scaler")
            self.inputScaler = self.scaler
            self.outputScaler = self.scaler
        else:
        # Setup data - Use same scaler for SyntheticBanding mode
            #print('Generating scaler... for mode:', self.mode)
            if self.mode == 'SyntheticBanding':
                #print('Using same scaler for SyntheticBanding mode')
                combined_data = np.concatenate((self.x, self.y), axis=0).astype(self.dtype)
                self.inputScaler = StandardScaler(combined_data, self.stats_faction)
                self.outputScaler = StandardScaler(combined_data, self.stats_faction)
                del combined_data
            else:
                self.inputScaler = StandardScaler(self.x, self.stats_faction)
                self.outputScaler = StandardScaler(self.y, self.stats_faction)
            

        # Setup data
        self.x = self.inputScaler.transform(self.x).astype(self.dtype)
        self.y = self.outputScaler.transform(self.y).astype(self.dtype)

        # Split data into test/training sets
        index = int(self.ratio * len(self.x)) # Split index
        self.x_train = self.x[0:index, :]
        self.y_train = self.y[0:index]
        self.x_test = self.x[index:,:]
        self.y_test = self.y[index:]

        # Clear memory
        del self.x
        del self.y

    def next_batch(self, batch_size):
        ''' Retrieves next samples of training data '''
        length = self.x.shape[0]
        indices = np.random.randint(0, length, batch_size)
        return [self.x[indices], self.y[indices]]

    def plot(self):
        pass

    def histogram(self):
        ''' Plots histogram plots of input/output data '''
        n_bins = 20
        dist1 = self.x.reshape(-1)
        dist2 = self.y.reshape(-1)

        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        axs[0].hist(dist1, bins=n_bins)
        axs[1].hist(dist2, bins=n_bins)

    def transform(self, data, type='input'):
        ''' Transforms data using datset settings. It formats and scales the data. 
            For input: It will format data, then scale. For output: It will inverse scale, then format to complex data '''

        if type == 'input':
            x, y = self.load_data(data)
            return self.inputScaler.transform(x)
        elif type == 'output':
            y = self.inputScaler.inverse_transform(data)
            y = dataformatter.real_imag_to_complex(y)
            return y
        
class StandardScaler:
    def __init__(self, data: np.ndarray, stats_faction : float = 1.0, mean = None, std = None):
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
            return
        
        if stats_faction == 1:
            self.mean = np.mean(data, dtype=np.float64)
            self.std = np.std(data, dtype=np.float64)
        else:
            indices = np.random.choice(range(data.shape[0]), size=data.shape[0] * stats_faction, replace=False)
            self.mean = np.mean(data[indices], dtype=np.float64)
            self.std = np.std(data[indices], dtype=np.float64)

    
    def transform(self, data) -> np.ndarray:
        ''' Transforms data using mean/std statistics '''
        return (data - self.mean) / self.std

    def inverse_transform(self, data) -> np.ndarray:
        ''' Transforms data using mean/std statistics '''
        return data * self.std + self.mean    