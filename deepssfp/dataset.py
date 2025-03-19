import os.path
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from deepssfp import dataloader, dataformatter, recon

# 'SyntheticBanding:1_3->2_4': 'SyntheticBanding'
modes = ['BandRemoval:4', 'BandRemoval:2', 'SyntheticBanding', 'SuperFOV']

class DataMode(Enum):
    BandRemoval4 = 'BandRemoval:4'
    BandRemoval2 = 'BandRemoval:2'
    SyntheticBanding = 'SyntheticBanding'
    SuperFOV = 'SuperFOV'

class Dataset:

    def __init__(self, mode, input_data=None, output_data=None):
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
        self.ratio = 0.8

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

            if output_data is None and self.mode != 'SyntheticBanding':
                y = [] 
                for slice in range(x.shape[0]):
                    y.append(recon.gs_recon(x[slice,:,:,:], pc_axis=2))
                y = np.stack(y, axis = 0)
            else:
                y = output_data

        else:
            # Load default data if no custom data provided
            x, y = dataloader.load()

        #print(f'X: {x.shape}')
        x, y = dataformatter.format_and_prepare_data(x, y, self.mode)
        #print(f'X: {x.shape} Y: {y.shape}')

        return x, y

    def generate(self):
        ''' Generates training/test dataset '''

        # Shuffle data
        indices = np.arange(self.SIZE)
        np.random.shuffle(indices)
        self.input = self.x[indices]
        self.output = self.y[indices]

        # Setup data
        self.inputScaler = StandardScaler(self.input)
        self.outputScaler = StandardScaler(self.output)
        self.input = self.inputScaler.transform(self.input)
        self.output = self.outputScaler.transform(self.output)

        # Split data into test/training sets
        index = int(self.ratio * len(self.input)) # Split index
        self.x_train = self.input[0:index, :]
        self.y_train = self.output[0:index]
        self.x_test = self.input[index:,:]
        self.y_test = self.output[index:]

    def next_batch(self, batch_size):
        ''' Retrieves next samples of training data '''
        length = self.input.shape[0]
        indices = np.random.randint(0, length, batch_size)
        return [self.input[indices], self.output[indices]]

    def plot(self):
        pass

    def histogram(self):
        ''' Plots histogram plots of input/output data '''
        n_bins = 20
        dist1 = self.input.reshape(-1)
        dist2 = self.output.reshape(-1)

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
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def transform(self, data):
        ''' Transforms data using mean/std statistics '''
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        ''' Transforms data using mean/std statistics '''
        return data * self.std + self.mean