import numpy as np
import matplotlib.pyplot as plt
from deepssfp import dataloader, dataformatter

modes = ['BandRemoval:4', 'BandRemoval:2', 'SyntheticBanding:1_3->2_4', 'EvenOdd']

class Dataset:

    def __init__(self, mode):
        
        x, y = dataloader.load()
        x, y = dataformatter.format_and_prepare_data(x, y, mode)

        self.mode = mode
        self.x = x
        self.y = y

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

    def generate(self):

        # Shuffle data
        indices = np.arange(self.SIZE)
        np.random.shuffle(indices)
        self.input = self.x[indices]
        self.output = self.y[indices]

        # Setup data
        self.input, input_mean, input_std = self.StandardScaler(self.input)
        self.output, output_mean, output_std = self.StandardScaler(self.output)

        # Split data into test/training sets
        index = int(self.ratio * len(self.input)) # Split index
        self.x_train = self.input[0:index, :]
        self.y_train = self.output[0:index]
        self.x_test = self.input[index:,:]
        self.y_test = self.output[index:]

    def next_batch(self, batch_size):
        length = self.input.shape[0]
        indices = np.random.randint(0, length, batch_size)
        return [self.input[indices], self.output[indices]]

    def StandardScaler(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    def MinMaxScalerByImage(self, data):
        s = data.shape
        data = np.reshape(data, (s[0], s[1] * s[2] * s[3]))
        data = np.swapaxes(data, 0, 1) / (np.max(data, axis=1) - np.min(data, axis=1))
        data = np.swapaxes(data, 0, 1)
        data = np.reshape(data, (s[0], s[1], s[2], s[3]))
        return data

    def save(self):
        filename = 'deep_ssfp_phantom_dataset'
        np.save(filename, [self])

    @classmethod
    def load(cls):
        ds = np.load('./deep_ssfp_phantom_dataset.npy', allow_pickle=True)
        return ds[0]

    def plot(self):
        pass

    def histogram(self):

        n_bins = 20
        dist1 = self.input.reshape(-1)
        dist2 = self.output.reshape(-1)

        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        axs[0].hist(dist1, bins=n_bins)
        axs[1].hist(dist2, bins=n_bins)