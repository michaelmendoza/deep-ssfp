from __future__ import division, print_function

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio

def load():

    # Load data from matlab data
    data = sio.loadmat('./data/trainData.mat')
    imgs = np.array(data['imgs'])
    out = np.array(data['em'])

    # Crop data
    x = 192
    y = 12
    z = 0
    w = 128
    imgs = imgs[x:x+w, y:y+w, z:z+w, :]
    out = out[x:x+w, y:y+w, z:z+w]

    # Swap axies 
    imgs = np.swapaxes(imgs,0,2);
    imgs = np.swapaxes(imgs,1,2);
    out = np.swapaxes(out,0,2);
    out = np.swapaxes(out,1,2);

    # Separate complex data into real/img components
    s = imgs.shape
    _imgs = np.zeros((s[0], s[1], s[2], 8))
    for n in [0,1,2,3]:
        _imgs[:,:,:,2*n] = imgs[:,:,:,n].real
        _imgs[:,:,:,2*n+1] = imgs[:,:,:,n].imag
    _out = np.zeros((s[0], s[1], s[2], 2))
    _out[:,:,:,0] = out.real
    _out[:,:,:,1] = out.imag
    return _imgs, _out


class DataSet:
    def __init__(self):
        self.imgs, self.out = load()
        self.SIZE = self.imgs.shape[0]
        self.WIDTH = self.imgs.shape[1]
        self.HEIGHT = self.imgs.shape[2]
        self.CHANNELS_IN = 8
        self.CHANNELS_OUT = 2
        self.ratio = 0.8

        self.generate();

    def generate(self):

        # Shuffle data 
        indices = np.arange(self.SIZE)
        np.random.shuffle(indices)
        self.input = self.imgs[indices]
        self.output = self.out[indices]

        # Setup data 
        self.input = self.whiten_data(self.input)
        
        # Split data into test/training sets
        index = int(self.ratio * len(self.input)) # Split index
        self.x_train = self.input[0:index, :]
        self.y_train = self.output[0:index]
        self.x_test = self.input[index:,:]
        self.y_test = self.output[index:]

    def whiten_data(self, data): 
        """ whiten dataset - zero mean and unit standard deviation """
        data = np.reshape(data, (self.SIZE, self.WIDTH * self.HEIGHT * self.CHANNELS_IN))
        data = (np.swapaxes(data,0,1) - np.mean(data, axis=1)) / np.std(data, axis=1)
        data = np.swapaxes(data,0,1)
        data = np.reshape(data, (self.SIZE, self.WIDTH, self.HEIGHT, self.CHANNELS_IN))
        return data

    def unwhiten_img(self, img): 
        """ remove whitening for a single image """ 
        img = np.reshape(img, (self.WIDTH * self.HEIGHT * self.CHANNELS_IN))
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        img = np.reshape(img, (self.WIDTH, self.HEIGHT, self.CHANNELS_IN))
        return img

    def next_batch(self, batch_size):
        length = self.input.shape[0]
        indices = np.random.randint(0, length, batch_size);
        return [self.input[indices], self.output[indices]]

    def plot(self, x, y):
        fig, axes = plt.subplots(ncols=2)
        _first = x[:,:,0] + 1j * x[:,:,1]
        _second = y[:,:,0] + 1j * y[:,:,1]
        axes[0].imshow(np.abs(_first))
        axes[1].imshow(np.abs(_second))
        plt.show()  

    def print(self):
        print("Data Split: ", self.ratio)
        print("Train => x:", self.x_train.shape, " y:", self.y_train.shape)
        print("Test  => x:", self.x_test.shape, " y:", self.y_test.shape)

if __name__ == "__main__":
    data = DataSet()
    data.print()
    x, y = data.train_batch(1)
    data.plot(x[0], y[0])
