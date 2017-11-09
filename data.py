from __future__ import division, print_function

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
import os


def load():

    data = sio.loadmat('./data/trainData.mat')
    imgs = np.array(data['imgs'])
    out = np.array(data['em'])
    imgs = imgs[192:320,12:140,14:114,:]
    out = out[192:320,12:140,14:114]

    imgs = np.swapaxes(imgs,0,2);
    imgs = np.swapaxes(imgs,1,2);
    out = np.swapaxes(out,0,2);
    out = np.swapaxes(out,1,2);

    s = imgs.shape
    _imgs = np.zeros((s[0], s[1], s[2], 8))
    for n in [0,1,2,3]:
        _imgs[:,:,:,2*n] = imgs[:,:,:,n].real
        _imgs[:,:,:,2*n+1] = imgs[:,:,:,n].imag
    _out = np.zeros((s[0], s[1], s[2], 2))
    _out[:,:,:,0] = out.real
    _out[:,:,:,1] = out.imag
    print('r/c:', _imgs.shape, _out.shape)

    return _imgs, _out


class DataSet:
    def __init__(self):
        self.imgs, self.out = load()

    def train_batch(self, batch_size):
        length = self.imgs.shape[0]
        indices = np.random.randint(0, length, batch_size);
        return [self.imgs[indices], self.out[indices]]

    def test_batch(self):
        return self.imgs[:,:,:, 64]

    def plot(self, index):
        fig, axes = plt.subplots(ncols=2)
        _first = self.imgs[index,:,:,0] + 1j * self.imgs[index,:,:,1]
        _second = self.out[index,:,:,0] + 1j * self.out[index,:,:,1]
        axes[0].imshow(np.abs(_first))
        axes[1].imshow(np.abs(_second))
        plt.show()  

    def plot_batch(self):
        x, y = data.train_batch(1)
        fig, axes = plt.subplots(ncols=2)
        _first = x[0,:,:,0] + 1j * x[0,:,:,1]
        _second = y[0,:,:,0] + 1j * y[0,:,:,1]
        axes[0].imshow(np.abs(_first))
        axes[1].imshow(np.abs(_second))
        plt.show()  

if __name__ == "__main__":
    data = DataSet()
    data.plot(50)
    data.plot_batch()
