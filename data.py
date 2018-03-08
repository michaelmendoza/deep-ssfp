from __future__ import division, print_function

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio

class DataSet:
    def __init__(self, useSubset=False):
        if(not useSubset):
            self.imgs, self.out = self.load_format_data()
        else:
             self.imgs, self.out = self.load_format_data_subset()

        self.SIZE = self.imgs.shape[0]
        self.HEIGHT = self.imgs.shape[1]
        self.WIDTH = self.imgs.shape[2]
        self.CHANNELS_IN = self.imgs.shape[3]
        self.CHANNELS_OUT = 2
        self.ratio = 0.8

        self.generate();

    def load(self):
        # Load data from matlab data
        data = sio.loadmat('./data/trainData.mat')
        imgs = np.array(data['imgs'])
        out = np.array(data['em'])

        # Crop data
        x = 132; y = 12; z = 32; wx = 256; wy = 128; wz = 64
        imgs = imgs[x:x+wx, y:y+wy, z:z+wz, :]
        out = out[x:x+wx, y:y+wy, z:z+wz]

        # Swap axies 
        imgs = np.swapaxes(imgs,0,2);
        imgs = np.swapaxes(imgs,1,2);
        out = np.swapaxes(out,0,2);
        out = np.swapaxes(out,1,2);
        return imgs, out

    def load_format_data(self):
         # Separate complex data into real/img components

        imgs, out = self.load()

        s = imgs.shape
        _imgs = np.zeros((s[0], s[1], s[2], 8))
        for n in [0,1,2,3]:
            _imgs[:,:,:,2*n] = imgs[:,:,:,n].real
            _imgs[:,:,:,2*n+1] = imgs[:,:,:,n].imag
        _out = np.zeros((s[0], s[1], s[2], 2))
        _out[:,:,:,0] = out.real
        _out[:,:,:,1] = out.imag
        return _imgs, _out


    def load_format_data_subset(self):
        # Separate complex data into real/img components for only 2 img sets

        imgs, out = self.load()

        s = imgs.shape
        _imgs = np.zeros((s[0], s[1], s[2], 4))
        _imgs[:,:,:,0] = imgs[:,:,:,0].real
        _imgs[:,:,:,1] = imgs[:,:,:,0].imag
        _imgs[:,:,:,2] = imgs[:,:,:,2].real
        _imgs[:,:,:,3] = imgs[:,:,:,2].imag

        _out = np.zeros((s[0], s[1], s[2], 2))
        _out[:,:,:,0] = out.real
        _out[:,:,:,1] = out.imag
        return _imgs, _out

    def generate(self):

        # Shuffle data 
        indices = np.arange(self.SIZE)
        np.random.shuffle(indices)
        self.input = self.imgs[indices]
        self.output = self.out[indices]

        # Setup data 
        self.input = self.normalize_data(self.input)
        self.output = self.normalize_data(self.output)

        # Split data into test/training sets
        index = int(self.ratio * len(self.input)) # Split index
        self.x_train = self.input[0:index, :]
        self.y_train = self.output[0:index]
        self.x_test = self.input[index:,:]
        self.y_test = self.output[index:]

        #TODO: NORMALIZE WITH ENTIRE DATASET? NOT SLICES
    def normalize_data(self, data): 
        s = data.shape 
        data = np.reshape(data, (s[0], s[1] * s[2] * s[3]))
        data = np.swapaxes(data, 0, 1) / (np.max(data, axis=1) - np.min(data, axis=1)) 
        data = np.swapaxes(data, 0, 1)
        data = np.reshape(data, (s[0], s[1], s[2], s[3]))
        return data

    def whiten_data(self, data): 
        """ whiten dataset - zero mean and unit standard deviation """
        s = data.shape
        data = np.reshape(data, (s[0], s[1] * s[2] * s[3]))
        data = (np.swapaxes(data, 0, 1) - np.mean(data, axis=1)) / np.std(data, axis=1)
        data = np.swapaxes(data, 0, 1)
        data = np.reshape(data, (s[0], s[1], s[2], s[3]))
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

    def plot(self, input, output, results):
        imgs = []
        imgs.append(input[:,:,0] + 1j * input[:,:,1])
        imgs.append(input[:,:,2] + 1j * input[:,:,3])
        if(input.shape[2] > 4):
            imgs.append(input[:,:,4] + 1j * input[:,:,5])
            imgs.append(input[:,:,6] + 1j * input[:,:,7])

        imgs.append(output[:,:,0] + 1j * output[:,:,1])
        imgs.append(results[:,:,0] + 1j * results[:,:,1])

        count = int(input.shape[2] / 2) # Count of plots  
        for i in range(count):   
            plt.subplot(1, count+2, i+1)
            plt.imshow(np.abs(imgs[i]), cmap='gray')
            plt.title('Image' + str(i+1))
            plt.axis('off')

        plt.subplot(1, count+2, i+2)
        plt.imshow(np.abs(imgs[i+1]), cmap='gray')
        plt.title('Elliptical Model')
        plt.axis('off')

        plt.subplot(1, count+2, i+3)
        plt.imshow(np.abs(imgs[i+2]), cmap='gray')
        plt.title('Results')
        plt.axis('off')
        plt.show()  

    def print(self):
        print("Data Split: ", self.ratio)
        print("Train => x:", self.x_train.shape, " y:", self.y_train.shape)
        print("Test  => x:", self.x_test.shape, " y:", self.y_test.shape)

if __name__ == "__main__":
    data = DataSet()
    data.print()
    x, y = data.next_batch(1)

    print(x.shape, y.shape)
    x = x[0,:,:,0] + 1j * x[0,:,:,1]
    y = y[0,:,:,0] + 1j * y[0,:,:,1]
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(x), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(y), cmap='gray')
    plt.axis('off')

    plt.show()  
