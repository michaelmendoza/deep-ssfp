from __future__ import division, print_function

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
from mr_utils.recon.ssfp import gs_recon
from mr_utils import view
# import glob
# from mr_utils.load_data import load_raw
# from mr_utils import view

class DataSet:

    learningModes = ['BandRemoval2', 'BandRemoval4', 'SyntheticBanding', 'EvenOdd']

    def __init__(self, learningMode):

        learningFn = [self.load_format_data, self.load_format_data_subset, self.load_format_synthetic_banding, self.load_format_even_odd_input_kspace_output_kspace]
        self.imgs, self.out = dict(zip(DataSet.learningModes, learningFn))[learningMode]()

        self.SIZE = self.imgs.shape[0]
        self.HEIGHT = self.imgs.shape[1]
        self.WIDTH = self.imgs.shape[2]
        self.CHANNELS_IN = self.imgs.shape[3]
        self.CHANNELS_OUT = self.out.shape[3]
        self.ratio = 0.8

        self.generate();

    ''' 
    Load raw loader -
    Input: [Depth/Slice, Width, Height, N]

    Depth/Slice -> Number of data slices
    N -> Number of phase cycled images
    '''
    # def loadRawData(self,directory):

    #     # In the directory provided, find all .dat files
    #     files = sorted(glob.glob('%s/*.dat' % directory))

    #     TEs = [ 3,6,12,24,4,5 ]
    #     pcs = [ 0,45,90,135,180,225,270,315 ]
    #     pc0 = np.zeros((512,352,8,len(TEs)),dtype='complex')

    #     groups = [ files[ii:ii+len(pcs)] for ii in range(0,len(files),len(pcs))]
    #     print(groups)
    #     # for file in files:
    #     #     # view(file,fft=True,avg_axis=3)
    #     #
    #     #     # Load, average, then save as npy
    #     #     data = load_raw(file)
    #     #     data = np.mean(data,axis=3)
    #     #     np.save(file,data)
    #     #     print('Saved %s' % file)

    def load(self):
        # Load data from matlab data
        #data = sio.loadmat('./data/trainData.mat')
        data = sio.loadmat('./data/trainData_20180509.mat')
        imgs = np.array(data['imgs'])
        out = np.array(data['em'])

        # Crop data
        #x = 132; y = 12; z = 32; wx = 256; wy = 128; wz = 64
        x = 128; y = 0; z = 32; wx = 256; wy = 256; wz = 64

        imgs = imgs[x:x+wx, y:y+wy, z:z+wz, :]
        out = out[x:x+wx, y:y+wy, z:z+wz]

        # Swap axies
        imgs = np.swapaxes(imgs,0,2);
        imgs = np.swapaxes(imgs,1,2);
        out = np.swapaxes(out,0,2);
        out = np.swapaxes(out,1,2);
        return imgs, out

    def load_brain(self):
        # Load npy data for each phase cycle
        pc0 = np.load('./data/brain/meas_MID23_TRUFI_STW_TE2_5_FID33594.dat_avg_coil_combined.npy')
        pc90 = np.load('./data/brain/meas_MID24_TRUFI_STW_TE2_5_dphi_90_FID33595.dat_avg_coil_combined.npy')
        pc180 = np.load('./data/brain/meas_MID25_TRUFI_STW_TE2_5_dphi_180_FID33596.dat_avg_coil_combined.npy')
        pc270 = np.load('./data/brain/meas_MID26_TRUFI_STW_TE2_5_dphi_270_FID33597.dat_avg_coil_combined.npy')

        # Shape should be (z,x,y,pc)
        imgs = np.stack((pc0,pc90,pc180,pc270)).transpose((3,1,2,0))
        # view(imgs.T)
        # out = np.stack((pc90,pc270)).transpose((3,1,2,0))

        # Output should be solution to ESM for each slice
        out = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]),dtype='complex')
        for kk in range(imgs.shape[0]):
            out[kk,...] = gs_recon(pc0[...,kk],pc90[...,kk],pc180[...,kk],pc270[...,kk])
        # view(out)

        return imgs,out


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

    def load_format_synthetic_banding(self):
        # Separate complex data into real/img components for only 2 img sets

        #imgs, out = self.load_brain() 
        imgs, out = self.load()

        s = imgs.shape
        _imgs = np.zeros((s[0], s[1], s[2], 4))
        _imgs[:,:,:,0] = imgs[:,:,:,0].real
        _imgs[:,:,:,1] = imgs[:,:,:,0].imag
        _imgs[:,:,:,2] = imgs[:,:,:,2].real
        _imgs[:,:,:,3] = imgs[:,:,:,2].imag

        _out = np.zeros((s[0], s[1], s[2], 4))
        _out[:,:,:,0] = imgs[:,:,:,1].real
        _out[:,:,:,1] = imgs[:,:,:,1].imag
        _out[:,:,:,2] = imgs[:,:,:,3].real
        _out[:,:,:,3] = imgs[:,:,:,3].imag
        return _imgs, _out

    def load_format_even_odd_input_kspace_output_kspace_data_consistency(self):
        # Input images are alternating even/odd lines of k-space taken from 2 phase cycled acquisitions (k-space)
        # Output is the elliptical singal model in k-space and fully sampled input images

        imgs, out = self.load()
        imgs = np.fft.fftshift(np.fft.fft2(imgs,axes=(1,2)),axes=(1,2))
        out = np.fft.fftshift(np.fft.fft2(out,axes=(1,2)),axes=(1,2))

        s = imgs.shape
        _imgs = np.zeros((s[0], s[1], s[2], 4))
        _imgs[:,:,:,0] = imgs[:,:,:,0].real
        _imgs[:,:,:,1] = imgs[:,:,:,0].imag
        _imgs[:,:,:,2] = imgs[:,:,:,2].real
        _imgs[:,:,:,3] = imgs[:,:,:,2].imag

        _out = np.zeros((s[0], s[1], s[2], 6))
        _out[:,:,:,0] = out.real
        _out[:,:,:,1] = out.imag

        # Enforce data consistency with additional "decoder"
        _out[:,:,:,2] = _imgs[:,:,:,0]
        _out[:,:,:,3] = _imgs[:,:,:,1]
        _out[:,:,:,4] = _imgs[:,:,:,2]
        _out[:,:,:,5] = _imgs[:,:,:,3]
        
        _ds = np.zeros((s[0], s[1], s[2], 2))
        _ds[:,::2,:,0] = _imgs[:,::2,:,0]
        _ds[:,1::2,:,0] = _imgs[:,1::2,:,2]
        _ds[:,::2,:,1] = _imgs[:,::2,:,1]
        _ds[:,1::2,:,1] = _imgs[:,1::2,:,3]


        return _ds, _out

    def load_format_even_odd_input_kspace_output_kspace(self):
        
        # Input images are alternating even/odd lines of k-space taken from 2 phase cycled acquisitions (k-space)
        # Output is the elliptical singal model in k-space

        imgs, out = self.load()
        imgs = np.fft.fftshift(np.fft.fft2(imgs,axes=(1,2)),axes=(1,2))
        out = np.fft.fftshift(np.fft.fft2(out,axes=(1,2)),axes=(1,2))

        s = imgs.shape
        _imgs = np.zeros((s[0], s[1], s[2], 4))
        _imgs[:,:,:,0] = imgs[:,:,:,0].real
        _imgs[:,:,:,1] = imgs[:,:,:,0].imag
        _imgs[:,:,:,2] = imgs[:,:,:,2].real
        _imgs[:,:,:,3] = imgs[:,:,:,2].imag

        _out = np.zeros((s[0], s[1], s[2], 2))
        _out[:,:,:,0] = out.real
        _out[:,:,:,1] = out.imag
        
        _ds = np.zeros((s[0], s[1], s[2], 2))
        _ds[:,::2,:,0] = _imgs[:,::2,:,0]
        _ds[:,1::2,:,0] = _imgs[:,1::2,:,2]
        _ds[:,::2,:,1] = _imgs[:,::2,:,1]
        _ds[:,1::2,:,1] = _imgs[:,1::2,:,3]

        # tmp = np.zeros(_ds.shape,dtype='complex')
        # o = 0
        # tmp[0,o::2,:,0] = _ds[0,o::2,:,0] + 1j*_ds[0,o::2,:,1]
        # tmp = np.fft.fft2(tmp,axes=(1,2))
        # plt.subplot(2,1,1)
        # plt.imshow(np.abs(np.fft.fft2(imgs,axes=(1,2))[0,:,:,0]))
        # plt.subplot(2,1,2)
        # plt.imshow(np.abs(tmp[0,:,:,0]))
        # plt.show()

        return _ds, _out

    def load_format_even_odd_separate_kspace(self):
        # Input images are even/odd lines of k-space taken from 2 phase cycled acquisitions into 2 separate images (k-space)
        # Output is the elliptical singal model in k-space
        pass 

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

    def plot_synthetic_banding(self, input, output, results):
        imgs = []
        imgs.append(input[:,:,0] + 1j * input[:,:,1])
        imgs.append(input[:,:,2] + 1j * input[:,:,3])

        imgs.append(output[:,:,0] + 1j * output[:,:,1])
        imgs.append(output[:,:,2] + 1j * output[:,:,3])

        # Run the elliptical model on input dataset as ground truth
        from mr_utils.recon.ssfp import gs_recon
        pc0 = input[:,:,0] + 1j*input[:,:,1]
        pc90 = output[:,:,0] + 1j*output[:,:,1]
        pc180 = input[:,:,2] + 1j*input[:,:,3]
        pc270 = output[:,:,2] + 1j*output[:,:,3]
        recon_truth = gs_recon(pc0,pc90,pc180,pc270)
        imgs.append(recon_truth)


        imgs.append(results[:,:,0] + 1j * results[:,:,1])
        imgs.append(results[:,:,2] + 1j * results[:,:,3])


        # Run the elliptical model to verify the reconstruction
        pc0 = input[:,:,0] + 1j*input[:,:,1]
        pc90 = results[:,:,0] + 1j*results[:,:,1]
        pc180 = input[:,:,2] + 1j*input[:,:,3]
        pc270 = results[:,:,2] + 1j*results[:,:,3]
        recon_using_prediction = gs_recon(pc0,pc90,pc180,pc270)
        imgs.append(recon_using_prediction)

        count = len(imgs) # Count of plots
        for i in range(count):
            plt.subplot(1, count, i+1)
            plt.imshow(np.abs(imgs[i]), cmap='gray')
            plt.axis('off')
        plt.show()

    def plot_evenodd(self, input, output, results):
        imgs = []

        inn = input[:,:,0] + 1j * input[:,:,1]
        imgs.append(np.abs(np.log(inn)))
        imgs.append(np.fft.ifft2(inn))

        out = output[:,:,0] + 1j * output[:,:,1]
        imgs.append(np.abs(np.log(out)))
        imgs.append(np.fft.ifft2(out))

        res = results[:,:,0] + 1j * results[:,:,1]
        imgs.append(np.abs(np.log(res)))
        imgs.append(np.fft.ifft2(res))

        count = len(imgs) # Count of plots
        for i in range(count):
            plt.subplot(1, count, i+1)
            plt.imshow(np.abs(imgs[i]), cmap='gray')
            plt.axis('off')
        plt.show()

    def print(self):
        print("Data Split: ", self.ratio)
        print("Train => x:", self.x_train.shape, " y:", self.y_train.shape)
        print("Test  => x:", self.x_test.shape, " y:", self.y_test.shape)

if __name__ == "__main__":

    # Import Dataset
    modes = DataSet.learningModes;
    data = DataSet(modes[2])
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
