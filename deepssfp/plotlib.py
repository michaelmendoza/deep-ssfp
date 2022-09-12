import math
import numpy as np
import matplotlib.pyplot as plt

def plot_model_history(history):
    ''' Plot Loss History ''' 
    plt.plot(np.log10(history.history['loss']), label='loss')
    plt.plot(np.log10(history.history['val_loss']), label='val_loss')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")
    plt.show()

def plot_model_results(index, x, y, predict, kspace=False):
    ''' Plot model results. Plots input, output and prediction image in a row of images. '''
    sx = x.shape
    sy = y.shape
    sp = predict.shape

    if(kspace):
        x = np.fft.ifft2(np.fft.fftshift(x, axes=(1,2)), axes=(1,2))
        y = np.fft.ifft2(np.fft.fftshift(y, axes=(1,2)), axes=(1,2))
        predict = np.fft.ifft2(np.fft.fftshift(predict, axes=(1,2)), axes=(1,2))

    num_fig = math.floor(sx[3] / 2) + math.floor(sy[3] / 2) + math.floor(sp[3] / 2)
    fig, axs = plt.subplots(1, num_fig, sharey=True, tight_layout=True, figsize=(15, 15))
    count = 0

    for ii in range(math.floor(sx[3] / 2)):
        v = x[index,:,:,2*ii] + 1j * x[index,:,:,2*ii+1]
        axs[count].imshow(np.abs(v), cmap='gray')
        count = count + 1

    for ii in range(math.floor(sy[3] / 2)):
        v = y[index,:,:,2*ii] + 1j * y[index,:,:,2*ii+1]
        axs[count].imshow(np.abs(v), cmap='gray')
        count = count + 1

    for ii in range(math.floor(sp[3] / 2)):
        v = predict[index,:,:,2*ii] + 1j * predict[index,:,:,2*ii+1]
        axs[count].imshow(np.abs(v), cmap='gray')
        count = count + 1

    plt.show()