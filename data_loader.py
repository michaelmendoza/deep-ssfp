
import numpy as np
import matplotlib.pyplot as plt
from rawdatarinator import readMeasDataVB15

class DataLoader:
    def __init__(self):
        data = readMeasDataVB15.readMeasDataVB15("data\\11062017_SSFP_Smoothing_DL_Phantom\meas_MID158_trufi_phi0_FID6703.dat", transformToImageSpace=True)

        key = 'imSpace'
        coil = 0
        num_avgs = data[key].shape[2]
        avg = (np.squeeze(np.sum(data[key],axis=2))/num_avgs)[:,:,coil]
        mag = np.log(np.absolute(avg))
        phase = np.angle(avg)
        f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
        ax1.imshow(mag,cmap='gray')
        ax2.imshow(phase,cmap='gray')
        ax1.set_title('log(Magnitude)')
        ax2.set_title('Phase')
        plt.show()

if __name__ == "__main__":
    DataLoader()
