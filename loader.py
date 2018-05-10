#data loader using rawDatarInator

from rawdatarinator.readMeasDataVB15 import readMeasDataVB15 as rmd
import scipy.io as sio

class DataLoader:
    def __init__(self):
        phi0 = rmd('../meas_MID164_trufi_phi0_FID6709.dat', transformToImageSpace=True)['kSpace']
        # phi90 = rmd('../meas_MID165_trufi_phi90_FID6710.dat', transformToImageSpace=True)['kSpace']
        # phi180 = rmd('../meas_MID166_trufi_phi180_FID6711.dat', transformToImageSpace=True)['kSpace']
        # phi270 = rmd('../meas_MID167_trufi_phi270_FID6712.dat', transformToImageSpace=True)['kSpace']
        # need to concatenate dictionaries here
        data = sio.loadmat('./data/trainData.mat')['imgs']
        # print(data.shape)
        # print(phi0.shape)

if __name__ == "__main__":
    DataLoader()

    
def load_image_data(import_name):
    	data = rmd(import_name, transformToImageSpace=True)['kSpace']
    	return data

def load_kSpace_data(import_name):
    	data = rmd(import_name)['kSpace']
    	return data

# kSpace is array from rawDatarInator and imgs is array from mat data

