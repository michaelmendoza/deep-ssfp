
basepath = '/Volumes/PNY SSD/MRI Data/10282017_SSPF Smoothing DL/';
filename = 'meas_MID885_trufi_phi0_FID4833.dat';
filepath = strcat(basepath, filename);

img = readMeasDataVB15(filepath);

im = zeros(512, 256, 256);
for n = 1:8
    n
    imgn = img(:,:,:,n);
    imgn = fft3c(imgn);
    im = im + imgn;
end
im = im / 8;

disp3d(im)

