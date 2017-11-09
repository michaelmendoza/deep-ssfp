% Converts data files to training data in a .mat

%basepath = "./10282017_SSPF Smoothing DL/";
%filenames = ["meas_MID885_trufi_phi0_FID4833.dat", "meas_MID894_trufi_phi90_FID4842.dat", ...
%            "meas_MID903_trufi_phi180_FID4851.dat", "meas_MID912_trufi_phi270_FID4860.dat"];

basepath = "./Merry_SSFP_Scans/ssfpknee/";
filenames = ["meas_MID40_SSFPdjp_TR5_4_TE2_7_PC0_FA15_FID1540.dat","meas_MID42_SSFPdjp_TR5_4_TE2_7_PC90_FA15_FID1542.dat", ...
             "meas_MID44_SSFPdjp_TR5_4_TE2_7_PC180_FA15_FID1544.dat", "meas_MID46_SSFPdjp_TR5_4_TE2_7_PC270_FA15_FID1546.dat"];     

for f = 1:length(filenames)
    f
    filepath = strcat(basepath, filenames(f));
    im = loadImg(filepath);
    
    figure(f);
    imshow(abs(im(:,:,4)),[]);
    
    if(f == 1)
        s = size(im);
        imgs = zeros(s(1), s(2), s(3), length(filenames));
    end
    
    imgs(:,:,:,f) = im;
end

s = size(imgs);
em = zeros(s(1), s(2), s(3));
for ss = 1:s(3)
    ss
    im1 = imgs(:,:,ss,1); im2 = imgs(:,:,ss,2);
    im3 = imgs(:,:,ss,3); im4 = imgs(:,:,ss,4);
    em(:,:,ss) = EllipticalModel(im1, im2, im3, im4);    
end
figure(6);
imshow(abs(em(:,:,4)), []);

save('trainDataLeg.mat', 'imgs', 'em');

function im = loadImg(filepath)
    img = readMeasDataVB15(filepath);
    s = size(img);

    im = zeros(s(1), s(2), s(3));
    for n = 1:s(4)
        imgn = img(:,:,:,n);
        imgn = fft3c(imgn);
        im = im + imgn;
    end
    im = im / s(4);
end