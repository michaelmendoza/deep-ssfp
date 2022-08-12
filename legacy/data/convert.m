% Converts data files to training data in a .mat

basepath = "./11062017_SSFP_Smoothing_DL_Phantom/";
filenames = ["meas_MID164_trufi_phi0_FID6709.dat", "meas_MID165_trufi_phi90_FID6710.dat", ...
            "meas_MID166_trufi_phi180_FID6711.dat", "meas_MID167_trufi_phi270_FID6712.dat"];

for f = 1:length(filenames)
    f
    filepath = strcat(basepath, filenames(f));
    im = loadImg(filepath);
    
    figure(f);
    imshow(abs(im(:,:,64)),[]);
    
    if(f == 1)
        s = size(im);
        imgs = zeros(s(1), s(2), s(3), length(filenames));
    end
    
    imgs(:,:,:,f) = im;
end

ss = imgs(:,:,:,1) .* imgs(:,:,:,1);
ss = ss + imgs(:,:,:,2) .* imgs(:,:,:,2);
ss = ss + imgs(:,:,:,3) .* imgs(:,:,:,3);
ss = ss + imgs(:,:,:,4) .* imgs(:,:,:,4);
figure(5);
imshow(abs(ss(:,:,64)), []);

s = size(imgs);
em = zeros(s(1), s(2), s(3));
for ss = 1:s(3)
    ss
    im1 = imgs(:,:,ss,1); im2 = imgs(:,:,ss,2);
    im3 = imgs(:,:,ss,3); im4 = imgs(:,:,ss,4);
    em(:,:,ss) = EllipticalModel(im1, im2, im3, im4);    
end
figure(6);
imshow(abs(em(:,:,64)), []);

save('trainData.mat', 'imgs', 'em');

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