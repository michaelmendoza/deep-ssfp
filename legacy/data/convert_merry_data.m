% Converts data files to training data in a .mat

%basepath = "./Merry_SSFP_Scans/KneeData3T/";
%filenames = ["meas_MID31_SSFPdjp_TR10_TE5_PC0_FID16217.dat"];

basepath = "./Merry_SSFP_Scans/ssfpknee/";
filenames = ["meas_MID40_SSFPdjp_TR5_4_TE2_7_PC0_FA15_FID1540.dat"];

filepath = strcat(basepath, filenames(1));
img = readMeasDataVB15(filepath);
%img = ifftshift(ifft2(ifftshift(img)));
imgn = fft3c(img(:,:,:,1));