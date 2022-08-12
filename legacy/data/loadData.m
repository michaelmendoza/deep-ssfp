function imgs = loadData(basepath, dirname)

% Get filepaths
filepath = strcat(basepath, dirname{1});
files = dir(fullfile(filepath,'*.dat'));
fileCount = length(files);
filenames = {};
for k=1:fileCount
    filenames{k} = fullfile(filepath,files(k).name);
end

% Load dat files
imgs = {};
for n=1:fileCount
    n
    img = readMeasDataVB15(filenames{n});       % Read datafile
    imgs{n} = ifftshift(ifft2(ifftshift(img))); % Transform from kspace to image
    size(img)
end

