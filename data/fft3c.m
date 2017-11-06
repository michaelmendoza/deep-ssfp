function X=fft3c(x)
[nx,ny,nz,nt] = size(x);
temp_c = zeros(nx,ny,nz,nt);
for i = 1:size(x,4)
   temp_a = zeros(nx,ny,nz);
   temp_a(:,:,:) = x(:,:,:,i);
   temp_b = fftshift(fftn(fftshift(temp_a)))/sqrt(length(temp_a(:)));
   temp_c(:,:,:,i) = temp_b(:,:,:);
   clear temp_a temp_b
end
X = temp_c;