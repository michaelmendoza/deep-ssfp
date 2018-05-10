import math
import numpy as np
import scipy.io
import scipy.misc
from loader import load_image_data, load_kSpace_data

# This function is a Python port of the MATLAB function implemented by the
# Bangerter Research group of the ellpitical signal model in the paper by
# Xiang, Qing-San and Hoff, Michael N. "Banding Artifact Removal for bSSFP
# Imaging with an Elliptical Signal Model", 2014.
# The function returns a complex image.

# I1 and I3 form one pair of 180 degree offset images, and I2 and I4 form
# the other pair.
def elliptical_model(I1, I2, I3, I4):
    # Iterate through each pixel and calculate M directly; then compare it to
    # the maximum magnitude of all four input images.  If it is greater,
    # replace it with the complex sum value.

    maximum = np.fmax(np.absolute(I1),np.absolute(I2))
    maximum = np.fmax(maximum,np.absolute(I3))
    maximum = np.fmax(maximum,np.absolute(I4))
    M = np.zeros(I1.shape,dtype=complex)
    replacements = 0
    
    CS = (I1 + I2 + I3 + I4)/4;

    for k in range(0,I1.shape[0]-1):
        for n in range(0,I1.shape[1]-1):
            M[k,n] = ((np.real(I1[k,n])*np.imag(I3[k,n]) - np.real(I3[k,n])*np.imag(I1[k,n]))* \
            (I2[k,n] - I4[k,n]) - (np.real(I2[k,n])*np.imag(I4[k,n]) - np.real(I4[k,n])*np.imag(I2[k,n]))* \
            (I1[k,n] - I3[k,n])) / ((np.real(I1[k,n]) - np.real(I3[k,n]))* \
            (np.imag(I2[k,n]) - np.imag(I4[k,n])) + (np.real(I2[k,n]) - np.real(I4[k,n]))* \
            (np.imag(I3[k,n]) - np.imag(I1[k,n]))) # Equation (13)

            # print(np.absolute(M[k,n]))
            # print(maximum[k,n])
            if (np.absolute(M[k,n])) > maximum[k,n] or math.isnan(np.absolute(M[k,n])):
                M[k,n] = CS[k,n] # This removes the really big singularities; without this the image is mostly black
                replacements += 1

    # Calculate the weight w for each pixel
    w1 = np.zeros(I1.shape,dtype=complex)
    w2 = np.zeros(I1.shape,dtype=complex)
    for k in range(0,I1.shape[0]-1):
        for n in range(0,I1.shape[1]-1):
            numerator1 = 0+0j
            denominator1 = 0+0j
            numerator2 = 0+0j
            denominator2 = 0+0j
            for x in range(-2,3):
                a = k + x
                for y in range(-2,3):
                    b = n + y

                    # if (a < 0) or (b < 0) or (a > I1.shape[0]-1) or (b > I1.shape[1]-1):
                    #   pass
                    # else:
                    if not ((a < 0) or (b < 0) or (a > I1.shape[0]-1) or (b > I1.shape[1]-1)):
                        numerator1 = numerator1 + (I3[a,b] - M[a,b]).conjugate() * (I3[a,b] - I1[a,b]) + \
                        (I3[a,b] - I1[a,b]).conjugate() * (I3[a,b] - M[a,b])

                        denominator1 = denominator1 + (I1[a,b] - I3[a,b]).conjugate() * (I1[a,b] - I3[a,b])

                        numerator2 = numerator2 + (I4[a,b] - M[a,b]).conjugate() * (I4[a,b] - I2[a,b]) + \
                        (I4[a,b] - I2[a,b]).conjugate() * (I4[a,b] - M[a,b])

                        denominator2 = denominator2 + (I2[a,b] - I4[a,b]).conjugate() * (I2[a,b] - I4[a,b])

            w1[k,n] = numerator1 / (2*denominator1) # Equation (18) - first pair
            w2[k,n] = numerator2 / (2*denominator2) # Equation (18) - second pair

    # Calculate the average weighted sum of image pairs
    # Equation (14) - averaged
    I = (np.multiply(I1,w1) + np.multiply(I3,(1 - w1)) + np.multiply(I2,w2) + np.multiply(I4,(1 - w2))) / 2
    #print("Replacements: ", replacements)
    #return([I,replacements])
    return I

# Import test data to try it out
# I1 = load_image_data('../meas_MID164_trufi_phi0_FID6709.dat')
# I1_averaged = np.average(I1, axis=3)
# print("Imported I1...")
# print(I1.shape)

# I2 = load_image_data('../meas_MID165_trufi_phi90_FID6710.dat')
# I2_averaged = np.average(I2, axis=3)
# print("Imported I2...")

# I3 = load_image_data('../meas_MID166_trufi_phi180_FID6711.dat')
# I3_averaged = np.average(I3, axis=3)
# print("Imported I3...")

# I4 = load_image_data('../meas_MID167_trufi_phi270_FID6712.dat')
# I4_averaged = np.average(I4, axis=3)
# print("Imported I4...")

# [I, replacements] = elliptical_model(I1_averaged, I2_averaged, I3_averaged, I4_averaged)
# scipy.misc.imsave('out.jpg',np.absolute(I))
# print("Image saved!")


#average the fourth dimension
#np.reshape to reshape D x W x H