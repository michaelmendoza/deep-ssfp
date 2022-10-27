# DeepSSFP 

DeepSSFP is library for image reconstuction for bSSFP. This library contains code implementations for a variety of deep learning techniques for accelerated bSSFP MRI imaging. These techiques are as follows:

1. Deep Learning for bSSFP Banding Reduction
2. Deep Learning Super-FOV for Accelerated bSSFP Banding Reduction
3. Synthetic Banding for bSSFP Data Augmentation for Banding Reduction
4. Synthetic Banding for bSSFP Data Augmentation for T1/T2 Map Estimation

## Introduction

SSFP is sensitive to off-resonance effects, which cause banding artifacts. Multiple SSFP images with different phase cycle amounts can be combined to suppress banding artifacts and for the estimation of quantitative biomarker like T1/T2 relaxation parameter mappings. Multiple methods for band suppression have been developed over the years, but each method has limitations. Often the most significant limitation for effective band reduction is the number of phased-cycled SSFP images required. Our research aimed to develop a machine learning-based model to combine multiple bSSFP images for improved banding artifact suppression and quantitative biomarker estimation with a reduced number of bSSFP acquisitions.

## Notebooks

Jupyter notebooks for examples of how to use the DeepSSFP library.

1. Deep Learning bSSFP Banding Reduction ([notebook](notebooks/deepssfp_bandremoval4_experiment_v0.ipynb))
2. Super-FOV for Accelerated bSSFP Banding Reduction ([notebook](notebooks/deepssfp_superfov_experiment_v0.ipynb))
3. Synthetic Banding for bSSFP Data Augmentation ([notebook](notebooks/deepssfp_syntheticbanding_experiment_v0.ipynb))

## Development

This project requires python 3.8+ and has the following dependancies: 
numpy, matplotlib, scikit-image, mapvbvd, tensorflow and jupyterlab.

To setup a python enviroment with conda:

1. Create a new conda environment from scatch 
> ```
> conda create -n deepssfp python=3.8 
> conda activate deepssfp
> ```
> Then install packages with pip:
> ```
> pip install numpy matplotlib scipy scikit-image seabornpy mapvbvd tensorflow jupyterlab 
> ```

- Run to API server
```
npm run api
```

## Features

Implementations variety of deep learning techniques SSFP Band Removal is included in this library. 

### Deep Learning for bSSFP Banding Reduction

<p align="center">
  <img src="assets/DeepSSFP-diagram.png" width="600">
</p>

SSFP banding artifact reduction can be recast as a supervised learning task. The relationship between multiple acquisition bSSFP image data and a band-free image emerges during the supervised learning from training. Applying supervised learning to banding artifact reduction reduces scan time by reducing the required number of phase-cycled bSSFP acquisitions. A U-Net architecture was designed and trained to model the elliptical signal model for band removal. Banding artifact-free images were generated as the ground truth data and the target image for training an end-to-end deep learning network. 

<p align="center">
  <img src="assets/DeepSSFP-results.png" width="600">
</p>

The ground truth data was generated using four phased cycled images as inputs to the elliptical signal model to generate a band-reduced image. 
Banding artifact-free images were generated using our model from two input images. The results of the band reduction using our deep learning models were compared against multiple standard methods for banding artifact suppression, including the maximum-intensity project (MIP), the elliptical signal model, and the sum of squares.

### Deep Learning Super-FOV for Accelerated bSSFP Banding Reduction

<p align="center">
  <img src="assets/sFOV-diagram.png" width="600">
</p>

We present a machine learning technique for bSSFP band removal using two undersampled phase-cycled bSSFP image acquisitions. A deep convolutional neural network was trained to solve a generalized SENSE reconstruction problem where bSSFP banding sensitivities are used instead of coil sensitivity maps. We demonstrate that a deep neural network can reduce banding artifacts in multiple acquisition bSSFP comparable to the elliptical signal model and reduce overall scan time by requiring half as many phase-cycled images.

<p align="center">
  <img src="assets/sFOV-results.png" width="600">
</p>


In our method, Undersampled k-space data from two phase-cycled acquisitions is fed to the model as input. The output is compared using mean squared error to the geometric solution to the elliptical signal model generated from four fully sampled phase-cycled acquisitions. The results of the technique show that our method is capable of generating banding artifact-free images of simliar quality compared to traditional methods with undersampled data. 

### Synthetic Banding for bSSFP Data Augmentation

<p align="center">
  <img src="assets/SyntheticBanding-diagram.png" width="600">
</p>

We present a deep learning method for synthesizing additional phase-cycled images from a subset of images that can be used with existing band reduction techniques to reduce scan time. Additionally, this method can be used for quantitative parameter biomarker estimation to reduce the number of images required in T1/T2 map estimation techiqiues such as PLANET.

<p align="center">
  <img src="assets/SyntheticBanding-results.png" width="600">
</p>

Examples of synthetically genearted images are demonstated above where 2 phase cycled images (0 &deg;, 180 &deg;) are used to generate 2 other phase cycled images (90 &deg;, 270 &deg;). These results show that the technique is capable of generating accurate synthetic phase cycled images.

<p align="center">
  <img src="assets/SyntheticBanding-diagram2.png" width="600">
</p>

<p align="center">
  <img src="assets/SyntheticBanding-results2.png" width="600">
</p>

The functionality of this techique is validated by using synthetically generated images to create a band free image using the elliptical singal model techique for band reduction. This was compared to the traditional method for banding artifact reduction. The results show that our method is comparable to the more traditonal techique with few input images.

