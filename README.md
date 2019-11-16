# 3D Multi Modal Brain Structure Segmentation using Adversarial Learning

This work has been accepted at 14th WiML Workshop, NeurIPS Conference 2019. Please find the poster [here](https://www.academia.edu/40950224/3D_Multi_Modal_Semantic_Segmentation_Using_Adversarial_Learning).

## Requirements

- The code has been written in Python (3.5.2) and Tensorflow (1.7.0)
- Make sure to install all the libraries given in requirement.txt (You can do so by the following command)
```
pip install -r requirement.txt
```

## Dataset
The annotated dataset was proviede by [MR Brains 2018](https://mrbrains18.isi.uu.nl/) for Grand Challenge on MR Brain
Segmentation at MICCAI 2018. This data consists of 7 sets of annotated brain MR images
(T1, T1 inversion recovery, and T2-FLAIR) with manual segmentations. These manual
segmentations have been made by experts in brain segmentation. Images were acquired on
a 3T scanner at the UMC Utrecht (the Netherlands).

The unannotated dataset is provided by [WMH Segmentation Challenge](https://wmh.isi.uu.nl/). This data
consists of brain MR images (T1 and T2-FLAIR). So we used only two modalities for training.

## How to use the code?
* Download the dataset and place it in data folder. 
```
$ python normalize_data.py
```
* The preprocessed images will be stored in mrbrains_normalized folder
* You can run standard 3D U-Net & 3D GAN (both Feature matching GAN and bad GAN) with this code and compare their performance.

## 3D U-Net

The architecture of 3D Unet used is shown in the figure below.

<p float="center">
  <img src="/images/unet.png" width="50%%" />
</p>

### How to run 3D U-Net?
```
$ cd multi_modal_gan
```
* Configure the flags according to your experiment.
* To run training
```
$ python train_3dunet.py --training
```
* This will train your model and save the best checkpoint according to your validation performance. 
* You can also resume training from saved checkpoint by setting the load_chkpt flag.
* You can run the testing to predict segmented output which will be saved in your result folder as ".nii.gz" files.
* To run testing
```
$ python train_3dunet.py--testing
```
* This code computes dice coefficient to evaluate the testing performance. Once the output segmented images are created you can use them to compute any other evaluation metrics : Hausdorff Distance and Volumetric Similarity

## 3D GAN

The architecture of 3D GAN used is shown in figure below. Parts of code are referenced from [1].
<p float="center">
  <img src="/images/gan.png" width="800px" />
</p>

### How to run 3D GAN?
```
$ cd multi_modal_gan
```
* Configure the flags according to your experiment.
* To run training
```
$ python train_3dgan.py --training
```
* By default it trains Feature Matching GAN based model. To train the bad GAN based model
```
$ python train_3dgan.py --training --badGAN
```
* To run testing
```
$ python train_3dgan.py --testing
``` 

## Results

#### Loss Curves
The training curves are shown in the figure below

<p float="left">
  <img src="/images/1_supervised_loss.png" width="24%" />
  <img src="/images/feature_match_loss.png" width="24%" /> 
  <img src="/images/fk_img_loss.png" width="24%" /> 
  <img src="/images/unsupervised_loss.png" width="24%" /> 
</p>


#### Dice Score Comparison over epochs on Validation Set

<p float="left">
  <img src="/images/1_basal_dice.png" width="29%" />
  <img src="/images/1_basal_ganglia_dice.png" width="29%" /> 
  <img src="/images/1_brain_stem_dice.png" width="29%" /> 
</p>

<p float="left">
  <img src="/images/1_cerebellum_dice.png" width="29%" />
  <img src="/images/1_cerebrospinal_fluid_dice.png" width="29%" /> 
  <img src="/images/1_cortical_gray_matter_dice.png" width="29%" /> 
</p>

<p float="left">
  <img src="/images/1_ventricles_dice.png" width="29%" />
  <img src="/images/1_white_matter_dice.png" width="29%" /> 
  <img src="/images/1_white_matter_lesions_dice.png" width="29%" /> 
</p>


#### Visual comparison of the segmentation by 3D Unet vs 3D GAN
<p float="left">
  <img src="/images/1_unet.png" width="29%" />
  <img src="/images/1_gan.png" width="29%" /> 
  <img src="/images/1_gt.png" width="29%" /> 
</p>

<p float="left">
  <img src="/images/14_unet.png" width="29%" />
  <img src="/images/14_gan.png" width="29%" /> 
  <img src="/images/14_gt.png" width="29%" /> 
</p>


|                  3D UNET               |             3D GAN               |             Ground Truth       |        
| -------------------------------------- | -------------------------------- | ------------------------------ |

## Contact
You can mail me at: shivangi.tum@gmail.com  

[1] [Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241)
