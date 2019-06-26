# Brain Structure Segmentation using Adversarial Learning
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

The architecture of 3D GAN is inspired from [1] and shown in figure below.
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

## Some results from our paper

* Visual comparison of the segmentation by each model, for two test subjects of the iSEG-2017 dataset, when training with different numbers of labeled examples.
<p float="left">
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject9.jpg" width="420" />
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject10.jpg" width="420" /> 
</p>

* Segmentation of Subject 10 of the iSEG-2017 dataset predicted by different GAN-based models, when trained with 2 labeled images. The red box highlights a region in the ground truth where all these models give noticeable differences.
<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/ganwar_mod.jpg" width="820"/>
<br>
* More such results can be found in the paper.

## Contact
You can mail me at: sanu.arnab@gmail.com  
If you use this code for your research, please consider citing the original paper:

- [1] [Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241)