#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Shivangi
"""

from __future__ import print_function
import scipy
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import nibabel as nib

train_list = ['1', '4', '5', '70']
val_list = ['148']
test_list = ['7', '14']
unlabelled_list = ['0','2','4','6','8','11','17','19','21','23','25']
cut = 10
thresh = 10


def normalize_annotated_images(data_dir, dir_list, dest_dir):
    """

    :param dir_list:
    :return:
    """
    for dir_name in dir_list:

        # Read the segmentation mask and cut the borders
        seg_img = sitk.ReadImage(os.path.join(data_dir, dir_name, 'segm.nii.gz'))
        seg_array = sitk.GetArrayFromImage(seg_img)
        seg_array = seg_array[:, cut:np.shape(seg_array)[1] - cut, cut:np.shape(seg_array)[2] - cut]

        # Read the FLAIR image and cut the borders
        flair_img = sitk.ReadImage(os.path.join(data_dir, dir_name, 'FLAIR.nii.gz'))
        flair_array = sitk.GetArrayFromImage(flair_img)

        # Read the T1 image and cut the borders
        t1_img = sitk.ReadImage(os.path.join(data_dir, dir_name, 'reg_T1.nii.gz'))
        t1_array = sitk.GetArrayFromImage(t1_img)


        # Normalize the cut FLAIR image
        brain_mask_flair = np.zeros(np.shape(flair_array), dtype='float32')
        brain_mask_flair[flair_array >= thresh] = 1
        brain_mask_flair[flair_array < thresh] = 0
        for iii in range(np.shape(flair_array)[0]):
            brain_mask_flair[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask_flair[iii, :, :])  # fill the holes inside brain
        flair_array = flair_array - np.mean(flair_array[brain_mask_flair == 1])
        flair_array /= np.std(flair_array[brain_mask_flair == 1])
        flair_array = flair_array[:, cut:np.shape(flair_array)[1] - cut, cut:np.shape(flair_array)[2] - cut]


        # Normalize the cut regualrized T1 image
        brain_mask_t1 = np.zeros(np.shape(t1_array), dtype='float32')
        brain_mask_t1[t1_array >= thresh] = 1
        brain_mask_t1[t1_array < thresh] = 0
        for iii in range(np.shape(t1_array)[0]):
            brain_mask_t1[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask_t1[iii, :, :])  # fill the holes inside br
        t1_array = t1_array - np.mean(t1_array[brain_mask_t1 == 1])
        t1_array /= np.std(t1_array[brain_mask_t1 == 1])
        t1_array = t1_array[:, cut:np.shape(t1_array)[1] - cut, cut:np.shape(t1_array)[2] - cut]

        if not os.path.exists(os.path.join(dest_dir, dir_name)):
            os.makedirs(os.path.join(dest_dir, dir_name))

        # Save the segmentation mask
        seg_image = nib.Nifti1Image(np.transpose(seg_array,[1,2,0]), None)
        imgname = 'segm.nii.gz'
        nib.save(seg_image, os.path.join(os.path.join(dest_dir, dir_name), imgname))

        # Save the FLAIR image
        flair_image = nib.Nifti1Image(np.transpose(flair_array,[1,2,0]), None)
        imgname = 'FLAIR.nii.gz'
        nib.save(flair_image, os.path.join(os.path.join(dest_dir, dir_name), imgname))

        # Save the T1 image
        t1_image = nib.Nifti1Image(np.transpose(t1_array,[1,2,0]), None)
        imgname = 'reg_T1.nii.gz'
        nib.save(t1_image, os.path.join(os.path.join(dest_dir, dir_name), imgname))


def normalize_unannotated_images(data_dir, dir_list, dest_dir):
    """

    :param dir_list:
    :return:
    """
    for dir_name in dir_list:

        print("Start " + os.path.join(data_dir, dir_name))
        # Read the FLAIR image and cut the borders
        flair_img = sitk.ReadImage(os.path.join(data_dir, dir_name, 'FLAIR.nii.gz'))
        flair_array = sitk.GetArrayFromImage(flair_img)

        # Read the T1 image and cut the borders
        t1_img = sitk.ReadImage(os.path.join(data_dir, dir_name, 'T1.nii.gz'))
        t1_array = sitk.GetArrayFromImage(t1_img)


        # Normalize the cut FLAIR image
        brain_mask_flair = np.zeros(np.shape(flair_array), dtype='float32')
        brain_mask_flair[flair_array >= thresh] = 1
        brain_mask_flair[flair_array < thresh] = 0
        for iii in range(np.shape(flair_array)[0]):
            brain_mask_flair[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask_flair[iii, :, :])  # fill the holes inside brain
        flair_array = flair_array - np.mean(flair_array[brain_mask_flair == 1])
        flair_array /= np.std(flair_array[brain_mask_flair == 1])
        flair_array = flair_array[:, cut:np.shape(flair_array)[1] - cut, cut:np.shape(flair_array)[2] - cut]


        # Normalize the cut regualrized T1 image
        brain_mask_t1 = np.zeros(np.shape(t1_array), dtype='float32')
        brain_mask_t1[t1_array >= thresh] = 1
        brain_mask_t1[t1_array < thresh] = 0
        for iii in range(np.shape(t1_array)[0]):
            brain_mask_t1[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                brain_mask_t1[iii, :, :])  # fill the holes inside br
        t1_array = t1_array - np.mean(t1_array[brain_mask_t1 == 1])
        t1_array /= np.std(t1_array[brain_mask_t1 == 1])
        t1_array = t1_array[:, cut:np.shape(t1_array)[1] - cut, cut:np.shape(t1_array)[2] - cut]

        if not os.path.exists(os.path.join(dest_dir, dir_name)):
            os.makedirs(os.path.join(dest_dir, dir_name))

        # Save the FLAIR image
        flair_image = nib.Nifti1Image(np.transpose(flair_array,[1,2,0]), None)
        imgname = 'FLAIR.nii.gz'
        nib.save(flair_image, os.path.join(os.path.join(dest_dir, dir_name), imgname))

        # Save the T1 image
        t1_image = nib.Nifti1Image(np.transpose(t1_array,[1,2,0]), None)
        imgname = 'T1.nii.gz'
        nib.save(t1_image, os.path.join(os.path.join(dest_dir, dir_name), imgname))
        print("End " + os.path.join(data_dir, dir_name))


if __name__ == '__main__':

    # Normalize training data
    normalize_annotated_images(data_dir=os.getcwd() + "/data/mrbrains/train/", dir_list=train_list,
                               dest_dir=os.getcwd()+"/data/mrbrains_normalized/train/")
    # Normalize validation data
    normalize_annotated_images(data_dir=os.getcwd() + "/data/mrbrains/val/", dir_list=val_list,
                               dest_dir=os.getcwd() + "/data/mrbrains_normalized/val/")
    # Normalize test data
    normalize_annotated_images(data_dir=os.getcwd() + "/data/mrbrains/test/", dir_list=test_list,
                               dest_dir=os.getcwd() + "/data/mrbrains_normalized/test/")

    # Normalize unlabelled data
    normalize_unannotated_images(data_dir=os.getcwd() + "/data/mrbrains/unlabelled/", dir_list=unlabelled_list,
                               dest_dir=os.getcwd() + "/data/mrbrains_normalized/unlabelled/")