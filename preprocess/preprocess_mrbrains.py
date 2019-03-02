import glob
import os
import shutil

import nibabel as nib
import numpy as np
import tensorflow as tf
from nipype.interfaces.ants import N4BiasFieldCorrection
from sklearn.utils import shuffle

# import pdb

F = tf.app.flags.FLAGS

seed = 0
np.random.seed(seed)

all_modalities = {'FLAIR', 'reg_T1'}


def get_filename(set_name, case_idx, input_name, loc):
    pattern = '{0}/{1}/{2}/{3}.nii.gz'
    return pattern.format(loc, set_name, case_idx, input_name)


# def get_set_name(case_idx):
#     return 'Training' if case_idx < 11 else 'Testing'


def read_data(case_idx, input_name, loc,set_name):
    #set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)
    print(image_path)

    return nib.load(image_path)


def read_vol(case_idx, input_name, dir,mode):
    image_data = read_data(case_idx, input_name, dir,mode)
    return image_data.get_data()


def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image


def normalise(case_idx, input_name, in_dir, out_dir,set_name, copy=False):
    #set_name = get_set_name(case_idx)
    image_in_path = get_filename(set_name, case_idx, input_name, in_dir)
    image_out_path = get_filename(set_name, case_idx, input_name, out_dir)
    if copy:
        shutil.copy(image_in_path, image_out_path)
    else:
        correct_bias(image_in_path, image_out_path)
    print(image_in_path + " done.")


"""
To extract patches from a 3D image
"""


def extract_patches(volume, patch_shape, extraction_step, datype='float32'):
    patch_h, patch_w, patch_d = patch_shape[0], patch_shape[1], patch_shape[2]
    stride_h, stride_w, stride_d = extraction_step[0], extraction_step[1], extraction_step[2]
    img_h, img_w, img_d = volume.shape[0], volume.shape[1], volume.shape[2]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_d = (img_d - patch_d) // stride_d + 1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    raw_patch_martrix = np.zeros((N_patches_img, patch_h, patch_w, patch_d), dtype=datype)
    k = 0

    # iterator over all the patches
    for h in range((img_h - patch_h) // stride_h + 1):
        for w in range((img_w - patch_w) // stride_w + 1):
            for d in range((img_d - patch_d) // stride_d + 1):
                raw_patch_martrix[k] = volume[h * stride_h:(h * stride_h) + patch_h, \
                                       w * stride_w:(w * stride_w) + patch_w, \
                                       d * stride_d:(d * stride_d) + patch_d]
                k += 1
    assert (k == N_patches_img)
    return raw_patch_martrix


"""
To extract labeled patches from array of 3D labeled images
"""


def get_patches_lab(FLAIR_vols, reg_T1_vols, label_vols, extraction_step,
                    patch_shape, validating, testing, num_images_training):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 2), dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d), dtype="uint8")
    for idx in range(len(FLAIR_vols)):
        y_length = len(y)
        if testing:
            print(("Extracting Patches from Labelled Image %2d ....") % (num_images_training + idx + 2))
        elif validating:
            print(("Extracting Patches from Labelled Image %2d ....") % (num_images_training + idx + 1))
        else:
            print(("Extracting Patches from Labelled Image %2d ....") % (1 + idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,
                                        datype="uint8")

        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d,
                                    patch_shape_1d, patch_shape_1d, 2), dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d,
                                    patch_shape_1d, patch_shape_1d), dtype="uint8")))

        y[y_length:, :, :, :] = label_patches

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        FLAIR_train = extract_patches(FLAIR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 0] = FLAIR_train[valid_idxs]

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        reg_T1_train = extract_patches(reg_T1_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :, 1] = reg_T1_train[valid_idxs]
    return x, y


"""
To preprocess the labelled training data
"""


def preprocess_dynamic_lab(dir, num_classes, extraction_step, patch_shape, num_images_training=4,
                           validating=False, testing=False, num_images_testing=2):
    train_idx = [1, 4, 5, 70]
    val_idx = [148]
    test_idx = [7, 14]
    cases = None
    mode = None

    if testing:
        print("Testing")
        r1 = num_images_training + 2
        r2 = num_images_training + num_images_testing + 2
        c = num_images_training + 1
        FLAIR_vols = np.empty((num_images_testing, 240, 240,48), dtype="float32")
        reg_T1_vols = np.empty((num_images_testing, 240, 240,48), dtype="float32")
        label_vols = np.empty((num_images_testing, 240, 240,48), dtype="uint8")
        mode = 'test'
        cases = test_idx
    elif validating:
        print("Validating")
        r1 = num_images_training + 1
        r2 = num_images_training + 2
        c = num_images_training
        FLAIR_vols = np.empty((1, 240, 240,48), dtype="float32")
        reg_T1_vols = np.empty((1, 240, 240,48), dtype="float32")
        label_vols = np.empty((1, 240, 240,48), dtype="uint8")
        mode = 'val'
        cases = val_idx
    else:
        print("Training")
        r1 = 1
        r2 = num_images_training + 1
        c = 0
        FLAIR_vols = np.empty((num_images_training,240, 240,48), dtype="float32")
        reg_T1_vols = np.empty((num_images_training, 240, 240,48), dtype="float32")
        label_vols = np.empty((num_images_training, 240, 240,48), dtype="uint8")
        mode = 'train'
        cases = train_idx


    iter = 0
    for case_idx in range(r1, r2):
        print(case_idx)
        FLAIR_vols[(case_idx - c - 1), :, :, :] = read_vol(cases[iter], 'FLAIR', dir,mode)
        reg_T1_vols[(case_idx - c - 1), :, :, :] = read_vol(cases[iter], 'reg_T1', dir,mode)
        label_vols[(case_idx - c - 1), :, :, :] = read_vol(cases[iter], 'segm', dir,mode)
        iter += 1
    FLAIR_mean = FLAIR_vols.mean()
    FLAIR_std = FLAIR_vols.std()
    FLAIR_vols = (FLAIR_vols - FLAIR_mean) / FLAIR_std
    reg_T1_mean = reg_T1_vols.mean()
    reg_T1_std = reg_T1_vols.std()
    reg_T1_vols = (reg_T1_vols - reg_T1_mean) / reg_T1_std

    for i in range(FLAIR_vols.shape[0]):
        FLAIR_vols[i] = ((FLAIR_vols[i] - np.min(FLAIR_vols[i])) /
                      (np.max(FLAIR_vols[i]) - np.min(FLAIR_vols[i]))) * 255
    for i in range(reg_T1_vols.shape[0]):
        reg_T1_vols[i] = ((reg_T1_vols[i] - np.min(reg_T1_vols[i])) /
                      (np.max(reg_T1_vols[i]) - np.min(reg_T1_vols[i]))) * 255
    FLAIR_vols = FLAIR_vols / 127.5 - 1.
    reg_T1_vols = reg_T1_vols / 127.5 - 1.
    x, y = get_patches_lab(FLAIR_vols, reg_T1_vols, label_vols, extraction_step, patch_shape, validating=validating,
                           testing=testing, num_images_training=num_images_training)
    print("Total Extracted Labelled Patches Shape:", x.shape, y.shape)
    if testing:
        return x, label_vols
    elif validating:
        return x, y, label_vols
    else:
        return x, y


"""
To extract labeled patches from array of 3D unlabeled images
"""


def get_patches_unlab(FLAIR_vols, reg_T1_vols, extraction_step, patch_shape, dir):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    label_ref = np.empty((1, 240,240,48), dtype="uint8")
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 2))
    label_ref = read_vol(70, 'segm', dir,"train")
    for idx in range(len(FLAIR_vols)):
        x_length = len(x)
        print(("Processing the Image Unlabelled %2d ....") % (idx + 11))
        label_patches = extract_patches(label_ref, patch_shape, extraction_step)

        # Select only those who are important for processing
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)

        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d,
                                    patch_shape_1d, patch_shape_1d, 2))))

        FLAIR_train = extract_patches(FLAIR_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 0] = FLAIR_train[valid_idxs]

        reg_T1_train = extract_patches(reg_T1_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 1] = reg_T1_train[valid_idxs]
    return x


"""
To preprocess the unlabeled training data
"""


def preprocess_dynamic_unlab(dir, extraction_step, patch_shape, num_images_training_unlab):
    FLAIR_vols = np.empty((num_images_training_unlab, 240, 240,48), dtype="float32")
    reg_T1_vols = np.empty((num_images_training_unlab, 240, 240,48), dtype="float32")
    cases = [0,2,4,6]
    for case_idx in range(len(cases)):
        FLAIR_vols[case_idx, :, :, :] = read_vol(cases[case_idx], 'FLAIR', dir,"unlabelled")
        reg_T1_vols[case_idx, :, :, :] = read_vol(cases[case_idx], 'T1', dir,"unlabelled")
        # print(read_vol(case_idx, 'T2', dir).shape)
    FLAIR_mean = FLAIR_vols.mean()
    FLAIR_std = FLAIR_vols.std()
    FLAIR_vols = (FLAIR_vols - FLAIR_mean) / FLAIR_std
    reg_T1_mean = reg_T1_vols.mean()
    reg_T1_std = reg_T1_vols.std()
    reg_T1_vols = (reg_T1_vols - reg_T1_mean) / reg_T1_std
    for i in range(FLAIR_vols.shape[0]):
        FLAIR_vols[i] = ((FLAIR_vols[i] - np.min(FLAIR_vols[i])) /
                      (np.max(FLAIR_vols[i]) - np.min(FLAIR_vols[i]))) * 255
    for i in range(reg_T1_vols.shape[0]):
        reg_T1_vols[i] = ((reg_T1_vols[i] - np.min(reg_T1_vols[i])) /
                      (np.max(reg_T1_vols[i]) - np.min(reg_T1_vols[i]))) * 255
    FLAIR_vols = FLAIR_vols / 127.5 - 1.
    reg_T1_vols = reg_T1_vols / 127.5 - 1.
    x = get_patches_unlab(FLAIR_vols, reg_T1_vols, extraction_step, patch_shape, dir)
    print("Total Extracted Unlabelled Patches Shape:", x.shape)
    return x


def preprocess_static(org_dir, prepro_dir, dataset="labeled", overwrite=False):
    if not os.path.exists(prepro_dir):
        os.makedirs(prepro_dir)
    for subject_folder in glob.glob(os.path.join(org_dir, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(prepro_dir,
                                              os.path.basename(os.path.dirname(subject_folder)), subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
    if dataset == "train":
            normalise(1, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(1, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(1, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)

            normalise(4, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(4, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(4, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)

            normalise(5, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(5, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(5, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)

            normalise(70, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(70, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(70, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)

    if dataset == "val":
            normalise(148, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(148, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(148, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)


    if dataset == "test":
            normalise(7, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(7, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(7, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)

            normalise(14, 'FLAIR', org_dir, prepro_dir,dataset)
            normalise(14, 'reg_T1', org_dir, prepro_dir,dataset)
            normalise(14, 'segm', org_dir, prepro_dir,dataset,
                      copy=True)
    else:
        for case_idx in range(11, 24):
            normalise(case_idx, 'T1', org_dir, prepro_dir,dataset)
            normalise(case_idx, 'T2', org_dir, prepro_dir,dataset)


"""
dataset class for preparing training data of basic U-Net
"""


class dataset(object):
    def __init__(self, num_classes, extraction_step, number_images_training, batch_size, patch_shape, data_directory):
        # Extract labelled and unlabelled patches
        self.batch_size = batch_size
        self.data_lab, self.label = preprocess_dynamic_lab(
            data_directory, num_classes, extraction_step,
            patch_shape, number_images_training)

        self.data_lab, self.label = shuffle(self.data_lab,
                                            self.label, random_state=0)
        print("Data_shape:", self.data_lab.shape)
        print("Data lab max and min:", np.max(self.data_lab), np.min(self.data_lab))
        print("Label unique:", np.unique(self.label))

    def batch_train(self):
        self.num_batches = len(self.data_lab) // self.batch_size
        for i in range(self.num_batches):
            yield self.data_lab[i * self.batch_size:(i + 1) * self.batch_size], \
                  self.label[i * self.batch_size:(i + 1) * self.batch_size]


"""
dataset_badGAN class for preparing data of our model
"""


class dataset_badGAN(object):
    def __init__(self, num_classes, extraction_step, number_images_training, batch_size,
                 patch_shape, number_unlab_images_training, data_directory):
        # Extract labelled and unlabelled patches,
        self.batch_size = batch_size
        self.data_lab, self.label = preprocess_dynamic_lab(
            data_directory, num_classes, extraction_step,
            patch_shape, number_images_training)

        self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)
        self.data_unlab = preprocess_dynamic_unlab(data_directory, extraction_step,
                                                   patch_shape, number_unlab_images_training)
        self.data_unlab = shuffle(self.data_unlab, random_state=0)

        # If training, repeat labelled data to make its size equal to unlabelled data
        factor = len(self.data_unlab) // len(self.data_lab)
        print("Factor for labeled images:", factor)
        rem = len(self.data_unlab) % len(self.data_lab)
        temp = self.data_lab[:rem]
        self.data_lab = np.concatenate((np.repeat(self.data_lab, factor, axis=0), temp), axis=0)
        temp = self.label[:rem]
        self.label = np.concatenate((np.repeat(self.label, factor, axis=0), temp), axis=0)
        assert (self.data_lab.shape == self.data_unlab.shape)
        print("Data_shape:", self.data_lab.shape, self.data_unlab.shape)
        print("Data lab max and min:", np.max(self.data_lab), np.min(self.data_lab))
        print("Data unlab max and min:", np.max(self.data_unlab), np.min(self.data_unlab))
        print("Label unique:", np.unique(self.label))

    def batch_train(self):
        self.num_batches = len(self.data_lab) // self.batch_size
        for i in range(self.num_batches):
            yield self.data_lab[i * self.batch_size:(i + 1) * self.batch_size], \
                  self.data_unlab[i * self.batch_size:(i + 1) * self.batch_size], \
                  self.label[i * self.batch_size:(i + 1) * self.batch_size]

# preprocess_static( actual_data_directory, preprocesses_data_directory, overwrite=True)

# For labelled data
# preprocess_static(org_dir="../data/mrbrains",prepro_dir= "../data/mrbrains_preprocessed",dataset="train", overwrite=True)
# preprocess_static(org_dir="../data/mrbrains",prepro_dir= "../data/mrbrains_preprocessed",dataset="val", overwrite=True)
# preprocess_static(org_dir="../data/mrbrains",prepro_dir= "../data/mrbrains_preprocessed",dataset="test", overwrite=True)
#
# # For unlabelled data
# preprocess_static(org_dir="../data/iSEG",prepro_dir= "../data/iSEG_preprocessed",dataset="unlabelled", overwrite=True)
