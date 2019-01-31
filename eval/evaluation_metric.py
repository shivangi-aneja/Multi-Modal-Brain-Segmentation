# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial


labels = {
          0: 'Background',
          1: 'Cortical gray matter',
          2: 'Basal ganglia',
          3: 'White matter',
          4: 'White matter lesions',
          5: 'Cerebrospinal fluid in the extracerebral space',
          6: 'Ventricles',
          7: 'Cerebellum',
          8: 'Brain stem',
          # The two labels below are ignored:
          # 9: 'Infarction',
          # 10: 'Other',
          }


def evaluate_stats(gt_mask, pred_mask, count):

    for i in range(count):
        dsc = getDSC(gt_mask[i],pred_mask[i])
        h95 = getHausdorff(gt_mask[i],pred_mask[i])
        vs = getVS(gt_mask[i],pred_mask[i])

        print('Dice', dsc, '(higher is better, max=1)')
        print('HD', h95, 'mm', '(lower is better, min=0)')
        print('VS', vs, '(higher is better, max=1)')



def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    dsc = dict()
    for k in labels.keys():
        testArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(testImage, k, k, 1, 0)).flatten()
        resultArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(resultImage, k, k, 1, 0)).flatten()

        # similarity = 1.0 - dissimilarity
        # scipy.spatial.distance.dice raises a ZeroDivisionError if both arrays contain only zeros.
        try:
            dsc[k] = 1.0 - scipy.spatial.distance.dice(testArray, resultArray)
        except ZeroDivisionError:
            dsc[k] = None

    return dsc


def getHausdorff(testImage, resultImage):
    """Compute the 95% Hausdorff distance."""
    hd = dict()
    for k in labels.keys():
        lTestImage = sitk.BinaryThreshold(testImage, k, k, 1, 0)
        lResultImage = sitk.BinaryThreshold(resultImage, k, k, 1, 0)

        # Hausdorff distance is only defined when something is detected
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(lTestImage)
        lTestSum = statistics.GetSum()
        statistics.Execute(lResultImage)
        lResultSum = statistics.GetSum()
        if lTestSum == 0 or lResultSum == 0:
            hd[k] = None
            continue

        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage = sitk.BinaryErode(lTestImage, (1, 1, 0))
        eResultImage = sitk.BinaryErode(lResultImage, (1, 1, 0))

        hTestImage = sitk.Subtract(lTestImage, eTestImage)
        hResultImage = sitk.Subtract(lResultImage, eResultImage)

        hTestArray = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)

        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
        testCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                           np.transpose(np.flipud(np.nonzero(hTestArray)))]
        resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                             np.transpose(np.flipud(np.nonzero(hResultArray)))]

        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]

        # Compute distances from test to result and vice versa.
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))

    return hd


def getVS(testImage, resultImage):
    """Volume similarity.
    VS = 1 - abs(A - B) / (A + B)
    A = ground truth in ML
    B = participant segmentation in ML
    """
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    vs = dict()
    for k in labels.keys():
        testStatistics.Execute(sitk.BinaryThreshold(testImage, k, k, 1, 0))
        resultStatistics.Execute(sitk.BinaryThreshold(resultImage, k, k, 1, 0))

        numerator = abs(testStatistics.GetSum() - resultStatistics.GetSum())
        denominator = testStatistics.GetSum() + resultStatistics.GetSum()

        if denominator > 0:
            vs[k] = 1 - float(numerator) / denominator
        else:
            vs[k] = None

    return vs