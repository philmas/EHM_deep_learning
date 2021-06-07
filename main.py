import glob
import os
from time import time

import cv2
import numpy as np
import pandas as pd
import pydegensac
import scipy
import scipy.io
import scipy.misc
import torch
from tqdm import tqdm

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image

# PARAMETERS

BASE = ""
OUTPUT = BASE + "output/"
INPUT = BASE + "input/"
MODULE_FILE = BASE + 'models/d2_tf.pth'

# D2NET METHODS
PREPROCESSING = 'caffe'
USE_RELU = True
OUTPUT_TYPE = 'npz'
MULTISCALE = True

# MAX EDGE SIZE (WIDTH OR HEIGHT)
MAX_EDGE = 1600

# MAX SUM OF EDGES (WIDTH + HEIGHT)
MAX_SUM_EDGES = 3200

# EXTRACTED FILE EXTENSION
OUTPUT_EXTENSION = '.d2-net'
# FUNCTION
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


# TODO SUPPORT PNG
def getImageUrls(folder):
    # if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    return glob.glob(os.path.join(folder, '*/*.jpg'))


def getD2Urls(folder):
    return glob.glob(os.path.join(folder, '*/*.d2-net'))


def goExtractNow(imageFolderPath):
    model = D2Net(model_file=MODULE_FILE, use_relu=USE_RELU, use_cuda=USE_CUDA)
    imageURLs = getImageUrls(imageFolderPath)
    for path in tqdm(imageURLs, total=len(imageURLs)):

        # imread returns (height, width, num_channels)
        image = cv2.imread(path)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > MAX_EDGE:
            fraction = MAX_EDGE / max(resized_image.shape)
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        if sum(resized_image.shape[: 2]) > MAX_SUM_EDGES:
            fraction = MAX_SUM_EDGES / sum(resized_image.shape[: 2])
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=PREPROCESSING
        )
        with torch.no_grad():
            if MULTISCALE:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=DEVICE
                    ),
                    model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=DEVICE
                    ),
                    model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        if OUTPUT_TYPE == 'npz':
            with open(path + OUTPUT_EXTENSION, 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=keypoints,
                    scores=scores,
                    descriptors=descriptors
                )
        elif OUTPUT_TYPE == 'mat':
            with open(path + OUTPUT_EXTENSION, 'wb') as output_file:
                scipy.io.savemat(
                    output_file,
                    {
                        'keypoints': keypoints,
                        'scores': scores,
                        'descriptors': descriptors
                    }
                )
        else:
            raise ValueError('Unknown output type.')
    del model
    torch.cuda.empty_cache()


def goMatch(needleFolderPath, haystackFolderPath):
    frame = pd.DataFrame({'img1': [], 'img2': [], 'y_pred': [], })
    needleImageUrls = getImageUrls(needleFolderPath)
    needleImageData = getD2Urls(needleFolderPath)
    haystackImageUrls = getImageUrls(haystackFolderPath)
    haystackImageData = getD2Urls(haystackFolderPath)
    for i in range(len(needleImageUrls)):
        image1 = np.array(cv2.imread(needleImageUrls[i]))
        feat1 = np.load(needleImageData[i])

        for j in range(len(haystackImageUrls)):
            print("Analyzing...")
            image2 = np.array(cv2.imread(haystackImageUrls[j]))
            feat2 = np.load(haystackImageData[j])

            t0 = time()
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(feat1['descriptors'], feat2['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            t1 = time()

            print("Time to extract matches: ", t1 - t0)
            print("Number of raw matches:", len(matches))

            match1 = [m.queryIdx for m in matches]
            match2 = [m.trainIdx for m in matches]

            keypoints_left = feat1['keypoints'][match1, : 2]
            keypoints_right = feat2['keypoints'][match2, : 2]

            np.random.seed(0)

            t0 = time()

            # H, inliers = cv2.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)
            H, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)

            t1 = time()
            print("Time for ransac: ", t1 - t0)

            n_inliers = np.sum(inliers)
            print('Number of inliers: %d.' % n_inliers)
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               # matchesMask = matchesMask,
                               flags=0)
            image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches,
                                     None, **draw_params)
            path3 = OUTPUT + "/output_" + str(i) + "_" + str(j) + ".jpg"

            pred = 0
            if n_inliers > 10:
                # if n_inliers/matches.shape[0] > 0.01:
                print("Match has been found!")
                pred = 1
            else:
                print("Match not found!")

            new_row = {'img1': str(needleImageUrls[i]), 'img2': str(haystackImageUrls[j]), 'y_pred': pred}
            frame = frame.append(new_row, ignore_index=True)
            cv2.imwrite(path3, image3)

    return frame


goExtractNow(INPUT)
writer = pd.ExcelWriter('opend2results.xlsx', engine='xlsxwriter')
df = goMatch(INPUT, INPUT)

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
