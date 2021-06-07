# import pydegensac
from time import time

import cv2
import numpy as np
import pandas as pd

import utils


def match_images(needleFolderPath, haystackFolderPath, output_dir):
    frame = pd.DataFrame({'img1': [], 'img2': [], 'y_pred': [], })
    needleImageUrls = utils.getImageUrls(needleFolderPath)
    needleImageData = utils.getD2Urls(needleFolderPath)
    haystackImageUrls = utils.getImageUrls(haystackFolderPath)
    haystackImageData = utils.getD2Urls(haystackFolderPath)
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

            H, inliers = cv2.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)
            # H, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)

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
            path3 = output_dir + "/output_" + str(i) + "_" + str(j) + ".jpg"

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
