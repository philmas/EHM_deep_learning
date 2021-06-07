import cv2
import numpy as np
import scipy
import scipy.io
import scipy.misc
import torch
from tqdm import tqdm

from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image
from utils import getImageUrls


def extract_images(image_folder_path, max_edge, max_sum_edge, multiscale, model, relu, cuda, preprocessing, device,
                   output_type, output_extension):
    model = D2Net(model_file=model, use_relu=relu, use_cuda=cuda)
    imageURLs = getImageUrls(image_folder_path)
    for path in tqdm(imageURLs, total=len(imageURLs)):

        # imread returns (height, width, num_channels)
        image = cv2.imread(path)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        resized_image = image
        if max(resized_image.shape) > max_edge:
            fraction = max_edge / max(resized_image.shape)
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        if sum(resized_image.shape[: 2]) > max_sum_edge:
            fraction = max_sum_edge / sum(resized_image.shape[: 2])
            width = int(resized_image.shape[0] * fraction)
            height = int(resized_image.shape[1] * fraction)
            dim = (width, height)
            resized_image = cv2.resize(resized_image, dim).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=preprocessing
        )
        with torch.no_grad():
            if multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        if output_type == 'npz':
            with open(path + output_extension, 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=keypoints,
                    scores=scores,
                    descriptors=descriptors
                )
        elif output_type == 'mat':
            with open(path + output_extension, 'wb') as output_file:
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
