"""
Blob detection with DoG (Difference of Gaussian) and LoG (Laplacian of Gaussian)
"""

from skimage import feature, filters
from skimage.transform import resize
from skimage import morphology
import pandas as pd
import re
import numpy as np
import glob
import glob2
import matplotlib.pylab as plt

import cv2
import os
import dicom

from projects.drutils import data
from projects.drutils import parser
from projects.drutils import fileio
from projects.mammo_seg import inbreast

def calc_dog(img, sigma, k=1.6):
    """Calculate difference of gaussian

    Args:
        img:
        sigma:
        k:

    Returns:

    """
    s1 = filters.gaussian(img, k * sigma)
    s2 = filters.gaussian(img, sigma)
    dog = s1 - s2
    return dog


def get_binary_mask(img, disk_size, option='erode'):
    """Get a binary mask based for input image

    Args:
        img:
        disk_size:
        option: currently supported option
            'erode'

    Returns:
        binary_mask:
    """
    if disk_size > 0:
        th = filters.threshold_otsu(img)
        binary_mask = (img > th)
        selem = morphology.disk(disk_size)
        if option == 'erode':
            binary_mask = morphology.binary_erosion(binary_mask, selem=selem)
        elif option == 'open':
            binary_mask = morphology.binary_opening(binary_mask, selem=selem)
        else:
            raise ValueError('Unsupported option {}'.format(option))
    else:
        binary_mask = None
    return binary_mask


class LogBlobDetector(object):
    """Blob detector using LoG (Laplacian of Gaussian)"""
    def __init__(self, img, disk_size=10, **kwargs):
        """Constructor

        Args:
            img:
            **kwargs: example kwargs include
                max_sigma=25, min_sigma=2, num_sigma=3, threshold=0.1
        """
        self.img = img
        self.disk_size = disk_size
        self.max_sigma = kwargs['max_sigma']
        self.min_sigma = kwargs['min_sigma']
        self.num_sigma = kwargs['num_sigma']
        self.threshold = kwargs['threshold']

    def detect(self):
        """Detect blobs in image

        Returns:
            blobs_log_filtered: a list of tuples (x, y, r) that lies within the mask

        """
        blobs_log = feature.blob_log(self.img,
                                     max_sigma=self.max_sigma,
                                     min_sigma=self.min_sigma,
                                     num_sigma=self.num_sigma,
                                     threshold=self.threshold)
        binary_mask = get_binary_mask(self.img, self.disk_size, option='erode')
        if binary_mask is not None:
            blobs_log_filtered = []
            for blob in blobs_log:
                y, x, r = blob
                if binary_mask[int(y), int(x)]:
                    blobs_log_filtered.append((y, x, r))
        else:
            blobs_log_filtered = blobs_log
        return blobs_log_filtered


def detector_batch_deploy(LogBlobDetector, inbreast_name_list):
    """Batch deploy blob detector

    Args:
        LogBlobDetector:
        inbreast_name_list:

    Returns:

    """
    inbreast_dict = inbreast.generate_inbreast_dict()
    for name in inbreast_name_list:
        print(name)
        # get paths
        try:
            mask_path = \
            list(glob.glob(os.path.join(inbreast_dict['basedir'], 'AllMASK_level2', '_combined', '{}*'.format(name))))[0]
            image_path = list(glob.glob(os.path.join(inbreast_dict['basedir'], 'AllPNG', '{}*'.format(name))))[0]
            stack_path = list(glob.glob(os.path.join(inbreast_dict['basedir'], 'stack', '{}*'.format(name))))[0]
        except:
            print('not found {}'.format(name))
            continue

        # read images
        img = plt.imread(image_path, -1)
        img_shape = img.shape
        img_mask = plt.imread(mask_path, -1)
        img_stack = plt.imread(stack_path, -1)
        img_overlay = (img_stack[:, img_stack.shape[1] // 2:])

        # get eroded binary mask
        det = LogBlobDetector(img, max_sigma=25, min_sigma=2, num_sigma=3, threshold=0.1, disk_size=50)
        blobs_log = det.detect()

        canvas = img_overlay.copy()
        for y, x, r in blobs_log:
            cv2.circle(canvas, (int(x), int(y)), int(r + 5), color=(102, 255, 0), thickness=2)
        # stack image side-by-side for comparison
        img_log = np.hstack([canvas, np.dstack([img] * 3)])
        plt.imsave(os.path.join(inbreast_dict['basedir'], 'log', '{}_log_th0.1.png'.format(name)), img_log)


if __name__ == "__main__":
    inbreast_dict = inbreast.generate_inbreast_dict()
    df = pd.read_csv(inbreast_dict['csv_path'])

    # loop over all files and save to file
    # name_list = df[~ (df['Findings'].str.contains('normal'))]['File Name'][:1]
    name_list = df['File Name'].tolist()[:2]
    print(name_list)
    detector_batch_deploy(LogBlobDetector, name_list)
