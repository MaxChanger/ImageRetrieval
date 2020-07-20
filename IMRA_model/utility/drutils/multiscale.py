"""This file includes utility functions and classes for multiscale patch crop"""
import tensorflow as tf
import cv2
import re
import shutil
import json
import glob2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage import measure
from tqdm import tqdm

from projects.drutils import fileio
from projects.drutils import augmentation

plt.rcParams['image.cmap'] = 'gray'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def generate_maskinfo(np_mask):
    """Return the mask corner and center coordinates

    Args:
        np_mask(numpy.array): the mask image, should be binary {0, 1}

    Returns:
        corners(list of float): [xmin(axis=0), ymin, xmax, ymax]
        centers(list of float): [centerx, centery]
    """
    # NB. The order of numpy is always in (y, x) or (row, col)
    y, x = np.where(np_mask == np.max(np_mask))
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    corners = [xmin, ymin, xmax, ymax]
    centers = [int((xmin + xmax) / 2), int((ymin + ymax) / 2)]
    return corners, centers


def get_lesion_size_ratio(corners, patch_shape):
    """Compute bbox to patch size ratio

    Args:
        corner_resize: tightest bbox coord of lesion (xmin, ymin, xmax, ymax)
        patch_shape: in the order of (y_size, x_size)

    Returns:
        lesion_size_ratio: sqrt(lesion_area / patch_area)
    """
    xmin, ymin, xmax, ymax = corners
    h, w = patch_shape
    lesion_size_ratio = ((xmax - xmin) * (ymax - ymin) / (h * w)) ** 0.5
    return lesion_size_ratio


def generate_rotate_list(rotations_per_axis=6, max_degree=30):
    """generate a list of degrees randomly

    Args:
        rotations_per_axis(integer): the number of degrees
        max_degree(float): the max degree will rotate

    Returns:
        degrees(list of float):  from -max_degree to +max_degree
    """
    # oversample by 20 times and select #rotation_per_axis numbers
    degrees = []
    if rotations_per_axis >= 1:
        degrees.extend(np.random.choice(
            np.arange(-max_degree, max_degree, max_degree / (20 * rotations_per_axis)), size=rotations_per_axis))
    return degrees


def generate_negative_sample(image_path, label_path, patch_size, neg_imagedir, isrotate=False,
                             ignore_padding=0,
                             n_patches=20,
                             key='',
                             nonezero_threshold=0.5,
                             scale=1.0,
                             resize_jitter_list=[0.75, 1.25],
                             max_trial_per_patch=5):
    """
    Generate the negative sample, random choose 100 points, to see if the result meet the demand
    Args:
        image_path(str)
        label_path(str): if empty, then use an all zero mask
        patch_size(int)
        neg_imagedir(str)

    Returns:
        None
    """
    assert image_path
    image = cv2.imread(image_path, -1)
    if label_path:
        label = cv2.imread(label_path, -1)
    else:
        print('Use all zero mask!')
        label = np.zeros_like(image, dtype=np.uint8)
    target_size = np.array([patch_size * 3, patch_size * 2])
    max_trial = n_patches * max_trial_per_patch  # for each patch try up to max_trial_per_patch times
    i = 0
    trial = 0
    max_nonzero_ratio = 0

    while trial <= max_trial and i < n_patches:
        trial += 1
        resize_ratio_lower, resize_ratio_upper = resize_jitter_list
        resize_jitter = np.random.uniform(resize_ratio_lower, resize_ratio_upper)
        image_resize = augmentation.resize(image, scale=resize_jitter*scale)
        label_resize = augmentation.resize(label, scale=resize_jitter*scale)
        image_resize_shape = np.asarray(image_resize.shape)
        if np.any(image_resize_shape < target_size):
            target_size = np.maximum(target_size, image_resize_shape)
            image_pad = augmentation.center_pad(image_resize, target_size)
            label_pad = augmentation.center_pad(label_resize, target_size)
        # Generate rotation angle randomly
        if isrotate:
            degree = generate_rotate_list(rotations_per_axis=1, max_degree=180)
            M = cv2.getRotationMatrix2D((image_pad.shape[0]/2, image_pad.shape[1]/2), degree[0], 1) # the rotation center must be tuple
            image_rotate = cv2.warpAffine(image_pad, M, (image_pad.shape[1], image_pad.shape[0]))
            label_rotate = cv2.warpAffine(label_pad, M, image_pad.shape)
            image_aug = image_rotate
            label_aug = label_rotate
        else:
            image_aug = image_pad
            label_aug = label_pad
        y = random.randint(patch_size / 2, image_aug.shape[0] - patch_size / 2)
        x = random.randint(patch_size / 2, image_aug.shape[1] - patch_size / 2)
        label_patch = label_aug[int(y - patch_size / 2): int(y + patch_size / 2),
                      int(x - patch_size / 2): int(x + patch_size / 2)]
        image_patch = image_aug[int(y - patch_size / 2): int(y + patch_size / 2),
                      int(x - patch_size / 2): int(x + patch_size / 2)]
        central_label_patch = label_patch[ignore_padding:-ignore_padding, ignore_padding:-ignore_padding]
        central_image_patch = image_patch[ignore_padding:-ignore_padding, ignore_padding:-ignore_padding]
        nonzero_ratio = np.count_nonzero(central_image_patch) / central_image_patch.size
        if not central_label_patch.any():
            max_nonzero_ratio = max(max_nonzero_ratio, nonzero_ratio)
            if nonzero_ratio >= nonezero_threshold:
                print('============', nonzero_ratio)
                i += 1
                neg_patch = image_patch
                neg_path = os.path.join(neg_imagedir, key, "{}_neg{:03d}_scale{:.2f}.png".format(key, i, scale))
                neg_label_path = os.path.join(neg_imagedir, key, "{}_neg{:03d}_scale{:.2f}_mask.png".format(key, i, scale))
                fileio.maybe_make_new_dir(os.path.dirname(neg_path))
                fileio.maybe_make_new_dir(os.path.dirname(neg_label_path))
                if neg_patch.shape == (patch_size, patch_size) and label_patch.shape == (patch_size, patch_size):
                    cv2.imwrite(neg_path, neg_patch)
                    cv2.imwrite(neg_label_path, label_patch)
                else:
                    continue
    print('max_nonzero_ratio', max_nonzero_ratio)


def stratefied_sampling_neg_and_pos(positive_patch_search_path,
                                    negative_patch_search_path,
                                    strata_regex_pattern,
                                    positive_dir,
                                    negative_dir,
                                    output_dir,
                                    max_ratio=2,
                                    seed=42):
    """Sample from positive and negative

    Args:
        positive_patch_search_path:
        negative_patch_search_path:
        strata_regex_pattern:
        positive_dir:
        negative_dir:
        output_path:
        max_ratio:
        seed: random seed for shuffling

    Returns:
        None
    """
    positive_files = glob2.glob(positive_patch_search_path)
    positive_files = [file for file in positive_files if re.search(strata_regex_pattern, file)]
    negative_files = glob2.glob(negative_patch_search_path)
    negative_files = [file for file in negative_files if re.search(strata_regex_pattern, file)]
    n_pos = len(positive_files)
    n_neg = len(negative_files)
    # if too many negatives, truncate at max_ratio
    if n_neg > n_pos * max_ratio:
        print('Truncate from {} to {} files'.format(n_neg, n_pos * max_ratio))
        negative_files = sorted(negative_files)
        np.random.seed(seed)
        np.random.shuffle(negative_files)
        negative_files = negative_files[:(n_pos * max_ratio)]
    # copy files
    for source_file in tqdm(positive_files):
        new_file = source_file.replace(positive_dir, output_dir)
        # print('{} --> {}'.format(source_file, new_file))
        fileio.maybe_make_new_dir(os.path.dirname(new_file))
        shutil.copyfile(source_file, new_file)
        # copy images
        shutil.copyfile(source_file.replace('_mask', ''), new_file.replace('_mask', ''))
    for source_file in tqdm(negative_files):
        new_file = source_file.replace(negative_dir, output_dir)
        # print('{} --> {}'.format(source_file, new_file))
        fileio.maybe_make_new_dir(os.path.dirname(new_file))
        shutil.copyfile(source_file, new_file)
        # copy images
        shutil.copyfile(source_file.replace('_mask', ''), new_file.replace('_mask', ''))


def affine_crop(image_array, crop_center, patch_shape, rotate_angle=0, mode='constant'):
    """Core function for rotation and crop patch

    Args:
        image_array(np.array): The original image
        crop_center(tuple): The center coordinate to crop (x,y)
        patch_shape(tuple): The final patch size, tuples of int (width, height)
        rotate_angle(float or int): rotation angle in degree unit
        mode: np.pad mode, can be `constant` or `reflect`

    Returns:
        np array, cropped patch array
    """
    if mode == 'reflect':
        x_center, y_center = crop_center
        w_patch, h_patch = patch_shape
        h_image, w_image = np.array(image_array.shape)
        xpad1 = -min(0, x_center - w_patch//2)
        xpad2 = max(0, x_center + (w_patch + 1)//2 - w_image)
        ypad1 = -min(0, y_center - h_patch//2)
        ypad2 = max(0, y_center + (h_patch + 1)//2 - h_image)
        image_array = np.pad(image_array, ((ypad1, ypad2), (xpad1, xpad2)), mode=mode)
        crop_center = np.array(crop_center) + np.array((ypad1, xpad1))

    radian = rotate_angle * np.pi / 180
    rot_mat = np.asarray([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]])
    # if dst_center is not int, it will run into unfaithful cropping when patch_size is odd number
    dst_center = (np.asarray(patch_shape).reshape(-1, 1) / 2).astype(np.int)
    trans_mat = dst_center - np.matmul(rot_mat, np.asarray(crop_center).reshape(2,1))
    dst_point = np.asarray([[0,0], [patch_shape[0], 0], [patch_shape[0], patch_shape[1]], [0, patch_shape[1]]]).T
    src_point = np.matmul(np.linalg.inv(rot_mat), (dst_point - trans_mat)).T
    M = cv2.getPerspectiveTransform(src_point.astype(np.float32), dst_point.T.astype(np.float32))
    patch_array = cv2.warpPerspective(image_array, M, patch_shape)
    return patch_array


def translate(crop_center, translation):
    """Translate the cropping center

    Args:
        crop_center: np array or list (x, y)
        translation: np array with the same shape as crop_center (x and y can be different)

    Return:
        np array: translated crop_center
    """
    crop_center = np.asarray(crop_center) + translation
    return crop_center


def get_crop_center_and_size_from_bbox(bbox):
    """Return crop center and size from bbox quadruple

    Note that the center and size are in the order of (x, y)

    Args:
        bbox:

    Returns:

    """
    ymin, xmin, ymax, xmax = bbox
    crop_center = [int((xmin + xmax) / 2), int((ymin + ymax) / 2)]
    crop_size = [xmax - xmin, ymax - ymin]
    return crop_center, crop_size


class BaseROICropper2D(object):
    """Base class for cropping ROIs from a single image

    Args:
        image_path:
        output_dir:
        patch_size:
        mask_path:
    """

    def __init__(self, image_path, output_dir, patch_size=512, mask_path=None):
        self.image_path = image_path
        self.output_dir = output_dir
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.name = os.path.basename(image_path).split(".")[0].split("_")[0]

    def load_image_arrays(self):
        image_array = cv2.imread(self.image_path, -1)
        if self.mask_path is not None:
            mask_array = cv2.imread(self.mask_path, -1)
        else:
            mask_array = None
        return image_array, mask_array

    @staticmethod
    def crop_once(image_array, crop_center, patch_shape, scale=1.0, aug_param_dict={}):
        """Crop image_array based on center, patch_shape and augmentation parameters

        Args:
            image_array:
            crop_center:
            patch_shape: shape of cropped patches in the orignal iamge. If None, infer
                from scale by (patch_size / scale, patch_size / scale)
            scale: the smaller the scale is, the larger the original patch is
            aug_param_dict:

        Returns:
            np array, cropped image array
        """
        # If aug_param_dict is empty, default to no augmentation
        rotate_angle = aug_param_dict.get('rotate_angle', 0)
        translation = aug_param_dict.get('translation', 0)
        is_flip = aug_param_dict.get('is_flip', False)
        resize_jitter = aug_param_dict.get('resize_jitter', 1)
        scale *= resize_jitter
        translation = np.asarray(translation / scale).astype(np.int)
        crop_center = translate(crop_center, translation)
        patch_shape = tuple(patch_shape) # this is required by cv2.warpPerspective
        patch_image_array = affine_crop(image_array, crop_center, patch_shape, rotate_angle)
        if is_flip:
            patch_image_array = np.fliplr(patch_image_array)
        return patch_image_array


class DetectedROICropper2D(BaseROICropper2D):
    """Crop ROIs from a single image according to a list of detection bboxes and corresponding scales

    Args:
        image_path:
        output_dir:
        bbox_dict_path_list: a list of path to bbox_dict json files. The bbox_dict has (name, info_dict) pair, and
            info_dict has at least the following fields:
                pred_bbox_list:
                pred_box_correct:
                gt_bbox_list:
                gt_box_covered:
        scale_list:
        patch_size:
        mask_path:
        crop_by_bbox: if True, use detection bbox as cropping criterion, otherwise use the scale of activation
        bbox_dilation: ratio to dilate bbox for cropping
        crop_mode: could be one of the following
            'square': crop a square with the side of max(patch_shape) centered at the bbox center
            'bbox': crop according to the bbox (the final patch may not be square)
    """
    def __init__(self, image_path, output_dir, bbox_dict_path_list, scale_list,
                 patch_size=512, mask_path=None, suffix='.png',
                 crop_by_bbox_size=False, do_resize=True,
                 bbox_dilation_ratio=1.0, bbox_dilation_size=0,
                 crop_mode='square'):
        super(DetectedROICropper2D, self).__init__(image_path=image_path, output_dir=output_dir,
                                                   patch_size=patch_size, mask_path=mask_path)
        self.bbox_dict_path_list = bbox_dict_path_list
        self.scale_list = scale_list
        self.suffix = suffix
        self.crop_by_bbox_size = crop_by_bbox_size
        self.bbox_dilation_ratio = bbox_dilation_ratio
        self.bbox_dilation_size = bbox_dilation_size
        self.crop_mode = crop_mode
        self.do_resize = do_resize
        assert self.crop_mode in ['square', 'bbox']

    def load_bboxes(self, bbox_dict_path):
        # load bbox_list for an image from bbox_dict_path
        with open(bbox_dict_path, 'r') as f_in:
            bbox_dict = json.load(f_in)
        if self.name in bbox_dict:
            bbox_list = bbox_dict[self.name]['pred_bbox_list']
            bbox_iscorrect_list = bbox_dict[self.name]['pred_box_correct']
        else:
            bbox_list = []
            bbox_iscorrect_list = []
        return bbox_list, bbox_iscorrect_list

    def write_arrays(self, image_sample, output_dir, name, scale, bbox_idx, iscorrect=False, label_sample=None, suffix='.png'):
        tp_or_fp_string = 'TP' if iscorrect else 'FP'
        if not os.path.isdir(os.path.join(output_dir, name)):
            os.mkdir(os.path.join(output_dir, name))
        out_imagepath = os.path.join(
            output_dir, name, "{}_{:03d}_{}_scale{:.2f}{}".format(name, bbox_idx, tp_or_fp_string, scale, suffix))
        cv2.imwrite(out_imagepath, image_sample)
        if label_sample is not None:
            out_labelpath = os.path.join(
                output_dir, name, "{}_{:03d}_{}_scale{:.2f}{}".format(name, bbox_idx, tp_or_fp_string, scale, suffix))
            cv2.imwrite(out_labelpath, label_sample)

    def crop_by_bbox_list(self, image_array, bbox_list, bbox_iscorrect_list, scale):
        for bbox_idx, (bbox, iscorrect) in enumerate(zip(bbox_list, bbox_iscorrect_list)):
            crop_center, crop_size = get_crop_center_and_size_from_bbox(bbox)
            if self.crop_by_bbox_size:
                logging.debug('raw crop_size {}'.format(crop_size))
                crop_size = np.array(crop_size) * self.bbox_dilation_ratio + 2 * self.bbox_dilation_size
                logging.debug('adjusted crop_size {}'.format(crop_size))
                if self.crop_mode == 'square':
                    patch_shape = np.array([max(crop_size), max(crop_size)]).astype(int)
                elif self.crop_mode == 'bbox':
                    patch_shape = crop_size.astype(int)
                patch_image_array = self.crop_once(image_array, crop_center, patch_shape=patch_shape, scale=scale)
                if self.do_resize:
                    patch_image_array = augmentation.resize(patch_image_array,
                                                            dst_shape=(self.patch_size, self.patch_size))
            else:
                patch_shape = np.asarray([self.patch_size / scale, self.patch_size / scale]).astype(np.int)
                patch_image_array = self.crop_once(image_array, crop_center, patch_shape=patch_shape, scale=scale)
            assert patch_image_array.shape == (self.patch_size, self.patch_size)
            self.write_arrays(image_sample=patch_image_array, label_sample=None,
                              output_dir=self.output_dir,
                              name=self.name,  scale=scale,
                              bbox_idx=bbox_idx, iscorrect=iscorrect, suffix=self.suffix)

    def deploy(self):
        image_array, mask_array = self.load_image_arrays()
        for bbox_dict_path, scale in zip(self.bbox_dict_path_list, self.scale_list):
            bbox_list, bbox_iscorrect_list = self.load_bboxes(bbox_dict_path)
            self.crop_by_bbox_list(image_array, bbox_list, bbox_iscorrect_list, scale)


class AugmentedROICropper2D(BaseROICropper2D):
    """Crop augmented ROIs from a single image (and corresponding mask)

    Total images = nb_scales * 1 *(1+nb_rotations)*nb_labels

    Args:
        image_path: str, the path to image
        mask_path: str, the path to mask
        output_dir: str, the output directory for both images and labels
        patch_size: int, size of final patch (square)
        max_degree: float: the maximum degree of rotation
        max_translation: int, the maximum value of translation
        upscales: list of float, if the mass is too small, resize the image with upscales
        downscales: list of float, if the mass is too big, resize the image with downscales
        size_list: list of two float, lower and upper bound of the ratio of mass_size / patch_size
        resize_jitter_list: list of two float, lower and upper bound of randomly resizing factor
        do_flip: bool, if true, perform random horizontal flip
    """

    def __init__(self, image_path, mask_path, output_dir,
                 patch_size=512, n_patches=1,
                 max_degree=0, max_translation=0,
                 upscales=(2.,), downscales=(0.5, 0.25, 0.125),
                 size_range=(0.25, 0.5),
                 resize_jitter_range=(1., 1.),
                 do_flip=False):
        super(AugmentedROICropper2D, self).__init__(image_path=image_path, output_dir=output_dir,
                                              patch_size=patch_size, mask_path=mask_path)
        self.do_flip = do_flip
        self.max_degree = max_degree
        self.max_translation = max_translation
        self.resize_jitter_range = resize_jitter_range
        self.n_patches = n_patches
        self.n_lesions = 0
        self.upscales = upscales
        self.downscales = downscales
        self.size_range = size_range

    def sample_augmentation_param(self):
        """Sample degree, translation, resize_jitter

        From self.max_degree and self.max_translation and self.resize_jitter_range
        """
        translation = np.random.randint(-self.max_translation // 2, self.max_translation // 2 + 1, size=2)
        degree = random.uniform(-self.max_degree, self.max_degree)
        resize_ratio_lower, resize_ratio_upper = self.resize_jitter_range
        resize_jitter = random.uniform(resize_ratio_lower, resize_ratio_upper)
        is_flip = self.do_flip and (random.uniform(0,1) > 0.5)
        aug_param_dict = {}
        aug_param_dict['rotate_angle'] = degree
        aug_param_dict['translation'] = translation
        aug_param_dict['resize_jitter'] = resize_jitter
        aug_param_dict['is_flip'] = is_flip
        return aug_param_dict

    def get_lesion_masks_and_scales(self, mask_array):
        """Determine the scales to crop patches based on the lesion size

        Args:
            mask_array: the mask array

        Return:
            tuple of list: a list of lesion masks, a list of scales for each lesion in the mask
        """
        size_lower, size_upper = self.size_range
        labeled_mask_array = measure.label(mask_array, connectivity=2)
        # the number of mass the image contains
        self.n_lesions = np.max(labeled_mask_array)
        scales = []
        lesion_masks = []
        for i in range(1, self.n_lesions + 1):
            # generate a mask image that only contains one connected component
            cur_label = (labeled_mask_array == i).astype(np.uint8) * 255
            corner_resize, center_resize = generate_maskinfo(cur_label)
            # get lesion size
            edge_ratio = get_lesion_size_ratio(corner_resize, (self.patch_size, self.patch_size))
            # find scale
            total_scales = sorted([1] + self.upscales + self.downscales)
            scale = 0
            if edge_ratio * min(total_scales) > size_upper:
                scale = min(total_scales)
            elif edge_ratio * max(total_scales) < size_lower:
                scale = max(total_scales)
            else:
                for scale in total_scales:
                    if size_lower <= edge_ratio * scale <= size_upper:
                        break
            scales.append(scale)
            lesion_masks.append(cur_label)
        assert np.all(scales), 'Some scale is zero {}'.format(scales)
        return lesion_masks, scales

    def get_crop_center_and_size_from_mask(self, labeled_mask_array):
        """Get center and size of lesion from binary mask

        Args:
            labeled_mask_array: binary array with only one connected component in it

        Returns:
            crop_center, crop_size
        """
        corners, centers = generate_maskinfo(labeled_mask_array)
        xmin, ymin, xmax, ymax = corners
        crop_size = (xmax - xmin, ymax - ymin)
        crop_center = centers
        return crop_center, crop_size

    def write_arrays(self, image_sample, output_dir, name, lesion_idx, aug_idx, scale, label_sample=None):
        if not os.path.isdir(os.path.join(output_dir, name)):
            os.mkdir(os.path.join(output_dir, name))
        out_imagepath = os.path.join(
            output_dir, name, "{}_{:03d}_{:03d}_scale{:.2f}.png".format(name, lesion_idx + 1, aug_idx, scale))
        cv2.imwrite(out_imagepath, image_sample)
        if label_sample is not None:
            out_labelpath = os.path.join(
                output_dir, name, "{}_{:03d}_{:03d}_scale{:.2f}_mask.png".format(name, lesion_idx + 1, aug_idx, scale))
            cv2.imwrite(out_labelpath, label_sample)

    def crop_by_mask(self, image_array, mask_array):
        """Crop lesion patches according to each connected component in mask_array

        Args:
            image_array:
            mask_array:

        Returns:
            None
        """
        lesion_masks, scales = self.get_lesion_masks_and_scales(mask_array)
        for lesion_idx, (lesion_mask, scale) in enumerate(zip(lesion_masks, scales)):
            crop_center, crop_size = self.get_crop_center_and_size_from_mask(lesion_mask)
            for aug_idx in range(self.n_patches):
                aug_param_dict = self.sample_augmentation_param()
                patch_shape = np.asarray([self.patch_size / scale, self.patch_size / scale]).astype(np.int)
                patch_image_array = self.crop_once(image_array, crop_center, patch_shape=patch_shape,
                                                   scale=scale, aug_param_dict=aug_param_dict)
                patch_mask_array = self.crop_once(lesion_mask, crop_center, patch_shape=patch_shape,
                                                  scale=scale, aug_param_dict=aug_param_dict)
                # binarize upscaled mask
                patch_mask_array = (patch_mask_array > patch_mask_array.max() * 0.5).astype(np.int8) * 255
                self.write_arrays(image_sample=patch_image_array, label_sample=patch_mask_array,
                                 output_dir=self.output_dir,
                                 name=self.name, lesion_idx=lesion_idx, aug_idx=aug_idx, scale=scale)

    def deploy(self):
        image_array, mask_array = self.load_image_arrays()
        self.crop_by_mask(image_array, mask_array)


class AugmentedROICropper2D_v2(BaseROICropper2D):
    """Crop augmented ROIs from a single image (and corresponding calc mask).

    Crop center randomly sampled from another image (cluster mask).
    First concat image and mask into one multi-channel image, then crop as usual.
    Finally split channels into differnt images.

    Total images = nb_scales * 1 *(1+nb_rotations)*nb_labels

    Args:
        image_path: str, the path to image
        mask_path: str, the path to mask
        output_dir: str, the output directory for both images and labels
        patch_size: int, size of final patch (square)
        max_degree: float: the maximum degree of rotation
        max_translation: int, the maximum value of translation
        upscales: list of float, if the mass is too small, resize the image with upscales
        downscales: list of float, if the mass is too big, resize the image with downscales
        size_list: list of two float, lower and upper bound of the ratio of mass_size / patch_size
        resize_jitter_list: list of two float, lower and upper bound of randomly resizing factor
        do_flip: bool, if true, perform random horizontal flip
    """

    def __init__(self, image_path, mask_path, output_dir,
                 patch_size=512, n_patches=1,
                 max_degree=0, max_translation=0,
                 upscales=(), downscales=(),
                 size_range=(np.inf, -np.inf),
                 resize_jitter_range=(1., 1.),
                 do_flip=False):
        super(AugmentedROICropper2D_v2, self).__init__(image_path=image_path, output_dir=output_dir,
                                              patch_size=patch_size, mask_path=mask_path)
        self.do_flip = do_flip
        self.max_degree = max_degree
        self.max_translation = max_translation
        self.resize_jitter_range = resize_jitter_range
        self.n_patches = n_patches
        self.n_lesions = -1
        self.upscales = upscales
        self.downscales = downscales
        self.size_range = size_range

    def load_image_arrays(self):
        image_array = cv2.imread(self.image_path, -1)
        assert isinstance(self.mask_path, list) and (len(self.mask_path) == 2)
        calc_mask_path, cluster_mask_path = self.mask_path
        calc_mask_array = cv2.imread(calc_mask_path, -1)
        cluster_mask_array = cv2.imread(cluster_mask_path, -1)
        assert (len(calc_mask_array.shape) == 2)
        assert (len(cluster_mask_array.shape) == 2)
        if image_array.shape != calc_mask_array.shape:
            print('{} != {}'.format(image_array.shape, calc_mask_array.shape))
            calc_mask_array = augmentation.crop_or_pad(calc_mask_array, image_array.shape)
        image_array = np.dstack([image_array, calc_mask_array])
        mask_array = cluster_mask_array
        return image_array, mask_array

    def split_image_array(self, image_sample):
        assert (len(image_sample.shape) == 3)
        assert (image_sample.shape[-1] == 2)
        image_sample, label_sample = image_sample[..., 0], image_sample[..., 1]
        return image_sample, label_sample

    def sample_augmentation_param(self):
        """Sample degree, translation, resize_jitter

        From self.max_degree and self.max_translation and self.resize_jitter_range
        """
        translation = np.random.randint(-self.max_translation // 2, self.max_translation // 2 + 1, size=2)
        degree = random.uniform(-self.max_degree, self.max_degree)
        resize_ratio_lower, resize_ratio_upper = self.resize_jitter_range
        resize_jitter = random.uniform(resize_ratio_lower, resize_ratio_upper)
        is_flip = self.do_flip and (random.uniform(0,1) > 0.5)
        aug_param_dict = {}
        aug_param_dict['rotate_angle'] = degree
        aug_param_dict['translation'] = translation
        aug_param_dict['resize_jitter'] = resize_jitter
        aug_param_dict['is_flip'] = is_flip
        return aug_param_dict

    def get_random_crop_center_from_mask(self, mask_array):
        """Get center and size of lesion from binary mask

        Args:
            mask_array: binary array

        Returns:
            crop_center
        """
        y_list, x_list = np.where(mask_array > 0)
        xy_list = list(zip(x_list, y_list))
        random_idx = np.random.choice(len(xy_list))
        crop_center = xy_list[random_idx]
        return crop_center

    def write_arrays(self, image_sample, output_dir, name, lesion_idx, aug_idx, scale, label_sample=None):
        if not os.path.isdir(os.path.join(output_dir, name)):
            os.mkdir(os.path.join(output_dir, name))
        out_imagepath = os.path.join(
            output_dir, name, "{}_{:03d}_{:03d}_scale{:.2f}.png".format(name, lesion_idx + 1, aug_idx, scale))
        cv2.imwrite(out_imagepath, image_sample)
        if label_sample is not None:
            out_labelpath = os.path.join(
                output_dir, name, "{}_{:03d}_{:03d}_scale{:.2f}_mask.png".format(name, lesion_idx + 1, aug_idx, scale))
            cv2.imwrite(out_labelpath, label_sample)

    def crop_by_mask(self, image_array, mask_array):
        """Crop lesion patches according to each connected component in mask_array

        Args:
            image_array:
            mask_array:

        Returns:
            None
        """
        crop_center = self.get_random_crop_center_from_mask(mask_array)
        for aug_idx in range(self.n_patches):
            aug_param_dict = self.sample_augmentation_param()
            scale = 1.0
            patch_shape = np.asarray([self.patch_size / scale, self.patch_size / scale]).astype(np.int)
            patch_image_array = self.crop_once(image_array, crop_center, patch_shape=patch_shape,
                                               scale=scale, aug_param_dict=aug_param_dict)
            patch_image_array, patch_mask_array = self.split_image_array(patch_image_array)
            # binarize upscaled mask
            patch_mask_array = (patch_mask_array > patch_mask_array.max() * 0.5).astype(np.int8) * 255
            self.write_arrays(image_sample=patch_image_array, label_sample=patch_mask_array,
                             output_dir=self.output_dir,
                             name=self.name, lesion_idx=0, aug_idx=aug_idx, scale=scale)

    def deploy(self):
        image_array, mask_array = self.load_image_arrays()
        self.crop_by_mask(image_array, mask_array)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('task', 'pos_crop', 'task name, can be pos_crop or neg_crop')
    # logging.getLogger().setLevel(logging.DEBUG)

    def get_dict_from_search_path(image_search_path, label_search_path):
        image_pngs = glob2.glob(image_search_path)
        label_pngs = glob2.glob(label_search_path)
        key_fn = lambda x: os.path.basename(x).split('.')[0].split("_")[0]
        image_dict = {key_fn(path): path for path in image_pngs}
        label_dict = {key_fn(path): path for path in label_pngs}
        return image_dict, label_dict

    if FLAGS.task == 'crop_by_bbox':
        output_dir = '/data/log/mammo/detection_patches_bbox2/'
        fileio.maybe_make_new_dir(output_dir)
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/*png"
        mask_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask_etc/*png"
        # NB. Run batch evaluate to generate bbox_dict json first
        scale = 1.0
        if scale == 1.0:
            bbox_dict_path = '/data/log/mammo/detection_patches/bbox_dict_ziwei_scale1.0.json'
            pred_search_path = r'/data/log/mammo/mass_train/Mammo_20180502-16h50PM26/eval_model.cpkt-280000-ziwei-scale1.0/*png'
        elif scale == 0.5:
            bbox_dict_path = '/data/log/mammo/detection_patches/bbox_dict_ziwei_scale0.5.json'
            pred_search_path = r'/data/log/mammo/mass_train/Mammo_20180502-16h50PM26/eval_model.cpkt-70000-ziwei-scale0.5/*png'
        elif scale == 0.25:
            bbox_dict_path = '/data/log/mammo/detection_patches/bbox_dict_ziwei_scale0.25.json'
            pred_search_path = r'/data/log/mammo/mass_train/Mammo_20180502-16h50PM26/eval_model.cpkt-90000-ziwei-scale0.25/*png'
        # find intersection of two sets of keys
        image_dict, label_dict = get_dict_from_search_path(pred_search_path, mask_search_path)
        keys = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
        # generate cropped images, predictions and masks
        for image_search_path, suffix in zip([image_search_path, pred_search_path, mask_search_path],
                                             ['_img.png', '_pred.png', '_mask.png']):
            image_dict, label_dict = get_dict_from_search_path(image_search_path, mask_search_path)
            for key in tqdm(keys[:]):
            # for key in ['11740735-4']:
                label_path = label_dict[key]
                image_path = image_dict[key]
                DetectedROICropper2D(image_path, output_dir,
                                     patch_size=512,
                                     mask_path=None,
                                     bbox_dict_path_list=[bbox_dict_path],
                                     scale_list=[scale],
                                     suffix=suffix,
                                     crop_by_bbox_size=True,
                                     bbox_dilation_ratio=4.0, # 4 times the bbox size
                                     bbox_dilation_size=-400,
                                     crop_mode='square').deploy()
    elif FLAGS.task == 'crop_by_mask':
        output_dir = r"/data/log/mammo/patches_multiscale_mass_0406_ziwei_test"
        fileio.maybe_make_new_dir(output_dir)
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask_3456/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # find intersection of two sets of keys
        keys = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
        all_info_dict = []
        for key in tqdm(keys[:5]):
        # for key in ['11737954-1']:
            mask_path = label_dict[key]
            image_path = image_dict[key]
            # label_path = '/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask/11740735-4_combined.png'
            # image_path = '/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/11740735-4.dcm.png'
            output_dir = '/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/test/'
            AugmentedROICropper2D(image_path, mask_path, output_dir,
                                  patch_size=512, n_patches=10,
                                  max_degree=180, max_translation=100,
                                  upscales=[2], downscales=[1 / 2.0, 1 / 4.0, 1 / 8.0],
                                  size_range=[1 / 4, 1 / 2],
                                  resize_jitter_range=[0.75, 1.25],
                                  do_flip=True).deploy()
    elif FLAGS.task == 'pos_crop_ziwei':
        output_dir = r"/data/log/mammo/patches_multiscale_mass_0406_ziwei"
        fileio.maybe_make_new_dir(output_dir)
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask_3456/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # find intersection of two sets of keys
        keys = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
        all_info_dict = []
        for key in tqdm(keys[:]):
        # for key in ['11737954-1']:
            label_path = label_dict[key]
            image_path = image_dict[key]
            info_dict = augment_roi_2D(image_path, label_path, output_dir,
                                       patch_size=512, n_patches=10,
                                       max_degree=180, max_translation=100,
                                       upscales=[1.0], downscales=[1/2.0, 1/4.0, 1/8.0, 1/16.0],
                                       size_list=[100/512, 200/512],
                                       resize_jitter_list=[0.75, 1.25],
                                       do_flip=True,
                                       padding="constant",
                                       key=key)
            all_info_dict.append(info_dict.items())

    elif FLAGS.task == 'neg_crop':
        neg_imagedir = r"/data/log/mammo/patches_multiscale_mass_0406_ziwei_neg"
    elif FLAGS.task == 'neg_crop_ziwei':
        neg_imagedir = r"/data/log/mammo/patches_multiscale_mass_0406_ziwei_neg"
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask_etc/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # crop from negative images as well
        keys = sorted(list(set(image_dict.keys())))
        all_info_dict = []
        for key in tqdm(keys[:]):
            print(key)
            label_path = label_dict.get(key, '')
            image_path = image_dict.get(key, '')
            generate_negative_sample(image_path, label_path, isrotate=True,key=key,
                                     neg_imagedir=neg_imagedir,
                                     patch_size=512,
                                     ignore_padding=128, # generally keep it as patch_size / 4
                                     n_patches=10,
                                     scale=1.0,
                                     nonezero_threshold=0.75,
                                     resize_jitter_list=[0.75, 1.25])
    if FLAGS.task == 'pos_crop_inbreast':
        output_dir = r"/data/log/mammo/patches_multiscale_mass_0406_inbreast_pos_small"
        fileio.maybe_make_new_dir(output_dir)
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/INbreast/AllPNG_norm_6_6/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/INbreast/mass_mask/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # find intersection of two sets of keys
        keys = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
        all_info_dict = []
        for key in tqdm(keys[:]):
        # for key in ['11737954-1']:
            label_path = label_dict[key]
            image_path = image_dict[key]
            info_dict = augment_roi_2D(image_path, label_path, output_dir,
                                       patch_size=512, n_patches=10,
                                       max_degree=180, max_translation=100,
                                       upscales=[1.0], downscales=[1/2.0, 1/4.0, 1/8.0, 1/16.0],
                                       size_list=[50/512, 100/512],
                                       resize_jitter_list=[0.75, 1.25],
                                       do_flip=True,
                                       padding="constant",
                                       key=key)
            all_info_dict.append(info_dict.items())
    elif FLAGS.task == 'neg_crop_inbreast':
        neg_imagedir = r"/data/log/mammo/patches_multiscale_mass_0406_inbreast_neg_scale0.12"
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/INbreast/AllPNG_norm_6_6/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/INbreast/mass_mask/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # crop from negative images as well
        keys = sorted(list(set(image_dict.keys())))
        all_info_dict = []
        for key in tqdm(keys[:]):
            print(key)
            label_path = label_dict.get(key, '')
            image_path = image_dict.get(key, '')
            generate_negative_sample(image_path, label_path, key=key,
                                     neg_imagedir=neg_imagedir,
                                     patch_size=512,
                                     ignore_padding=128, # generally keep it as patch_size / 4
                                     n_patches=10,
                                     scale=0.125,
                                     nonezero_threshold=0.5, # 0.5 for scale [0.125], 0.75 for scales [0.25, 0.5 and 1.0]
                                     resize_jitter_list=[0.75, 1.25])
            #             all_info_dict.append(info_dict.items())

    elif FLAGS.task == 'split_ziwei':
        positive_patch_search_path = r'/data/log/mammo/patches_multiscale_mass_0406_ziwei_pos/**/*_mask.png'
        negative_patch_search_path = r'/data/log/mammo/patches_multiscale_mass_0406_ziwei_neg_scale1.00/**/*_mask.png'
        strata_regex_pattern = 'scale1.00'
        positive_dir = positive_patch_search_path.split('*')[0]
        negative_dir = negative_patch_search_path.split('*')[0]
        output_dir = r'/data/log/mammo/patches_multiscale_mass_0406_ziwei_stratefied/'
        stratefied_sampling_neg_and_pos(positive_patch_search_path=positive_patch_search_path,
                                        negative_patch_search_path=negative_patch_search_path,
                                        strata_regex_pattern=strata_regex_pattern,
                                        positive_dir=positive_dir,
                                        negative_dir=negative_dir,
                                        output_dir=output_dir)
    elif FLAGS.task == 'split_inbreast':
        positive_patch_search_path = r'/data/log/mammo/patches_multiscale_mass_0406_inbreast_pos_small/**/*_mask.png'
        negative_patch_search_path = r'/data/log/mammo/patches_multiscale_mass_0406_inbreast_neg_scale0.12/**/*_mask.png'
        strata_regex_pattern = 'scale0.12'
        positive_dir = positive_patch_search_path.split('*')[0]
        negative_dir = negative_patch_search_path.split('*')[0]
        output_dir = r'/data/log/mammo/patches_multiscale_mass_0406_inbreast_stratefied_small/'
        stratefied_sampling_neg_and_pos(positive_patch_search_path=positive_patch_search_path,
                                        negative_patch_search_path=negative_patch_search_path,
                                        strata_regex_pattern=strata_regex_pattern,
                                        positive_dir=positive_dir,
                                        negative_dir=negative_dir,
                                        output_dir=output_dir)

    elif FLAGS.task == 'calc_cluster_crop':
        output_dir = r"/data/log/mammo/calc_cluster_crop_affine_pos"
        fileio.maybe_make_new_dir(output_dir)
        image_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/calc_cluster/png/*png"
        label_search_path = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/calc_cluster/bootstrap_mask_cleaned/*png"
        image_dict, label_dict = get_dict_from_search_path(image_search_path, label_search_path)
        # find intersection of two sets of keys
        keys = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
        # print(keys, image_dict, label_dict)
        all_info_dict = []
        for key in tqdm(keys[:]):
        # for key in ['11737954-1']:
            print(key)
            # mask_path is [calc_mask_path, cluster_mask_path]
            cluster_mask_path = label_dict[key].replace('bootstrap_mask_cleaned', 'calc_cluster_all')
            calc_mask_path = label_dict[key]
            mask_path = [calc_mask_path, cluster_mask_path]
            image_path = image_dict[key]
            AugmentedROICropper2D_v2(image_path, mask_path, output_dir,
                                  patch_size=512, n_patches=10,
                                  max_degree=5, max_translation=5,
                                  upscales=[1.], downscales=[1.0],
                                  size_range=[np.inf, -np.inf],
                                  resize_jitter_range=[0.75, 1.25],
                                  do_flip=True).deploy()
