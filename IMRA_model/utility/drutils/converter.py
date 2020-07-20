"""
Conversion of different format
"""
import SimpleITK as sitk
import cv2
import functools
import glob2
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from projects.drutils import fileio
from projects.drutils import augmentation
from projects.drutils import data


# parse list from args
def csv_to_list(csv_string, type=str):
    return [type(item.strip()) for item in csv_string.strip().split(',') if item.strip()]


def scale_to_255(image_array):
    """Default normalizing functor to scale image to [0, 255]"""
    a_min = min(image_array.min(), 0)
    a_max = image_array.max()
    image_array = (image_array - a_min)/ (a_max - a_min) * 255
    return image_array


def dicom2png(dicom_filepath, png_filepath=None, normalize_functor=scale_to_255, dryrun=False):
    """Convert dicom image to png file

    Args:
        dicom_filepath:
        png_filepath:
        normalize_functor: normalizing function

    Returns:
        image_array: a numpy array containing the image
    """
    image_array = data.get_pixel_array_from_dicom_path(dicom_filepath, to_bit=-1)
    if normalize_functor:
        image_array = normalize_functor(image_array)
    if image_array.max() <= 1:
        image_array = (image_array * 255).astype(np.uint8)
    if not dryrun:
        fileio.maybe_make_new_dir(os.path.dirname(png_filepath))
        cv2.imwrite(png_filepath, image_array)
    return image_array


def dicom2png_batch(dicom_search_path, png_filepath_replacer, nfiles=None, normalize_functor=scale_to_255):
    """Convert png to nii in batch mode

    Args:
        dicom_search_path: can be a glob2 search pattern or directory
        png_filepath_replacer: a function to convert dicom filepath to png filepath
        nfiles: max number of files to convert
        normalize_functor: a function to normalize input images

    Returns:
        None
    """
    if os.path.isdir(dicom_search_path):
        file_list = glob2.glob(os.path.join(dicom_search_path, '**', '*dcm'))
    else:
        file_list = glob2.glob(dicom_search_path)
    if nfiles is not None:
        file_list = file_list[:nfiles]
    for dicom_filepath in tqdm(file_list):
        png_filepath = png_filepath_replacer(dicom_filepath)
        print('{} --> {}'.format(dicom_filepath, png_filepath))
        dicom2png(dicom_filepath, png_filepath, normalize_functor=normalize_functor)


def png2nii(png_filepath, nii_filepath):
    """Convert png to nii format

    Args:
        png_filepath:
        nii_filepath:

    Returns:
        None
    """
    image = sitk.ReadImage(png_filepath)
    # make parent directory otherwise sitk will not write files
    fileio.maybe_make_new_dir(os.path.dirname(nii_filepath))
    sitk.WriteImage(image, nii_filepath)
    # visualization
    # img_array = sitk.GetArrayFromImage(image)
    # plt.imshow(img_array)


def png2nii_batch(png_folder, nii_folder, nfiles=None):
    """Convert png to nii in batch mode

    Args:
        png_folder:
        nii_folder:

    Returns:
        None
    """
    assert os.path.isdir(png_folder), 'input is not a valid folder'
    file_list = glob2.glob(os.path.join(png_folder, '**', '*dcm'))
    if nfiles is not None:
        file_list = file_list[:nfiles]
    for i, png_filepath in enumerate(file_list):
        nii_filepath = png_filepath.replace(png_folder, nii_folder + os.sep)
        nii_filepath = nii_filepath.replace('.png', '.nii')
        print('{}: {} --> {}'.format(i, png_filepath, nii_filepath))
        png2nii(png_filepath, nii_filepath)


def sitk_2d_to_3d(sitk_image_2d, is_binary=False):
    """Convert 2d simple itk image to 3d

    Args:
        sitk_image_2d: a 2d sitk image

    Returns:
        sitk_image_3d: a 3d sitk image with depth padded to 1

    """
    png_array2D = sitk.GetArrayFromImage(sitk_image_2d)
    if is_binary:
        png_array2D = (png_array2D > 0).astype(np.uint8)
    print(np.unique(png_array2D))
    png_array3D = png_array2D.reshape((1,) + png_array2D.shape)
    sitk_image_3d = sitk.GetImageFromArray(png_array3D)
    return sitk_image_3d


def resize_png(png_file, target_h):
    """Resize png to target_h x target_h. Pad on the right by zero if non-square

    Args:
        png_file: input png file
        target_h: target new height

    Returns:

    """
    output_png_file = png_file + '_{}x{}.png'.format(target_h, target_h)

    image_array = plt.imread(png_file, -1)
    h, w = image_array.shape
    assert h >= w

    target_w = int(target_h / h * w)
    image_array_new = cv2.resize(image_array, (target_w, target_h), cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_h))
    canvas[:target_h, :target_w] = image_array_new
    plt.imshow(canvas)
    print(output_png_file)
    cv2.imwrite(output_png_file, canvas)


class ClaheConverter(object):
    """Batch apply CLAHE to all files in a search path

    Args:
        image_search_path:
        output_dir:
    """
    def __init__(self, image_search_path, output_dir):
        self.image_files = glob2.glob(image_search_path)
        self.output_dir = output_dir

    @staticmethod
    def apply_clahe(image_array, clipLimit=2.0, tileGridSize=(8,8)):
        """Apply Contrast Limited Adaptive Histogram Equalization

        Args:
            image_array:
            clipLimit:
            tileGridSize:

        Returns:
            image_array_clahe:
        """
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        image_array_clahe = clahe.apply(image_array)
        return image_array_clahe

    def deploy(self, clipLimit=2.0, tileGridSize=(8,8)):
        fileio.maybe_make_new_dir(self.output_dir)
        for image_file in tqdm(sorted(self.image_files)):
            image_array = plt.imread(image_file, -1)
            image_array = self.apply_clahe(image_array, clipLimit=clipLimit, tileGridSize=tileGridSize)
            output_file_path = os.path.join(self.output_dir, os.path.basename(image_file))
            cv2.imwrite(output_file_path, image_array)


class NormalizePathces(object):
    """Normalize patches

    Args:
        min_pct:
        max_pct:
        verbose: whether to output debuging message
        target_shape:

    Methods:
        process: process numpy array
        batch_process: with file io
    """

    def __init__(self, min_pct=0, max_pct=100, debug=False, target_shape=None):
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.debug = debug
        self.target_shape = target_shape

    def convert_to_gray(self, patch_array):
        return cv2.cvtColor(patch_array, cv2.COLOR_BGR2GRAY)

    def crop_or_pad(self, patch_array):
        if self.target_shape is not None:
            patch_array = augmentation.center_crop_or_pad(patch_array,
                                                          target_shape=self.target_shape)
        return patch_array

    def process(self, patch_array):
        if np.percentile(patch_array, 0) < 5:
            a_min = np.percentile(patch_array[patch_array > 5], 10)
        else:
            a_min = np.percentile(patch_array[patch_array > 5], self.min_pct)
        a_max = np.percentile(patch_array[patch_array > 5], self.max_pct)
        if self.debug:
            print('amin {} amx {}'.format(a_min, a_max))
        patch_array = np.clip(patch_array, a_min=a_min, a_max=a_max)
        return (((patch_array - a_min) / (a_max - a_min)) * 255).astype(np.uint8)

    def load_image(self, patch_path):
        return fileio.load_image_to_array(patch_path)

    def get_output_path(self, patch_path, path_converter_fn):
        return path_converter_fn(patch_path)

    def write(self, output_path, patch_array, write_rgb=False):
        fileio.maybe_make_new_dir(os.path.dirname(output_path))
        if write_rgb:
            patch_array = np.dstack([patch_array] * 3)
        cv2.imwrite(output_path, patch_array)

    def batch_process(self, input_search_path, path_converter_fn=None, dryrun=True, write_rgb=False):
        """

        Args:
            input_search_path:
            path_converter_fn:
            dryrun:
            write_rgb:

        Returns:

        """
        patch_paths = glob2.glob(input_search_path)
        for patch_path in tqdm(patch_paths[:]):
            patch_array = self.load_image(patch_path)
            patch_array = self.convert_to_gray(patch_array)
            patch_array = self.crop_or_pad(patch_array)
            patch_array = self.process(patch_array)
            output_path = self.get_output_path(patch_path, path_converter_fn)
            if dryrun:
                print('write to {}'.format(output_path))
            else:
                self.write(output_path, patch_array, write_rgb=write_rgb)


def binarize_mask(search_path, rename_fn=None):
    if rename_fn is None:
        rename_fn = lambda x: x.replace('mask.png', 'binary_mask.png')
    for filepath in tqdm(glob2.glob(search_path)):
        print(filepath)
        image_array = plt.imread(filepath, -1)
        binary_array = (image_array > 0).astype(np.uint8)
        new_filepath = rename_fn(filepath)
        cv2.imwrite(new_filepath, binary_array)


if __name__ == '__main__':
    binarize_mask(
        '/data/pliu/data/inbreast/calc_patches_ignore_single_point/train_mini/**/*mask.png',
        rename_fn=None
    )
