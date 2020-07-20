"""This file contains utility functions used for file io"""
import cv2
import pandas as pd
import os
import json
import errno
import shutil
import numpy as np
import glob2
from PIL import Image
import nibabel as nib
import logging
from matplotlib import pylab as plt

def maybe_make_new_dir(new_directory):
    """ Make directory if it does not already exist """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        logging.debug('Making new directory {}'.format(new_directory))


def copy_tree(from_folder, to_folder, **kwargs):
    """Recursively copies from one folder to another

    This function is a wrapper for distutils.dir_util.copy_tree() but ignores time and mode copy

    Often time and mode are not important but sometimes causes permission error. This wrapper
    by default just copies but ignore such error.

    Args:
        from_folder:
        to_folder:

    Returns:

    """
    from distutils.dir_util import copy_tree
    preserve_times = kwargs.pop('preserve_times', 0)
    preserve_mode = kwargs.pop('preserve_mode', 0)
    update = kwargs.pop('update', 0)
    copy_tree(from_folder, to_folder,
              preserve_times=preserve_times,
              preserve_mode=preserve_mode,
              update=update,
              **kwargs)


def overwrite_guard(output_dir):
    """If the output_dir folder already exists and is non-empty, throw an error

    Args:
        output_dir:

    Returns:
        None
    """
    #
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        msg = 'Output folder is not empty: {}\nDo you still want to proceed?'.format(output_dir)
        shall = input("%s (y/N) " % msg).lower() == 'y'
        if not shall:
            raise FileExistsError('Please empty {} and proceed.'.format(output_dir))
    else:
        maybe_make_new_dir(output_dir)


def load_image_to_array(path, dtype=np.float32, mode='unchanged'):
    """Load image to array"""
    try:
        if mode == 'rgb':
            # load as RGB
            image_array = np.array(cv2.imread(path, 1), dtype)
        elif mode == 'unchanged':
            image_array = np.array(cv2.imread(path, -1), dtype)
        elif mode == 'gray':
            image_array = np.array(cv2.imread(path, 0), dtype)
        else:
            raise ValueError('unsupported mode')
    except:
        if path.endswith('.nii'):
            image_array = nib.load(path).get_fdata().T
        else:
            raise IOError('Cannot open {}'.format(path))
    return image_array


def silentremove(filepath):
    """Remove a filename if it exists

    Args:
        filepath: path to the file to be removed

    Returns:

    """

    try:
        os.remove(filepath)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


def read_list_from_txt(txt_file, sep=',', field=None):
    """
    Get a list of filepath from a txt file
        Args:
            txt_file: file path to a txt, each line of it contains a file path.
                If there are more than one comma separated field, get the field_idx'th field
            field: optional, index of field to extract, if -1 then read the whole line
        Return:
            file_path_list: a list of file paths
    """
    with open(txt_file, 'r') as f_in:
        lines = f_in.readlines()
    if field is None:
        file_path_list = [line.strip() for line in lines]
    else:
        file_path_list = [np.array(line.strip().split(sep))[field].tolist() for line in lines]
    return file_path_list


def get_list_of_filepath_from_txt(txt_file, field=0):
    """
    Get a list of filepath from a txt file
    """
    return read_list_from_txt(txt_file, field)


def get_list_of_labels_from_txt(txt_file, field=1):
    """
    Get a list of filepath from a txt file
    """
    return read_list_from_txt(txt_file, field)


def dump_list_to_file(filename_list, txt_file, name='', label_list=[]):
    """
    Write a list to file, with
        Args:
            filename_list: a list of filenames
            txt_file: file to a text file to write list to
            name: optional, name of the list
            label_list: optional, list of labels corresponding to filename_list
        Return:
            None
    """
    print('Writing list {} of length {} to txt file {}'.format(name, len(filename_list), txt_file))
    with open(txt_file, 'w', encoding='utf-8') as f_out:
        if label_list:
            assert len(filename_list) == len(label_list), 'filename_list and label_list must have the same length!'
            for filename, label in zip(filename_list, label_list):
                f_out.write('{},{}\n'.format(filename, label))
        else:
            for filename in filename_list:
                f_out.write('{}\n'.format(filename))


def substitute_string_in_txt(old_string, new_string, filepath):
    """
    Substitute old_string with new_string in text file filepath
    """
    backup_filepath = filepath + '.tmp'
    shutil.copyfile(filepath, backup_filepath)
    new_filepath = '/tmp/tmp.txt'
    with open(new_filepath, 'w') as f_out:
        with open(backup_filepath, 'r') as f_in:
            for line in f_in:
                newline = line.replace(old_string, new_string)
                f_out.write(newline)
    shutil.copyfile(new_filepath, filepath)


def remove_leading_char_a(filepath):
    """Remove the leading character `a` in the image file name to match original nih dataset"""
    file_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)[1:]
    new_filepath = os.path.join(file_dir, basename)
    return new_filepath


def list_file_in_directory(directory, ext='', path_level='noext', depth=1):
    """
    List files in a directory.
    Args:
        directory:
        ext: extension (suffix) to search directory for
        path_level: optional. Default to 'noext'. It can take the following values:
            'full': full path
            'partial': partial path starting from directory
            'basename': basename
            'noext': basename without extension
        depth: can be 1 or -1 (recursive).

    Returns:
        A list of filenames
    """
    # make sure there is one and only one separator char at the end of directory
    directory = directory.rstrip(os.path.sep) + os.path.sep
    if depth == 1:
        filepath_list = glob2.glob(os.path.join(directory, '*' + ext))
    elif depth == -1:
        filepath_list = glob2.glob(os.path.join(directory, '**', '*' + ext))
    else:
        raise ValueError('`depth` can only be 1 or -1 (recursive)!')
    filepath_list = [filepath for filepath in filepath_list if os.path.isfile(filepath)]
    if path_level == 'full':
        return filepath_list
    if path_level == 'partial':
        filepath_list = [filepath.replace(directory, '') for filepath in filepath_list]
    if path_level == 'basename':
        filepath_list = [os.path.basename(filepath) for filepath in filepath_list]
    if path_level == 'noext':
        filepath_list = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepath_list]
    return filepath_list


def get_ignore_file_list(raw_dir, cleaned_dir, ext=''):
    """
    Get a list of files that are in raw_dir but not in cleaned_dir
    Args:
        raw_dir:
        cleaned_dir:

    Returns:
        list_to_ignore: a list of file names to ignore
    """

    raw_filelist = list_file_in_directory(raw_dir, ext=ext, path_level='basename')
    cleaned_filelist = list_file_in_directory(cleaned_dir, ext=ext, path_level='basename')
    print(len(raw_filelist))
    print(len(cleaned_filelist))

    list_to_ignore = list(set(raw_filelist) - set(cleaned_filelist))
    return list_to_ignore


def read_image_from_path(image_path, channels=3):
    """Read image from a file

    Args:
        image_path: path to the image file
        channels: can be 1 or 3. If 1, then the shape of image_array is (height, width). If 3,
            the shape of image_array is (height, width, 3)

    Returns:
        image_array
    """
    assert channels in [1, 3]
    image_array = plt.imread(image_path, -1)
    if channels == 3:
        if len(image_array.shape) == 2:
            image_array = np.dstack([image_array] * 3)
        assert len(image_array.shape) == 3 and image_array.shape[-1] == 3
    if channels == 1:
        assert len(image_array.shape) == 2
    return image_array


def filter_list(input_list, key_fn=(lambda x: x), filter_keys=None):
    """Filter a list with a list of keys

    Args:
        input_list: list to be filtered
        key_fn: a function to generate keys from elements in the list
        filter_keys: keys to intersect with

    Returns:
        filtered_input_list: filtered list
    """
    if filter_keys is None:
        filtered_input_list = input_list
    else:
        input_dict = {key_fn(x): x for x in input_list}
        keys = set(input_dict.keys()) & set(filter_keys)
        keys = sorted(list(keys))
        filtered_input_list = [input_dict[key] for key in keys]
    return filtered_input_list


def load_json(json_path):
    with open(json_path, 'r') as f_in:
        data_dict = json.load(f_in)
    return data_dict


def write_json(data_dict, json_path):
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)