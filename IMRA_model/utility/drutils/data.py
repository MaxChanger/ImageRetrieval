"""This file contains utility functions used for numpy data manipulation"""
import json
import logging
try:
    import dicom
except:
    import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class NumpyEncoder(json.JSONEncoder):
    """This is a Encoder used to dump numpy arrays to json files.

    It also converts np.int64 (not python serializable) to python int

    Example:
        a = np.array([1, 2, 3])
        print(json.dumps({'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder))
    Output:
        {"aa": [2, [2, 3, 4], [1, 2, 3]], "bb": [2]}

    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float16):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def convert_to_unit8(pixel_array, from_bit=16, to_bit=8):
    """
    Convert dicom dataset to an uint8 numpy array

    Args:
        pixel_array: a numpy array
        from_bit: bit to convert from
        to_bit: bit to convert to
    Returns:
        pixel_array: a converted pixel_array
    """
    if from_bit == to_bit:
        return pixel_array
    # TODO: this is not exactly right. As 0-255 only has 2**8-1 scales
    pixel_array = pixel_array * (2 ** (to_bit - from_bit))
    if to_bit == 8:
        pixel_array = pixel_array.astype(np.uint8)
    else:
        raise ValueError('Unsupported bit type {}-bit!'.format(to_bit))
    return pixel_array


def get_new_dimensions(orig_shapes, min_dimension, max_dimension):
    """Get new dimensions based on the target shape limits

    The output size can be described by two cases:
        1. If the image can be rescaled so its minimum dimension is equal to the
             provided value without the other dimension exceeding max_dimension,
             then do so.
        2. Otherwise, resize so the largest dimension is equal to max_dimension.

    Args:
        orig_shapes:
        min_dimension:
        max_dimension:

    Returns:
        new_shapes: a tuple of new dimensions
    """

    min_target = min(orig_shapes)
    max_target = max(orig_shapes)
    if max_target * min_dimension / min_target < max_dimension:
        ratio = min_dimension / min_target
    else:
        ratio = max_dimension / max_target
    new_shapes = tuple(int(shape * ratio) for shape in orig_shapes)

    return new_shapes


def get_pixel_array_from_dicom_path(filepath, mismatch=1, to_bit=8, floor=None, ceiling=None):
    """
    Read image from dicom file and conver to numpy array.
    Args:
        filepath: dicom filepath
        mismatch: number of pixels to drop in pixel_array in case of a shape mismatch
        to_bit: bit to convert to, 8 and 16 supported. Return raw array if set to -1.
        floor: manually override bit conversion
        ceiling: manually override bit conversion
    Returns:
        pixel_array: a numpy array containing the image stored in dicom file
    """
    # read dicom files
    ds = dicom.read_file(filepath)

    # Get image numpy array
    # Image dicom file is in 16 bit and needs to be converted

    try:
        try:
            pixel_array = ds.pixel_array
        except:
            # pydicom cannot handle lossless jpeg
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([filepath])
            sitk_img = reader.Execute()
            pixel_array = sitk.GetArrayFromImage(sitk_img)[0, ...]
        try:
            if ds.PresentationLUTShape == 'INVERSE':
                pixel_array = pixel_array.max() - pixel_array
        except:
            logging.debug('PresentationLUTShape is INVERSE!')
        if to_bit == -1:
            # return the raw image
            return pixel_array
        if floor is not None and ceiling is not None:
            pixel_array = np.clip(ds.pixel_array, a_min=floor, a_max=ceiling)
            pixel_array = (pixel_array.astype(float) - floor) / (ceiling - floor) * (2 ** to_bit - 1)
            if to_bit == 8:
                pixel_array = pixel_array.astype(np.uint8)
            elif to_bit == 16:
                pixel_array = pixel_array.astype(np.uint16)
            else:
                raise ValueError('Unsupported bit type {}-bit!'.format(to_bit))
        elif ds.BitsStored != to_bit:
            print('Converting from {}-bit to {}-bit'.format(ds.BitsStored, to_bit))
            pixel_array = convert_to_unit8(pixel_array, to_bit=to_bit)
    except:
        # Some mask has size mismatch of exactly one, then manually discard one element
        try:
            # all masks are stored in uint8 format
            pixel_array = np.fromstring(ds.PixelData, dtype=np.uint8)
            pixel_array = pixel_array[mismatch:].reshape((ds.Rows, ds.Columns))
        except:
            raise ValueError('The img size mismatches in {} and is not {}'.format(filepath, mismatch))
    return pixel_array



def gen_single_input(single_image_path):
    """Read from image path and return a 3 channel color image in the format of numpy array

    Args:
        single_image_path:

    Returns:
        img_color: a 3 channel numpy array
    """
    filepath = single_image_path
    img = plt.imread(filepath) * 255
    img = img.astype(np.float32)
    # shenzhen dataset has 3 channels
    if len(img.shape) == 3:
        img_color = img
        # some nih png file has four channels RGBA
        # e.g., '/data/dataset/images/images_003/00006074_000.png'
        # use first 3 channels RGB only
        if img.shape[-1] == 4:
            img_color = img[:, :, :3]
    # most nih dataset has single grayscale channel
    elif len(img.shape) == 2:
        img_color = np.dstack([img] * 3)
    return img_color



def input_generator(filepath_list=[], dirname=None):
    """
    Yield a generator of image numpy array and the corresponding filepath
    Args:
        filepath_list:
        dirname:

    Yields:
        img_color:
        filepath
    """
    if not filepath_list and dirname:
        print('******* dirname specified!')
        filepath_list = [os.path.join(dirname, filename)
                     for filename in os.listdir(dirname) if filename.endswith('.png')]
    for filepath in filepath_list:
        img_color = gen_single_input(filepath)
        img_color = np.reshape(img_color, [-1])
        print('************* Input image array:')
        print([pix for pix in img_color[:100]])
        yield img_color, filepath


def diff_df(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)

def concat_df(input_csv_path_list, output_csv_path=None):
    """Concatenate csv files and return the combined dataframe

    Args:
        input_csv_path_list:
        output_csv_path:

    Returns:

    """
    df_all = None
    for csv_path in input_csv_path_list:
        df = pd.read_csv(csv_path)
        print('{}: length {}'.format(csv_path, len(df)))
        try:
            df_all = pd.concat([df_all, df])
        except:
            df_all = df
    if output_csv_path:
        df_all.to_csv(output_csv_path, index=False)
    print('concatenated df length {}'.format(len(df_all)))
    return df_all
