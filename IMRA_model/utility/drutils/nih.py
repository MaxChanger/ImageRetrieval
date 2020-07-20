"""This file contains utility functions specific to nih dataset"""

import pandas as pd
import re
try:
    import dicom
except:
    import pydicom as dicom
import os
import glob2
import glob

from projects.drutils import fileio

def get_name_list():
    """
        # counts of images with different
        Atelectasis: 11535
        Cardiomegaly: 2772
        Effusion: 13307
        Infiltration: 19870
        Mass: 5746
        Nodule: 6323
        Pneumonia: 1353
        Pneumothorax: 5298
        Consolidation: 4667
        Edema: 2303
        Emphysema: 2516
        Fibrosis: 1686
        Pleural_Thickening: 3385
        Hernia: 227
        No Finding: 60412
    """
    NAME_LIST = ["Atelectasis",         # 0
                 "Cardiomegaly",        # 1
                 "Effusion",            # 2
                 "Infiltration",        # 3
                 "Mass",                # 4
                 "Nodule",              # 5
                 "Pneumonia",           # 6
                 "Pneumothorax",        # 7
                 "Consolidation",       # 8
                 "Edema",               # 9
                 "Emphysema",           # 10
                 "Fibrosis",            # 11
                 "Pleural_Thickening",  # 12
                 "Hernia",              # 13
                 "Tuberculosis",        # 14
                 "Image_Type",          # 15
                 "Costophrenic_Angle",  # 16
                 "Pneumothorax_Apex"]   # 17

    return NAME_LIST

def generate_nih_dict():
    """Generate ddsm dictionary containing important constants of ddsm dataset

    Keys:
        'csv_path':
        'basedir':
        'all_dicom_dir':
        'imagedir_list':
        'maskdir_list':

    Args:

    Returns:
        ddsm_dict: dictionary containing important constants of ddsm dataset

    """
    nih_dict = {}

    nih_dict['csv_path'] = r'/media/Data/Data02/Datasets/DR/NIH/Data_Entry_2017.csv'
    nih_dict['NAME_LIST'] = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                             "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                             "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "Tuberculosis"]
    nih_dict['basedir'] = r'/media/Data/Data02/Datasets/DR/NIH/'
    nih_dict['all_image_path_txt'] = r'/media/Data/Data02/Datasets/DR/NIH/images/All/images.txt'
    nih_dict['annotation_dir'] = r'/media/Data/Data02/Datasets/DR/NIH/annotation/ver_2/images_Mass_or_Nodule/annotations/xmls'
    return nih_dict


def get_image_path_from_filename(filename):
    """Get full path to image from filename

    Args:
        filename: image name, could be basename or barename (without ext)

    Returns:
        image_path
    """
    # convert filename to barename
    barename = os.path.basename(filename)
    barename = os.path.splitext(barename)[0]

    all_image_path_txt = generate_nih_dict()['all_image_path_txt']
    image_path_list = fileio.read_list_from_txt(all_image_path_txt, field=0)
    image_path_list = [path for path in image_path_list if barename in path]
    print(image_path_list)
    assert len(image_path_list) == 1, 'Found {} matching files!'.format(len(image_path_list))
    image_path = image_path_list[0]

    return image_path


def convert_label_to_one_hot_index(label):
    """Convert label to one-hot index with '|' as spliter

    Example:
    Input:
        'Atelectasis|Cardiomegaly|Fibrosis'
    Output:
        '11000000000100'
    """
    index = list('00000000000000')
    NAME_LIST = generate_nih_dict()['NAME_LIST']
    for i, name in enumerate(NAME_LIST):
        if name in label:
            index[i] = '1'
    return ''.join(list(index))



def get_label_from_csv(image_name, df_or_csv, is_one_hot=True):
    """
    Read label from csv given an image file. Only for NIH dataset.
    Args:
        image_name: basename of the image file path
        df_or_csv: dataframe created from csv file, or the csv file
        is_one_hot: optional, if true, convert string to one hot encoding
    Return:
        label: a string
    """

    # generate df
    if isinstance(df_or_csv, str):
        try:
            df = pd.read_csv(df_or_csv)
        except:
            raise IOError("Invalid csv file {}!".format(df_or_csv))
    elif isinstance(df_or_csv, pd.core.frame.DataFrame):
        df = df_or_csv
    else:
        raise ValueError("df_or_csv is not df nor csv!")

    # look up image label in df
    basename = os.path.basename(image_name)
    # sometimes basename does not contain suffix
    row = df.loc[df['Image Index'].str.contains(basename)]
    try:
        label = row["Finding Labels"].item()
    except:
        print('This row has multiple occurances \n{}'.format(row))

    if is_one_hot:
        label = convert_label_to_one_hot_index(label)
    return label


def get_image_level_label_from_filename(image_name):
    """Given the image name, retrieve image-level label

    Args:
        image_name:

    Returns:
        label: image-level label
    """

    csv_path = generate_nih_dict()['csv_path']
    label_string = get_label_from_csv(image_name, csv_path, is_one_hot=False)
    label = int("Mass" in label_string or "Nodule" in label_string)

    return label
