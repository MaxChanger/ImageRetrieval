"""
This file contains common utility functions for object detection
"""
import json

import glob2
import numpy as np
import os
import pandas as pd
import xmltodict

from projects.drutils import data
from projects.drutils import fileio
from projects.mammo_seg.dataset import ddsm


def write_result_dict(result_dict,
                      output_filepath,
                      is_append=True):
    """Write result_dict to text file

    result_dict has the following keys:
        key:
        detection_boxes:
        detection_scores:
        detection_classes:

    Args:
        result_dict:
        output_filepath:
        is_append: if True, append to existing file

    Returns:
        None
    """
    result_dict_copy = dict(result_dict)
    result_dict_copy.pop('original_image')  # original image is extremely space consuming in text format
    for key, val in result_dict_copy.items():
        if type(val) is np.ndarray:
            result_dict_copy[key] = result_dict_copy[key].tolist()
        if isinstance(val, bytes):
            result_dict_copy[key] = result_dict_copy[key].decode('utf-8')
    open_mode = 'a' if is_append else 'w'
    with open(output_filepath, open_mode) as f_out:
        json.dump(result_dict_copy, f_out, sort_keys=True, indent=4, separators=(',', ': '))


def get_result_dict_list_from_txt(result_dict_list_path):
    """Get a list of json objects from a file path

    Each json object (dictionary) from a text file. Each dictionary is
        evaluated from tensor_dict defined in ``

    Args:
        result_dict_list_path: path to a txt containing multiple json objects

    Returns:
        result_dict_list: a list of dictionaries
    """
    with open(result_dict_list_path, 'r') as f_in:
        text = f_in.read()
    # use `}{` to break multiple json objects
    # TODO: Add more robust parsing mechanism to detect if there is multiple json in the same txt
    text = text.replace('}{', '}###{')
    json_text_list = text.split('###')

    result_dict_list = []
    for json_text in json_text_list:
        result_dict = json.loads(json_text)
        for key, val in result_dict.items():
            if isinstance(val, list):
                result_dict[key] = np.array(val)
        result_dict_list.append(result_dict)

    return result_dict_list



def get_bbox_list_from_xml_file(xml_path, is_rescale=False, min_dimension=-1, max_dimension=-1, class_name='pathology'):
    """Get list of bbox coordinates given a path of an xml file

    All coordinates are in the order of (ymin, xmin, ymax, xmax), following the convention in `tf_example_decoder.py`

    Args:
        xml_path: the path to the xml annotation file
        is_rescale: flag indicating whether to do rescale or not
        min_dimension:
        max_dimension:

    Returns:
        bbox_list: numpy array of coordinates
        class_list: list of bbox classes
    """
    if not xml_path:
        print('xml path not found {}'.format(xml_path))
        return [], []
    with open(xml_path, 'r') as f_in:
        doc = xmltodict.parse(f_in.read())
    # print(json.dumps(doc, sort_keys=True,
    #                                 indent=4, separators=(',', ': ')))
    if is_rescale:
        if min_dimension == -1 and max_dimension == -1:
            raise ValueError('min_dimension and max_dimension cannot both be -1 when is_rescale is True')
        width = float(doc['annotation']['size']['width'])
        height = float(doc['annotation']['size']['height'])
        # TODO: add option to use partial dimensions
        width_new, height_new = data.get_new_dimensions((width, height), min_dimension, max_dimension)
        rescale_factor = width_new / width
    else:
        rescale_factor = 1

    bbox_list = []
    obj_list = doc['annotation']['object']
    # 'object' may be a list of jsons
    if not isinstance(obj_list, list):
        obj_list = [obj_list]
    for obj in obj_list:
        bbox_coord = []
        for key in ['ymin', 'xmin', 'ymax', 'xmax']:
            bbox_coord.append(float(obj['bndbox'][key]) * rescale_factor)
        bbox_list.append(bbox_coord)
    bbox_list = np.array(bbox_list)
    # print(bbox_list)

    # TODO: get everything from csv
    barename = os.path.basename(xml_path)
    barename = os.path.splitext(barename)[0]
    df = pd.read_csv(ddsm.generate_ddsm_dict()['csv_path'])
    rows = df[df['filename'].str.contains(barename)]
    class_list = rows[class_name].tolist()
    print(class_list)

    return bbox_list, class_list


def get_xml_file_path_from_image_name(image_name, xml_dir_or_txt):
    """Retrieve xml filepath from xml dir

    Args:
        image_name:
        xml_dir_or_txt:
    Returns:
        xml_path:
    """
    if os.path.isfile(xml_dir_or_txt):
        filepaths = fileio.read_list_from_txt(xml_dir_or_txt, field=-1)
    elif os.path.isdir(xml_dir_or_txt):
        filepaths = list(glob2.glob(os.path.join(xml_dir_or_txt, '**', '*xml')))
    else:
        raise ValueError('xml_dir_or_txt is neither a directory nor file')
    image_name_no_ext = os.path.splitext(os.path.basename(image_name))[0]
    xml_path_list = []
    for filepath in filepaths:
        if image_name_no_ext in filepath:
            xml_path_list.append(filepath)
            # print(filepath)
    assert len(xml_path_list) <= 1, 'xml_path_list expect 0 or 1 element but found {}!'.format(len(xml_path_list))
    if len(xml_path_list) == 1:
        xml_path = xml_path_list[0]
    else:
        xml_path = None
    return xml_path
