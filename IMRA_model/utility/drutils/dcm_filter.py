#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob2
import os
try:
    import dicom
except:
    import pydicom as dicom
import shutil
from tqdm import tqdm
import json

def writeinfo(output_filename, content):
    """Write the content to txt
    Args:
        output_filename:
        content:

    Returns:

    """
    if not os.path.isfile(output_filename):
        mode = "w"
    else:
        mode = "a"

    with open(output_filename, mode, encoding="utf-8") as f:
        f.write(content)
        f.write('\n')

def recursive_copy(src_dir, dst_dir):
    """Copy all files form src_dir to dst_dir with the same filename
    Args:
        src_dir:
        dst_dir:

    Returns:

    """
    files = os.listdir(src_dir)
    for file in files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        shutil.copyfile(src_file, dst_file)


class DicomFilter(object):
    """Filter out the dicom data that do not meed the demand in json file
    Args:
        input_dir:
        config_file: jsonfile
        output_dir

    """
    def __init__(self, input_dir, config_file, output_dir):
        self._input_dir = input_dir
        self._config_file = config_file
        self._output_dir = output_dir
        self._all_dcm_list = glob2.glob(os.path.join(self._input_dir, "**", "*"))
        self._inlier_txt = os.path.join(self._input_dir, "inlier.txt")
        self._outlier_txt = os.path.join(self._input_dir, "outlier.txt")
        self._notag_txt = os.path.join(self._input_dir, "notag.txt")

    def read_config(self):
        pass

    def filter_fracture(self, iscopy=False):
        """

        Args:
            iscopy:

        Returns:

        """
        fracture_filtered = []
        print("*********fracture filter******")
        for dcm in tqdm(self._all_dcm_list):
            if os.path.isdir(dcm):
                continue
            try:
                ds = dicom.read_file(dcm)
            except:
                continue
            # print(dcm)
            try:
                study_description = ds.StudyDescription
                content = dcm + "\t" + study_description
                if 'chest' in study_description.lower() or 'rib' in study_description.lower() or 'body' in study_description.lower():
                    fracture_filtered.append(dcm)
                    # writeinfo(self._inlier_txt, content)
                    continue
                else:
                    writeinfo(self._outlier_txt, content)
            except:
                content = dcm + " No StudyDescription Flag"
                writeinfo(self._notag_txt, content)
                print("No StudyDescription Flag")
            try:
                series_description = ds.SeriesDescription
                content = dcm + "\t" + series_description
                if 'chest' in series_description.lower() or 'rib' in series_description.lower() or 'body' in series_description.lower():
                    fracture_filtered.append(dcm)
                    # writeinfo(self._inlier_txt, content)
                    continue
                else:
                    writeinfo(self._outlier_txt, content)
            except:
                content = dcm + " No SeriesDescription Flag"
                writeinfo(self._notag_txt, content)
                print("No SeriesDescription Flag")
            try:
                body_part_examined = ds.BodyPartExamined
                content = dcm + "\t" + body_part_examined
                if 'chest' in body_part_examined.lower() or 'rib' in body_part_examined.lower() or 'body' in body_part_examined.lower():
                    fracture_filtered.append(dcm)
                    # writeinfo(self._inlier_txt, content)
                    continue
                else:
                    writeinfo(self._outlier_txt, content)
            except:
                content = dcm + " No BodyPartExamined Flag"
                writeinfo(self._notag_txt, content)
                print("No BodyPartExamined Flag")
            try:
                protocol_name = ds.ProtocolName
                content = dcm + "\t" + protocol_name
                if 'chest' in protocol_name.lower() or 'rib' in protocol_name.lower() or 'body' in protocol_name.lower():
                    fracture_filtered.append(dcm)
                    # writeinfo(self._inlier_txt, content)
                    continue
                else:
                    writeinfo(self._outlier_txt, content)
            except:
                content = dcm + " No Protocol Flag"
                writeinfo(self._notag_txt, content)
                print("No Protocol Flag")
        self._all_dcm_list = fracture_filtered

    def filter_PA(self, iscopy=False):
        """
        Returns:
            None
            Write out at most three txt file inlier.txt, outlier.txt, notag.txt

        """
        filtered = []
        print("*********PA filter******")
        for dcm in tqdm(self._all_dcm_list):
            if os.path.isdir(dcm):
                continue
            # print(dcm)
            img = dicom.read_file(dcm)
            try:
                view_position = img.ViewPosition
            except:
                try:
                    view_position = img.ProtocolName
                except:
                    content = dcm + " No ViewPosition and ProtocolName tag"
                    writeinfo(self._notag_txt, content)
                    continue
            if view_position == "PA":
                filtered.append(dcm)
                output_txt = self._inlier_txt
                # copy the file
                if iscopy:
                    if not os.path.exists(self._output_dir):
                        os.makedirs(self._output_dir)
                    src_dcm = dcm
                    dst_dcm = os.path.join(self._output_dir, os.path.basename(src_dcm))
                    shutil.copyfile(src_dcm, dst_dcm)
            else:
                output_txt = self._outlier_txt
            content = dcm + "\t" + view_position
            writeinfo(output_txt, content)


if __name__ == "__main__":
    root_dir = r"/media/Data/Data02/Datasets/DR/Collection/China_Sample"
    for dir in os.listdir(root_dir):
        input_dir = os.path.join(root_dir, dir)
        output_dir = r""
        config_file = r""
        dcm_filter = DicomFilter(input_dir, config_file, output_dir)
        dcm_filter.filter()