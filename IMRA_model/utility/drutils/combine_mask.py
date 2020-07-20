import os
import numpy as np

from tqdm import tqdm
import cv2
import matplotlib.pylab as plt
from collections import defaultdict


# combine into lesion masks based on lesion ids
# lesion ids are in the format of `patient-view-lesion`

class LesionMaskCombiner(object):
    """This class combines lesions based on lesion id

    Args:
        config: config dict with the following keys
            'write_flag': whether to write to disk
            'output_dir': if write_flag the target to write to
            'lesion_keys': the keys to selct and combine

    """

    def __init__(self, config):
        self.write_flag = config['write_flag']
        self.output_dir = config['output_dir']
        self.lesion_keys = config['lesion_keys']
        self.verbose = config['verbose']
        self.patient_masks_dict = defaultdict(list)

    def _write(self):
        if self.verbose:
            print('writing {} images'.format(len(self.patient_masks_dict)))
        for key, masks in tqdm(self.patient_masks_dict.items()):
            mask_array = None
            for mask_path in masks:
                tmp_mask = plt.imread(mask_path, -1).astype(np.bool)
                if mask_array is not None:
                    mask_array += tmp_mask
                else:
                    mask_array = tmp_mask
            mask_array = mask_array.astype(np.uint) * 255
            output_path = os.path.join(self.output_dir, '{}_combined.png'.format(key))
            cv2.imwrite(output_path, mask_array)

    def _combine_dict(self, lesion_dict):
        assert not set(self.lesion_keys) - set(lesion_dict.keys()), 'all keys are covered in lesion_dict'
        for lesion_key in self.lesion_keys:
            mask_path = lesion_dict[lesion_key]
            patient, view, lesion_id = lesion_key.split('-')
            patient_key = '-'.join([patient, view])
            self.patient_masks_dict[patient_key].append(mask_path)

    def process(self, lesion_dict):
        """
        Returns:
            patient_masks_dict: each val is a list of masks corresponding to the patient
        """
        if self.verbose:
            print('processing {} lesions'.format(len(self.lesion_keys)))

        # combine dict
        self._combine_dict(lesion_dict)

        # write to disk
        if self.write_flag:
            self._write()

        return self.patient_masks_dict



