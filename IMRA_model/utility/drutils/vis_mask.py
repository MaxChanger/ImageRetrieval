"""This file contains utility functions used for mask annotation visualization"""

import os
import numpy as np
import argparse
from skimage import measure
import glob2
from tqdm import tqdm
import matplotlib.pylab as plt
import sys

sys.path.append('/data/SigmaPy/projects/mammo_seg/mask_rcnn/')
from projects.mammo_seg.mask_rcnn.mrcnn import visualize
from projects.drutils import fileio
from projects.drutils import roc

class MaskVisualizer(object):
    """Visualize mask annotations"""

    def __init__(self, config):
        self.output_dir = config['output_dir']
        self.class_names = config['class_names']
        self.class_name_fn = config['class_name_fn']
        self.filter_keys = config['filter_keys']
        self.show_orig = config['show_orig']
        self.subplot_size = config['subplot_size']

    @staticmethod
    def get_box_from_mask(mask):
        y_range, x_range = np.where(mask > 0)
        x_min, x_max = np.min(x_range), np.max(x_range)
        y_min, y_max = np.min(y_range), np.max(y_range)
        return y_min, x_min, y_max, x_max

    def get_images_and_annotation(self, png_path, mask_path_list, class_names, parse_path_fn):
        """
        parse_path_fn: function to parse class_name from path
        """
        image = plt.imread(png_path, -1)
        image_3ch = np.dstack([image] * 3)
        masks = []
        boxes = []
        class_ids = []
        if type(mask_path_list) == str:
            mask_path_list = [mask_path_list]
        if mask_path_list:
            for mask_path in mask_path_list:
                mask = plt.imread(mask_path, -1).astype(np.bool)
                # connected component analysis
                labeled_mask_array = measure.label(mask, connectivity=2)
                for i in range(np.max(labeled_mask_array)):
                    mask = (labeled_mask_array == i + 1)
                    masks.append(mask)
                    box = self.get_box_from_mask(mask)
                    boxes.append(box)
                    class_name = parse_path_fn(mask_path)
                    class_id = class_names.tolist().index(class_name)
                    class_ids.append(class_id)
            masks = np.dstack(masks)
        masks = np.array(masks)
        boxes = np.array(boxes)
        class_ids = np.array(class_ids)
        return image_3ch, boxes, masks, class_ids

    @staticmethod
    def visualize_multiple_gt(image_3ch, boxes_dict, masks_dict, class_ids_dict, class_names, key,
                              fig_dir=None,
                              show_orig=False,
                              subplot_size=(16, 16)):
        assert set(boxes_dict.keys()) == set(masks_dict.keys()) == set(class_ids_dict.keys())
        n_annotation_series = len(boxes_dict.keys())
        if show_orig:
            # show original image without annotation
            axes = roc.get_ax(1, n_annotation_series + 1, size=subplot_size)
            ax, axes = axes[0], axes[1:]
            empty_array = np.array([])
            visualize.display_instances(image=image_3ch,
                                        boxes=empty_array,
                                        masks=empty_array,
                                        class_ids=empty_array,
                                        class_names=class_names,
                                        show_mask=False,
                                        show_bbox=False,
                                        ax=ax,
                                        title='orig image',
                                        verbose=False)
        else:
            axes = roc.get_ax(1, n_annotation_series)
        series_keys = boxes_dict.keys()
        assert len(axes) == len(series_keys)
        for idx, (ax, series_key) in enumerate(zip(axes, series_keys)):
            # Display GT bbox and mask
            visualize.display_instances(image=image_3ch,
                                        boxes=boxes_dict[series_key],
                                        masks=masks_dict[series_key],
                                        class_ids=class_ids_dict[series_key],
                                        class_names=class_names,
                                        ax=ax,
                                        title=series_key,
                                        verbose=False)

        # Save to model log dir
        fig_dir = fig_dir or '/tmp/tmp/'
        fileio.maybe_make_new_dir(fig_dir)
        fig_path = os.path.join(fig_dir, 'gt_{}.png'.format(key))
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close('all')

    def _get_keys(self, png_dict, mask_dict):
        mask_dict_keys = set()
        for annotation_series in mask_dict.keys():
            mask_dict_keys |= set(mask_dict[annotation_series].keys())
        keys = list(set(mask_dict_keys) & set(png_dict.keys()))

        if self.filter_keys is not None:
            keys = list(set(keys) & set(self.filter_keys))
        keys = sorted(keys)
        return keys

    def _get_dicts(self, png_dict, mask_dict, key):
        # populate boxes_dict, masks_dict, class_id_dict
        boxes_dict = {}
        masks_dict = {}
        class_ids_dict = {}
        for data_series_key in list(mask_dict.keys())[:]:
            png_path = png_dict[key]
            mask_path_list = mask_dict[data_series_key].get(key, [])

            (image_3ch,
             boxes_dict[data_series_key],
             masks_dict[data_series_key],
             class_ids_dict[data_series_key]) = self.get_images_and_annotation(
                png_path, mask_path_list, self.class_names, parse_path_fn=self.class_name_fn)
        return image_3ch, boxes_dict, masks_dict, class_ids_dict

    def process(self, png_dict, mask_dict):
        # batch generating stack images
        keys = self._get_keys(png_dict, mask_dict)
        for key in tqdm(keys[:]):
            # populate boxes_dict, masks_dict, class_id_dict
            image_3ch, boxes_dict, masks_dict, class_ids_dict = self._get_dicts(png_dict, mask_dict, key)
            # visualize
            self.visualize_multiple_gt(image_3ch, boxes_dict, masks_dict, class_ids_dict, self.class_names, key,
                                       fig_dir=self.output_dir,
                                       show_orig=self.show_orig,
                                       subplot_size=self.subplot_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        png_search_path = '/data/log/test/AllPNG/**/*png'
        mask_search_path1 = '/data/log/mammo/calc_cluster/above_100/**/*png'
        output_dir = '/data/log/test/'
        filter_txt = None

        dir_key_fn = lambda x: '-'.join(x.split(os.sep)[-3:-1])
        file_key_fn = lambda x: os.path.basename(x).split('.')[0].split('_')[0]
        mask_dict1 = {file_key_fn(x): x for x in glob2.glob(mask_search_path1)}
        mask_dict = {'combined': mask_dict1, 'combined2': mask_dict1}
        png_dict = {file_key_fn(x): x for x in glob2.glob(png_search_path)}

        if filter_txt is not None:
            filter_keys = [file_key_fn(x) for x in fileio.read_list_from_txt(filter_txt)]
        else:
            filter_keys = None

        config = {}
        config['output_dir'] = output_dir
        config['class_name_fn'] = lambda x: 'mass'
        config['class_names'] = np.array(['mass'])
        config['key_fn'] = lambda x: os.path.basename(x).split('.')[0].split('_')[0]
        config['png_search_path'] = png_search_path
        config['filter_keys'] = filter_keys
        config['show_orig'] = True
        config['subplot_size'] = (20, 30)

        visualizer = MaskVisualizer(config=config)
        visualizer.process(png_dict, mask_dict)