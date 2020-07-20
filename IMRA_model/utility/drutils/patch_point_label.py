"""Utility classes and functions to crop images to patches"""
import glob
import glob2
import json
import logging
from datetime import datetime
import cv2
import numpy as np
import os
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
os.sys.path.append('/data1/MedicalImage/User/xing/SigmaPy')
from projects.drutils import fileio
from projects.drutils import augmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class PairedDictGenerator(object):
    def __init__(self, image_search_path, mask_search_path, output_dir,
                 image_suffix=".tif", mask_suffix='_mask.tif'):
        self.output_dir = output_dir
        self.image_files = self._find_data_files(image_search_path, image_suffix)
        self.mask_files = self._find_data_files(mask_search_path, mask_suffix)
        print(mask_suffix,mask_search_path,self.mask_files)

    def get_paired_image_and_mask(self, join='inner', key_names=('image', 'mask')):
        """Create paired image and mask files

        Find common files in two folders and create dictionary paired_dict. Each key has two keys:
            'image': path to the image file
            'mask': path to the mask file

        Args:
            join: {'inner', 'outer', 'left', 'right'}
                'inner': find intersection between two lists
                'outer': find union of two lists
                'left': first list (image list)
                'right': second list (mask list)
        Returns:
            paired_dict: paired_dict[key]['image'] and paired_dict[key]['mask'] is a file
        """
        image_key, mask_key = key_names
        # print(self.image_files)
        self.image_dict = {self.get_image_key_fn(filepath):filepath for filepath in self.image_files}
        self.mask_dict = {self.get_mask_key_fn(filepath): filepath for filepath in self.mask_files}
        if join == 'inner':
            keys = set(self.image_dict.keys()) & set(self.mask_dict.keys())
        elif join == 'outer':
            keys = set(self.image_dict.keys()) | set(self.mask_dict.keys())
        elif join == 'left':
            keys = set(self.image_dict.keys())
        elif join == 'right':
            keys = set(self.mask_dict.keys())
        else:
            raise KeyError('Unsupported join method {}'.format(join))
        paired_dict = {}
        for key in keys:
            paired_dict[key] = {}
            paired_dict[key][image_key] = self.image_dict.get(key, '')
            paired_dict[key][mask_key] = self.mask_dict.get(key, '')
        logging.debug('paired_dict with length {}'.format(len(paired_dict)))
        filepath = os.path.join(self.output_dir, 'paired_dict.json')
        fileio.maybe_make_new_dir(os.path.dirname(filepath))
        with open(filepath, 'w') as f_out:
            json.dump(paired_dict, f_out, indent=4, sort_keys=True)
        return paired_dict

    @staticmethod
    def _find_data_files(search_path, suffix):
        print(search_path)
        all_files = glob2.glob(search_path)
        print(all_files)
        return [name for name in all_files if suffix in name]

    @staticmethod
    def get_image_key_fn(filepath):
        """Get key for paired_dict
        Example:
            '/media/Data/Data02/Datasets/Mammogram/INbreast/mass_mask/22678646_combined.png'
            --> 22678646

        Args:
            filepath:

        Returns:
            A string as key in paired_dict
        """


        # FIXME: use a generalized method to get key for different dataset
        if 'DDSM' in filepath:
            return os.path.basename(filepath).split('.')[0]
        return os.path.basename(filepath).split('.')[0].split('_')[0]


    @staticmethod
    def get_mask_key_fn(filepath):
        """Get key for paired_dict
        Example:
            '/media/Data/Data02/Datasets/Mammogram/INbreast/AllPNG/53586805_e5f3f68b9ce31228_MG_R_CC_ANON.dcm.png'
            --> 53586805

        Args:
            filepath:

        Returns:
            A string as key in paired_dict
        """

        # FIXME: use a generalized method to get key for different dataset
        if 'DDSM' in filepath:
            return ('_').join(os.path.basename(filepath).split('.')[0].split('_')[:-2])
        return os.path.basename(filepath).split('.')[0].split('_')[0]



class VectorizedPairedDictGenerator(PairedDictGenerator):
    """Extended PairedDictGenerator

    Each of image_search_path and mask_search_path can be a list of glob search patterns

    Below is the doctest:

    >>> from pathlib import Path
    >>> import os
    >>> import shutil
    >>> test_path = r'/tmp/test/'
    >>> for name in ['1', '2', '3']:
    ...   for folder in ['images', 'mask1', 'mask2']:
    ...     filename = os.path.join(test_path, folder, name + '.png')
    ...     os.makedirs(os.path.dirname(filename), exist_ok=True)
    ...     Path(filename).touch()
    >>> image_search_path = os.path.join(test_path, 'images', '*png')
    >>> mask_search_path = [os.path.join(test_path, 'mask1', '*png'), os.path.join(test_path, 'mask2', '*png')]
    >>> output_dir = test_path
    >>> image_suffix=".png"
    >>> mask_suffix='.png'
    >>> dict = VectorizedPairedDictGenerator(
    ...     image_search_path, mask_search_path,
    ...     output_dir, image_suffix, mask_suffix).get_paired_image_and_mask()
    >>> shutil.rmtree(test_path)
    """
    def __init__(self, image_search_path, mask_search_path, output_dir,
                 image_suffix=".tif", mask_suffix='_mask.tif'):
        self.output_dir = output_dir
        self.image_files = self._find_data_files(image_search_path, image_suffix)
        print('------------mask start--------------')
        self.mask_files = self._find_data_files(mask_search_path, mask_suffix)
        self.is_vector = (
            (isinstance(image_search_path, (list, tuple)) and len(image_search_path) > 1)
            or (isinstance(mask_search_path, (list, tuple)) and len(mask_search_path) > 1)
        )

    @staticmethod
    def _find_data_files(search_path_list, suffix):
        if not isinstance(search_path_list, (list, tuple)):
            search_path_list = [search_path_list]
        image_files_list = []
        for single_image_search_path in search_path_list:
            tmp_image_files = PairedDictGenerator._find_data_files(single_image_search_path, suffix)
            image_files_list.append(tmp_image_files)  # list of list
        assert len(set(len(image_files) for image_files in image_files_list)) <= 1, \
            'Different folders have different number of files for list {}'.format(search_path_list)
        # flatten list of list
        image_files = [item for image_files in image_files_list for item in image_files]
        return image_files

    def get_paired_image_and_mask(self, join='inner'):
        """Create paired image and mask files

        Find common files in two folders and create dictionary paired_dict. Each key has two keys:
            'image': path to the image file
            'mask': path to the mask file

        Args:
            join: {'inner', 'outer', 'left', 'right'}
                'inner': find intersection between two lists
                'outer': find union of two lists
                'left': first list (image list)
                'right': second list (mask list)
        Returns:
            paired_dict: paired_dict[key]['image'] and paired_dict[key]['mask'] is a list
        """

        # each key in image_dict and mask_dict corresponds to a list
        if not self.is_vector:
            self.image_dict = {self.get_image_key_fn(filepath): filepath for filepath in self.image_files}
            self.mask_dict = {self.get_mask_key_fn(filepath): filepath for filepath in self.mask_files}
        else:
            self.image_dict = {}
            for filepath in self.image_files:
                key = self.get_image_key_fn(filepath)
                if key not in self.image_dict:
                    self.image_dict[key] = []
                self.image_dict[key].append(filepath)
            self.mask_dict = {}
            for filepath in self.mask_files:
                key = self.get_mask_key_fn(filepath)
                if key not in self.mask_dict:
                    self.mask_dict[key] = []
                self.mask_dict[key].append(filepath)
        if join == 'inner':
            keys = set(self.image_dict.keys()) & set(self.mask_dict.keys())
        elif join == 'outer':
            keys = set(self.image_dict.keys()) | set(self.mask_dict.keys())
        elif join == 'left':
            keys = set(self.image_dict.keys())
        elif join == 'right':
            keys = set(self.mask_dict.keys())
        else:
            raise KeyError('Unsupported join method {}'.format(join))
        if self.is_vector:
            empty_val = []
        else:
            empty_val = ''
        paired_dict = {}
        for key in keys:
            paired_dict[key] = {}
            paired_dict[key]['image'] = self.image_dict.get(key, empty_val)
            paired_dict[key]['mask'] = self.mask_dict.get(key, empty_val)
        logging.debug('paired_dict with length {}'.format(len(paired_dict)))
        filepath = os.path.join(self.output_dir, 'paired_dict.json')
        fileio.maybe_make_new_dir(os.path.dirname(filepath))
        with open(filepath, 'w') as f_out:
            json.dump(paired_dict, f_out, indent=4, sort_keys=True)
        return paired_dict


class PatchConverter(PairedDictGenerator):
    """Convert images to patches in the same folder

    Note: All coordinates are in (y, x) or (h, w) order.

    Args:
        image_search_path: a glob search pattern to find all images and labels
        mask_search_path:
        output_dir:
        block_size: patch size. If (-1, -1), return the whole image
        overlap:
        image_suffix:
        mask_suffix:
        remove_zero_threshold: applied to LABEL patch. The number of non-zero pixels
            below which to discard the label patch. If -1 do not discard any label patch.
        remove_fg_threshold: applied to IMAGE patch. Default to 0.5. Discard patches
            if 50% of image patch is background
    """
    def __init__(self, image_search_path, mask_search_path, output_dir,
                 block_size=(100, 100), overlap=(0, 0),
                 image_suffix=".tif", mask_suffix='_mask.tif',
                 remove_zero_threshold=-1, ignore_padding=0, remove_fg_threshold=0.5, scale=1.0):
        super(PatchConverter, self).__init__(image_search_path, mask_search_path, output_dir,
                                             image_suffix=image_suffix, mask_suffix=mask_suffix)
        # NB. update block_size, overlap, central_size and padding_size together
        self.block_size = block_size
        self.overlap = overlap
        self.central_size = np.array(self.block_size) - np.array(self.overlap)
        assert np.all(item % 2 == 0 for item in self.overlap), 'overlap must be even integers!'
        # padding_size used to compensate for prediction center cropping
        self.padding_size = (overlap[0] // 2, overlap[1] // 2)
        # Temp folder to store cropped patches. Use timestamp to avoid racing condition
        time_now = datetime.now()
        time_string = time_now.strftime("%Y%m%d-%Hh%M%p%S")
        self.prediction_patch_dir = '/tmp/tmp_patches_' + time_string
        self.remove_zero_threshold = remove_zero_threshold
        self.ignore_padding = ignore_padding
        fileio.maybe_make_new_dir(self.output_dir)
        self.remove_fg_threshold = remove_fg_threshold
        self.scale = scale

    def crop_patches(self, image_array, top_left=(0, 0), padding='zero'):
        """

        Args:
            image_array: a numpy array containing the image
            top_left: a tuple containing the (y, x) coordinate of the patch
                the patch will have shape (h, w) = self.block_size
            padding: if patch is partially outside of image, use padding
                'zero': padding with zero
                'mirror': mirror padding

        Returns: a numpy array of the patch of from image_array

        """
        # print(image_array.shape)
        height, width = image_array.shape
        y, x = top_left
        h_patch, w_patch = self.block_size
        # TODO: move padding to self._preprocess_image()
        y_padding, x_padding = self.padding_size
        image_array_padded = np.zeros((height + 2 * y_padding, width + 2 * x_padding))
        image_array_padded[y_padding:(y_padding + height), x_padding:(x_padding + width)] = image_array
        y += y_padding
        x += x_padding
        patch_array = image_array_padded[y:(y + h_patch), x:(x + w_patch)]
        # pad patch_array to block_size if patch array is partial
        patch_array = augmentation.crop_or_pad(patch_array, self.block_size)
        assert patch_array.shape == tuple(self.block_size), 'patch_array shape {}'.format(patch_array.shape)
        return patch_array

    def generate_top_left_list(self, image_shape, method='valid'):
        """Generate a list of coordinates of the top left corner point

        Args:
            image_shape:
            method: can be 'valid' or 'padding'
                'valid': adjust end block so that it does not extend beyond valid image
                'padding': pad partial patches with zeroes
            state:
                'train': during training, crop from the original image
                'predict': during prediction, crop from padded image to generate the whole image mask


        Returns:
            top_left_list: a list of (y, x) tuples
        """
        height, width = image_shape
        h_block, w_block = self.block_size
        h_overlap, w_overlap = self.overlap
        h_padding, w_padding = self.padding_size
        # generate block increments
        dy = h_block - h_overlap
        dx = w_block - w_overlap
        assert dy > 0 and dx > 0, "overlap larger than block size!"
        # crop from padded image
        # from (-h_padding, -w_padding) to (height + h_padding, width + w_padding)
        y_list = list(range(-h_padding, height + h_padding - h_overlap, dy))
        x_list = list(range(-w_padding, width + w_padding - w_overlap, dx))
        logging.debug('x_list before adjustment: {}'.format(x_list))
        logging.debug('y_list before adjustment: {}'.format(y_list))
        if method == 'valid':
            y_list[-1] = height + h_padding - h_block
            x_list[-1] = width + w_padding - w_block
        elif method == 'padding':
            # padding implemented in self.crop_patches()
            pass
        top_left_list = [(y, x) for y in y_list for x in x_list]
        logging.debug('image total size {} x {}'.format(height, width))
        logging.debug('x_list: {}'.format(x_list))
        logging.debug('y_list: {}'.format(y_list))
        return top_left_list

    @staticmethod
    def _get_patch_name(key, idx, name='', is_mask=False, output_dir=None):
        if not is_mask:
            file_name = '{}_{:03d}{}.png'.format(key, idx, name)
        else:
            file_name = '{}_{:03d}{}_mask.png'.format(key, idx, name)
        file_path = os.path.join(output_dir, file_name)
        return file_path

    def save_patch_to_file(self, image_array, key, idx, name='', is_mask=False, output_dir=None):
        """Save image patches to self.output_dir

        Args:
            image_array:
            key:
            idx: the index number of pathces corresponding to the same key
            is_mask: boolean to indicate if the patch is image or mask

        Returns:
            None
        """
        output_dir = output_dir or self.output_dir # if output_dir is None, use self.output_dir
        file_path = self._get_patch_name(key, idx, name=name, is_mask=is_mask, output_dir=output_dir)
        cv2.imwrite(file_path, image_array)

    def batch_convert_patches(self, n_batch=None, method='valid'):
        """Convert image to patches

        Args:
            n_batch: number of images to convert

        Returns:
            None
        """
        paired_dict = self.get_paired_image_and_mask()
        keys = sorted(paired_dict.keys())
        if n_batch is not None:
            keys = keys[:n_batch]
        for key in tqdm(keys):
            logging.info('Cropping {}'.format(paired_dict[key]['image']))
            image_array = fileio.load_image_to_array(paired_dict[key]['image'])
            if len(image_array.shape)>2:
                image_array = image_array[:,:,1]
            image_array = self._preprocess_image(image_array)
            logging.info('Cropping {}'.format(paired_dict[key]['mask']))
            mask_array = fileio.load_image_to_array(paired_dict[key]['mask'])
            mask_array = self._preprocess_mask(mask_array, image_array.shape)
            top_left_list = self.generate_top_left_list(image_array.shape, method=method)
            idx = 0
            idx_neg = 0
            # print(key,image_array.shape,mask_array.shape)
            for top_left in top_left_list:
                y, x = top_left
                name = '_y{}_x{}'.format(y, x) # add name to indicate original position of the patch
                patch_array = self.crop_patches(mask_array, top_left)
                if self.ignore_padding > 0:
                    central_patch_array = patch_array[self.ignore_padding:-self.ignore_padding,
                                          self.ignore_padding:-self.ignore_padding]
                else:
                    central_patch_array = patch_array
                if central_patch_array.astype(np.bool).sum() >= self.remove_zero_threshold:
                    print(central_patch_array.astype(np.bool).sum())
                    self.save_patch_to_file(patch_array, key, idx, name=name, is_mask=True)
                    patch_array = self.crop_patches(image_array, top_left)
                    self.save_patch_to_file(patch_array, key, idx, name=name, is_mask=False)
                    idx += 1
                else:
                    # negative patches, also save to file in a different directory if image contains more than half
                    # FG. Otherwise discard.
                    patch_array = self.crop_patches(image_array, top_left)
                    if patch_array.astype(np.bool).sum() >= patch_array.size * self.remove_fg_threshold:
                        name = name + '_neg'
                        patch_array = self.crop_patches(mask_array, top_left)
                        self.save_patch_to_file(patch_array, key, idx_neg, name=name, is_mask=True)
                        patch_array = self.crop_patches(image_array, top_left)
                        self.save_patch_to_file(patch_array, key, idx_neg, name=name, is_mask=False)
                        idx_neg += 1

    def _preprocess_image(self, image_array, interpolation=cv2.INTER_AREA):
        """Image level preprocessing. Reimplement base class dummy method"""
        image_array = augmentation.resize(image_array, dst_shape=(0, 0), scale=self.scale, interpolation=interpolation)
        # image_array = augmentation.pad(image_array, padding=(self.padding_size))
        return image_array

    def _preprocess_mask(self, mask_array, target_shape=None, interpolation=cv2.INTER_AREA):
        """Image level preprocessing. Reimplement this to add preprocessing"""
        print(mask_array.shape)
        if target_shape and mask_array != target_shape:
            # dsize in the order of (x, y)
            print('mask not equal to image')
            mask_array = augmentation.resize(mask_array, dst_shape=target_shape[::-1], interpolation=interpolation)
        logging.debug('mask shape {}'.format(mask_array.shape))
        return mask_array[:,:,0]

    def split_train_and_test(self, valid_size=0.2, test_size=0.1, random_state=42,
                             valid_txt_path=None, train_txt_path=None,
                             dry_run=True):
        """Split the output files in output_dir into train, valid and test dir

        Split ratio
            train : valid : test = (1 - valid_size - test_size) : valid_size : test_size

        Note: If both valid_txt_path and train_txt_path are verified, use these txt to split patches. In this case,
            there is no test set.

        Args:
            valid_size: ratio of valid/all to split data
            test_size: ratio of test/train to split data
            dry_run: if True, do not move files
        Returns:
            None
        """
        mask_files = self._find_data_files(os.path.join(self.output_dir, '*png'), suffix='_mask.png')
        image_files = [file_path.replace('_mask', '') for file_path in mask_files]

        def _move_files(files_list, subdir='train'):
            for file_path in files_list:
                new_dir = os.path.join(os.path.dirname(file_path), subdir)
                fileio.maybe_make_new_dir(new_dir)
                new_path = os.path.join(new_dir, os.path.basename(file_path))
                if dry_run:
                    logging.debug('move {} to {}'.format(file_path, new_path))
                else:
                    shutil.move(file_path, new_path)

        image_files_valid = []
        image_files_train = []
        image_files_test = []
        mask_files_valid = []
        mask_files_train = []
        mask_files_test = []
        if valid_txt_path and train_txt_path:
            valid_keys = [self.get_image_key_fn(filepath) for filepath in fileio.read_list_from_txt(valid_txt_path)]
            train_keys = [self.get_image_key_fn(filepath) for filepath in fileio.read_list_from_txt(train_txt_path)]

            for image_file, mask_file in zip(image_files, mask_files):
                if self.get_image_key_fn(image_file) in valid_keys:
                    image_files_valid.append(image_file)
                    mask_files_valid.append(mask_file)
                elif self.get_image_key_fn(image_file) in train_keys:
                    image_files_train.append(image_file)
                    mask_files_train.append(mask_file)
                else:
                    raise ValueError('valid_txt_path and train_txt_path is not collectively exhaustive!')
        else:
            image_files_train, image_files_test, mask_files_train, mask_files_test = \
                train_test_split(image_files, mask_files,
                                 test_size=test_size + valid_size,
                                 random_state=random_state)
            image_files_valid, image_files_test, mask_files_valid, mask_files_test = \
                train_test_split(image_files_test, mask_files_test,
                                 test_size=test_size / (test_size + valid_size),
                                 random_state=random_state)
        logging.debug(len(image_files_train))
        logging.debug(len(image_files_valid))
        logging.debug(len(image_files_test))
        _move_files(image_files_train, subdir='train')
        _move_files(mask_files_train, subdir='train')
        _move_files(image_files_valid, subdir='valid')
        _move_files(mask_files_valid, subdir='valid')
        _move_files(image_files_test, subdir='test')


class ToyPatchConverter(PatchConverter):
    """Toy model for sanity check"""
    def __init__(self):
        image_search_path = r'/data/log/mammo/toy/random_image.png*'
        mask_search_path = r'/data/log/mammo/toy/random_mask.png*'
        output_dir = r'/data/log/mammo/toy/random/'
        super().__init__(image_search_path, mask_search_path, output_dir,
                           block_size=(512, 512), overlap=(256, 256),
                           image_suffix='.png', mask_suffix='.png',
                           remove_zero_threshold=100)
    def deploy(self):
        self.batch_convert_patches()


class MassOrCalcPatchConverter(PatchConverter):
    def __init__(self, image_search_path, mask_search_path, output_dir, valid_txt_path=None, train_txt_path=None,
                 dataset='inbreast', scale=0.25, ignore_padding=0, remove_zero_threshold=0, remove_fg_threshold=0.5,
                 crop_method='valid', whole_image=False):
        if whole_image:
            # override parameters for whole image
            remove_zero_threshold = 0
            remove_fg_threshold = 0
            crop_method = 'padding'
        super().__init__(image_search_path, mask_search_path, output_dir,
                         block_size=(256, 256), overlap=(128, 128),
                         image_suffix='.png', mask_suffix='.png',
                         scale=scale,
                         ignore_padding=ignore_padding,
                         remove_zero_threshold=remove_zero_threshold,
                         remove_fg_threshold=remove_fg_threshold)
        self.valid_txt_path = valid_txt_path
        self.train_txt_path = train_txt_path
        self.dataset = dataset
        self.crop_method = crop_method

    def get_image_key_fn(self, filepath):
        if self.dataset in ['ddsm']:
            return os.path.basename(filepath).split('.')[0]
        elif self.dataset in ['china', 'inbreast']:
            return os.path.basename(filepath).split('.')[0].split('_')[0]

    def get_mask_key_fn(self, filepath):
        if self.dataset in ['ddsm']:
            return os.path.basename(filepath).split('.')[0]
        elif self.dataset in ['china', 'inbreast']:
            return os.path.basename(filepath).split('.')[0].split('_')[0]

    def deploy(self):
        self.batch_convert_patches(method=self.crop_method)
        self.split_train_and_test(test_size=0.1, valid_size=0.2, dry_run=False,
                                  valid_txt_path=self.valid_txt_path,
                                  train_txt_path=self.train_txt_path)

if __name__ == '__main__':
    print('so far so good')
    import doctest
    doctest.testmod()

    # set up flags
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('task', '', 'mass or calc training')
    tf.app.flags.DEFINE_string('dataset', 'inbreast', 'inbreast or ddsm or china for mass training')
    tf.app.flags.DEFINE_boolean('ignore_single_point', True, 'Whether to ignore single point annotation for calc')
    tf.app.flags.DEFINE_boolean('whole_image', False, 'Whether to crop whole image')
    tf.app.flags.DEFINE_float('scale', 0.25, 'Ratio to scale down original image before cropping patches')

    logging.getLogger().setLevel(logging.INFO)

    print(FLAGS.task == 'Calc_crop_label')
    logging.debug(FLAGS.task)
    ## patch conversion

    if FLAGS.task == 'Calc_crop_label_with_Gaussian_points':
        image_search_path = r'/data1/Image_data/Mammography_data/object_detector/All_B_data//AllPNG/*png'
        if FLAGS.ignore_single_point:
            mask_search_path = r'/data1/Image_data/Mammography_data/object_detector/All_B_data//AllPMask/*png'
        else:
            mask_search_path = r'/data1/Image_data/Mammography_data/object_detector/All_B_data//AllPMask/*png'

        output_dir = '/data1/Image_data/Mammography_data/object_detector/All_B_data/calc_patches_point_256'
        print('start...',output_dir)

        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                                 scale=1.0,
                                 ignore_padding=0,
                                 remove_zero_threshold=50).batch_convert_patches(method='valid')
        print('end...', output_dir)

    if FLAGS.task == 'Calc_crop_label':
        image_search_path = r'/data1/Image_data/Mammography_data/object_detector/Mammo_calc_cluster/Wqy/AllPNG/RM/*png'
        if FLAGS.ignore_single_point:
            mask_search_path = r'/data1/Image_data/Mammography_data/object_detector/Mammo_calc_cluster/Wqy/AllBMask/*png'
        else:
            mask_search_path = r'/data1/Image_data/Mammography_data/object_detector/Mammo_calc_cluster/Wqy/AllBMask/*png'

        output_dir = '/data1/Image_data/Mammography_data/log/Wqy/calc_patches_point_256'
        print('start...',output_dir)

        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                                 scale=1.0,
                                 ignore_padding=0,
                                 remove_zero_threshold=200).batch_convert_patches(method='valid')
        print('end...', output_dir)


    if FLAGS.task == 'calc_crop':
        image_search_path = r'/data1/Image_data/Mammography_data/INbreast/AllPNG_norm_6_6/*png'
        if FLAGS.ignore_single_point:
            mask_search_path = r'/data1/Image_data/Mammography_data/INbreast/calc_mask_ignore_single_point/*png'
        else:
            mask_search_path = r'/data1/Image_data/Mammography_data/INbreast/calc_mask/*png'
        valid_txt_path = r'/data1/Image_data/Mammography_data/INbreast/valid.txt'
        train_txt_path = r'/data1/Image_data/Mammography_data/INbreast/train.txt'
        output_dir = '/data1/Image_data/Mammography_data/log/calc_patches'
        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                               valid_txt_path=valid_txt_path,
                               train_txt_path=train_txt_path,
                               scale=1.0,
                               ignore_padding=0,
                               remove_zero_threshold=25).deploy()

    # for a smaller size datasize.
    if FLAGS.task == 'calc_crop_synth':
        image_search_path = r'/data1/Image_data/Mammography_data/INbreast/Calc_synthesis/20190711/synthesis_image/*png'
        if FLAGS.ignore_single_point:
            mask_search_path = r'/data1/Image_data/Mammography_data/INbreast/Calc_synthesis/20190711/synthesis_mask/*png'
        else:
            mask_search_path = r'/data1/Image_data/Mammography_data/INbreast/calc_mask/*png'
        valid_txt_path = r'/data1/Image_data/Mammography_data/INbreast/Calc_synthesis/20190711/valid.txt'
        train_txt_path = r'/data1/Image_data/Mammography_data/INbreast/Calc_synthesis/20190711/train.txt'
        output_dir = '/data1/Image_data/Mammography_data/INbreast/Calc_synthesis/20190711/calc_patches/'
        fileio.maybe_make_new_dir(output_dir)
        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                                 valid_txt_path=valid_txt_path,
                                 train_txt_path=train_txt_path,
                                 scale=1.0,
                                 ignore_padding=0,
                                 remove_zero_threshold=25).deploy()


    if FLAGS.task == 'calc_cluster_crop':
        image_search_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/calc_cluster/png/*png'
        mask_search_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/calc_cluster/bootstrap_mask_cleaned/*png'
        valid_txt_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/evaluation/valid.txt'
        train_txt_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/evaluation/train.txt'
        output_dir = '/data/log/mammo/calc_cluster_patches'
        # output_dir = '/data/log/mammo/calc_cluster_crop_affine_pos'
        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                               valid_txt_path=valid_txt_path,
                               train_txt_path=train_txt_path,
                               scale=1.0,
                               ignore_padding=0,
                               remove_zero_threshold=25).deploy()
    elif FLAGS.task == 'mass_crop':
        if FLAGS.dataset == 'inbreast':
            image_search_path = r'/media/Data/Data02/Datasets/Mammogram/INbreast/AllPNG_norm_6_6/*png'
            mask_search_path = r'/media/Data/Data02/Datasets/Mammogram/INbreast/mass_mask/*png'
            valid_txt_path = r'/media/Data/Data02/Datasets/Mammogram/INbreast/valid.txt'
            train_txt_path = r'/media/Data/Data02/Datasets/Mammogram/INbreast/train.txt'
        elif FLAGS.dataset == 'ddsm':
            image_search_path = r'/media/Data/Data02/Datasets/Mammogram/CBIS_DDSM/mass_mask/image_norm_6_6/*png'
            mask_search_path = r'/media/Data/Data02/Datasets/Mammogram/CBIS_DDSM/mass_mask/*png'
            valid_txt_path = None
            train_txt_path = None
        elif FLAGS.dataset == 'china':
            image_search_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/AllPNG_norm_6_6/*png'
            mask_search_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/mass_mask_etc/*png'
            valid_txt_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/valid.txt'
            train_txt_path = r'/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/train.txt'
        else:
            raise ValueError('Unsupported dataset {}'.format(FLAGS.mass_dataset))
        output_dir = '/data/log/mammo/mass_patches_area'
        MassOrCalcPatchConverter(image_search_path, mask_search_path, output_dir,
                               valid_txt_path=valid_txt_path,
                               train_txt_path=train_txt_path,
                               scale=FLAGS.scale,
                               dataset=FLAGS.dataset,
                               whole_image=FLAGS.whole_image,
                               ignore_padding=128,
                               remove_zero_threshold=100).deploy()
    else:
        raise ValueError('Unknown task: {}'.format(FLAGS.task))


