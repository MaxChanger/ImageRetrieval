"""This file contains utility functions used for numpy-based input augmentation

TODO: make more general for 3D images
"""

import logging
import cv2
import random
import scipy.ndimage
from skimage.filters import threshold_otsu, gaussian
try:
    import dicom
except:
    import pydicom as dicom
import numpy as np
from io import BytesIO
from skimage.morphology import erosion, square, disk
from skimage import measure

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def apply_mirroring(data, labels):
    """
    Apply mirroring to left, right, top, and bottom
    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (alphabet limited to (0, 1))

    Returns:
        Mirrored data and labels
    """
    data_shape = data.shape[0:2]
    data = np.lib.pad(data, ((data_shape[0]-1, data_shape[0]-1), (data_shape[1]-1, data_shape[1]-1)), 'reflect')
    labels = np.lib.pad(labels, ((data_shape[0]-1, data_shape[0]-1), (data_shape[1]-1, data_shape[1]-1), (0, 0)), 'reflect')
    return data, labels


def random_rotation(data, labels):
    """
    Perform random rotation on data and labels
    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (alphabet limited to (0, 1))

    Returns:
        Rotated data and labels
    """
    angle_deg = random.uniform(-180.0, 180.0)
    data = scipy.ndimage.interpolation.rotate(data, angle_deg, reshape=False)
    labels = scipy.ndimage.interpolation.rotate(labels, angle_deg, reshape=False)
    return data, labels


def random_flip_left_right(data, labels):
    """
    Perform random flip left and right on data and labels
    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (alphabet limited to (0, 1))

    Returns:
        Randomly flipped data and labels
    """
    flip = bool(random.getrandbits(1))
    if flip:
        data = np.fliplr(data)
        labels = np.fliplr(labels)
    return data, labels


def get_next_dividable_shape(input_shape, block_shape):
    """Get the minimum new_shape >= shape that is dividable by block_shape

    Args:
        input_shape: original shape
        block_shape: shape to be multiples of. Can be scalar, or a list with the same shape as input_shape

    Returns:
        new_shape
    """
    input_shape = np.array(input_shape)
    block_shape = np.array(block_shape)
    residual_shape = input_shape - (input_shape // block_shape) * block_shape
    # if residual_shape == (0, 0), do not change shape
    new_shape = input_shape + (block_shape - residual_shape) % block_shape
    return new_shape


def crop_or_pad(image_array, target_shape):
    """Crop or pad image_array to target_shape

    Use the top left corner (0, 0) as anchor and only pad or crop in the bottom right corner.

    NB: this only works for 2D for now

    Args:
        image_array:
        target_shape:

    Returns:

    """
    dtype = image_array.dtype
    source_shape = image_array.shape[:2]
    target_shape = target_shape[:2]
    if tuple(source_shape) == tuple(target_shape):
        return image_array
    max_shape = tuple(np.max([source_shape, target_shape], axis=0).astype(np.int))
    image_array_new = np.zeros(max_shape, dtype=dtype)
    image_array_new[:source_shape[0], :source_shape[1]] = image_array
    image_array_new = image_array_new[:target_shape[0], :target_shape[1]]
    assert tuple(image_array_new.shape) == tuple(target_shape)
    return image_array_new


def center_pad(image, target_shape, mode='constant'):
    """Pad image symmetrically to target_shape

    Args:
        image: input np array
        target_shape: final shape
        mode: np.pad mode

    """
    target_shape = np.asarray(target_shape)
    source_shape = np.array(image.shape)
    # top/left padding and bottom/right padding
    padding_1 = (target_shape - source_shape) // 2
    padding_2 = (target_shape - source_shape) - (target_shape - source_shape) // 2
    image = np.pad(image, list(zip(padding_1, padding_2)), mode=mode)
    assert image.shape == tuple(target_shape)
    return image


def center_crop(data, crop_shape, labels=None):
    """
    Perform random cropping after optional padding
    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (unique values limited to (0, 1)), could be None
        crop_shape: target shape after cropping

    Returns:
        Randomly cropped data and labels
    """
    data_shape = data.shape[0:2]
    assert (crop_shape[0] <= data_shape[0])
    assert (crop_shape[1] <= data_shape[1])

    nh = int((data_shape[0] - crop_shape[0]) / 2)
    nw = int((data_shape[1] - crop_shape[1]) / 2)
    data = data[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    if labels is not None:
        labels = labels[nh:nh + crop_shape[0], nw:nw + crop_shape[1], :]
        return data, labels
    return data


def center_crop_or_pad(image, target_shape, mode='constant'):
    """Center crop or pad to target_shape

    Only works for 2D images

    Args:
        image:
        target_shape:
        mode:

    Returns:

    """
    pad_target_shape = np.maximum(np.array(target_shape)[:2], image.shape[:2])
    image = center_pad(image, target_shape=pad_target_shape, mode=mode)
    image = center_crop(image, crop_shape=target_shape)
    return image


def crop_around_point(image_array, center_yx, target_shape):
    """Center crop an image array around a point

    Args:
        image_array:
        center_yx:
        target_shape:

    Returns:

    """
    pad_y, pad_x = ((np.array(target_shape) + 1) // 2).astype(np.int)
    pad_width = ((pad_y, pad_y), (pad_x, pad_x))
    image_array = np.pad(image_array, pad_width=pad_width, mode='constant')
    ymin, xmin = (np.array(center_yx) + np.array([pad_y, pad_x]) - np.array(target_shape) // 2).astype(np.int)
    ymax, xmax = (np.array([ymin, xmin]) + np.array(target_shape)).astype(np.int)
    cropped_array = image_array[ymin:ymax, xmin:xmax]
    assert cropped_array.shape == tuple(target_shape)
    return cropped_array


def random_crop(data, labels, crop_shape, padding=None):
    """
    Perform random cropping after optional padding
    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (alphabet limited to (0, 1))
        crop_shape: target shape after cropping
        padding: how many pixels to pad before cropping

    Returns:
        Randomly cropped data and labels
    """
    data_shape = data.shape[0:2]

    if padding:
        data_shape = (data_shape[0] + 2 * padding, data_shape[1] + 2 * padding)
        npad = ((padding, padding), (padding, padding), (0, 0))
        data = np.lib.pad(data, pad_width=npad, mode='constant', constant_values=0)
        labels = np.lib.pad(labels, pad_width=npad, mode='constant', constant_values=0)

    nh = random.randint(0, data_shape[0] - crop_shape[0])
    nw = random.randint(0, data_shape[1] - crop_shape[1])
    data = data[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    labels = labels[nh:nh + crop_shape[0], nw:nw + crop_shape[1], :]
    return data, labels


def random_resize(data, labels):
    """Perform random resizing

    Args:
        data: data array representing 1 image [h, w]
        labels: labels array [h, w, 2] (alphabet limited to (0, 1))

    Returns:
        Randomly resized data and labels, potentially with different shape
    """
    data_shape = data.shape[0:2]
    resize_ratio = np.random.uniform(low=1.0, high=1.2, size=2)

    data = scipy.ndimage.interpolation.zoom(input=data, zoom=resize_ratio)
    labels = scipy.ndimage.interpolation.zoom(input=labels, zoom=np.append(resize_ratio, 1.0))
    labels = np.around(labels)
    return data, labels


def resize(image, scale=None, dst_shape=(0, 0), interpolation=None):
    """Customize resize wrapper of cv2.resize

    Automatically select best interpolation method.
    Note: Pay special attention to the x and y dimension. Numpy uses (y, x) order but openCV
        uses (x, y) order.

    Args:
        image:
        scale:
        dst_shape: (x, y) order
        interpolation:
    """
    src_shape = np.asarray(image.shape) # in the order of (y, x, ...)
    src_shape = src_shape[:2][::-1] # get the first two dimension and flip them
    if scale is not None:
        dst_shape = (src_shape * scale).astype(np.int)
    else:
        dst_shape = np.asarray(dst_shape).astype(np.int)
    if interpolation is None:
        if (scale is not None and scale >= 1) or np.any(dst_shape > src_shape):
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
    image_resized = cv2.resize(image, tuple(dst_shape), interpolation=interpolation)
    return image_resized


def soft_rescale(data):
    """Soft scale data back to [0, 1]

    If data is in [0, 1], do nothing. Otherwise, scale the side outside this bound back to [0, 1]

    Args:
        data:

    Returns:

    """
    a_max = max(data.max(), 1)
    a_min = min(data.min(), 0)
    data = (data - a_min) / (a_max - a_min)
    return data


def random_brightness(data, labels, max_delta=0.2):
    """Perform random brightness adjustment. Add a random number to the image

    Args:
        data: a float array in [0, 1]
        labels:
        max_delta: maximum adjustment, in [-1, 1]

    Returns:

    """
    delta = np.random.uniform(low=-max_delta, high=max_delta)
    data = data + delta
    # scale back to [0, 1]
    data = soft_rescale(data)
    return data, labels


def random_contrast(data, labels, lower=0.8, upper=1.2):
    """Perform random contrast adjustment for 2d images
    For each `x` pixel in a channel, `(x - mean) * contrast_factor + mean`.

    Args:
        data: numpy array with values in [0, 1]
        labels:
        lower: lower bound of contrast adjustment, [0, 1]
        upper: upper bound of contrast adjustment, [1, inf]

    Returns:

    """
    contast_factor = np.random.uniform(low=lower, high=upper)
    mean = data.mean()
    data = (data - mean) * contast_factor + mean
    # scale back to [0, 1]
    data = soft_rescale(data)
    return data, labels


def pad(image_array, padding=(0, 0)):
    """Pad image with zero

    Args:
        image_array:
        padding:

    Returns:

    """
    shape = np.array(image_array.shape)
    new_shape = shape + 2 * padding
    image_array_padded = np.zeros(new_shape)
    image_array_padded[padding[0]:(shape[0] - padding[0]), padding[1]:(shape[1] - padding[1])] = image_array
    return image_array_padded


def normalize(image_array, a_min=-np.inf, a_max=np.inf,
              how='extend', lower_sigma=3, upper_sigma=6, bg_thresh=None, force_otsu=False):
    """Clip image and then normalize to [0, 1]

    Args:
        image_array: the input numpy array
        a_min:
        a_max:
        lower_sigma:
        upper_sigma:
        bg_thresh:
        how: the method to normalize, can be `optimize` or `extend`
            `optimize`: use automatic histogram normalization to optimize contrast
            `extend`: clip image and extend max to 1 and min to 0

    Returns:

    """
    image_array = image_array.astype(np.float)
    if how == 'optimize':
        image_array = normalize_auto(image_array, lower_sigma=lower_sigma, upper_sigma=upper_sigma,
                                     bg_thresh=bg_thresh, force_otsu=force_otsu)
    elif how == 'extend':
        if 0 < a_min < 1 and 0 < a_max < 1:
            if bg_thresh:
                image_array_fg = image_array[image_array > bg_thresh]
            a_max = np.percentile(image_array_fg, a_max)
            a_min = np.percentile(image_array_fg, a_min)
        image_array = np.clip(np.fabs(image_array), a_min, a_max)
        image_array -= np.amin(image_array)
        if np.amax(image_array) == 0:
            # uniform image
            return np.ones_like(image_array)
        image_array /= np.amax(image_array)
    elif how == 'raw':
        image_array /= 255
    else:
        raise ValueError('Unknown option {}'.format(how))
    return image_array


def normalize_auto(image_array, lower_sigma=2, upper_sigma=4, bg_thresh=None, bg_percentile=20, force_otsu=False):
    """Clip mammo to appropriate window

    Note: goals of this auto normalization algorithm:
        1. Set backgound to 0
        2. Maximize contrast while discarding minimum information
        3. Applying this function twice should yield the same results, i.e., this function should be idempotent,
            f(f(x)) = f(x). https://en.wikipedia.org/wiki/Idempotence#Unary_operation

    Args:
        image_array: the input numpy array
        bg_thresh: ignore pixel values < bg_thresh in calculating 6
        lower_sigma:
        upper_sigma:
        auto_norm: boolean, whether to use automated normalization

    Returns:
        image_array_clipped: a numpy array with range [0, 1]
    """
    # select the fg pixels
    image_array = image_array.astype(np.float)
    if not bg_thresh:
        # if 20 pct is 0, then background is 0, set bg_thresh = 0; otherwise use otsu to find bg_thresh
        if not force_otsu and np.percentile(image_array, bg_percentile) == 0:
            bg_thresh = 0
        else:
            bg_thresh = threshold_otsu(image_array)
    print('background threshold {}'.format(bg_thresh))
    image_array_fg = image_array[image_array > bg_thresh]
    # select 5 pct to 95 pct to perform robust normalization
    pct_5 = np.percentile(image_array_fg, 5)
    pct_95 = np.percentile(image_array_fg, 95)
    image_array_fg_robust = image_array_fg[(image_array_fg > pct_5) & (image_array_fg < pct_95)]
    std = np.std(image_array_fg_robust)
    mean = np.mean(image_array_fg_robust)
    # set (mean - lower_sigma * std) to 0, and (mean + upper_sigma * std) to 1
    a_min = mean - lower_sigma * std
    a_max = mean + upper_sigma * std
    # set bg pixels to a_min. Sometimes bg_threshold > a_min
    image_array[image_array <= bg_thresh] = a_min
    # clip
    image_array_clipped = np.clip(image_array, a_min=a_min, a_max=a_max)
    image_array_clipped = (image_array_clipped - a_min) / (a_max - a_min)
    return image_array_clipped


def binary_mask_to_probability_mask(image_array, ero_factor=4, blur_factor=12):
    """Convert binary mask to probability

    Erode first and then blur with a Gaussian kernel to largely constrain non-zero values within the original mask

    Args:
        image_array:
        ero_factor: erosion kernel size is 1/ero_factor of mask size (geometric averaged size)
        blur_factor: blurring kernel size is 1/blur_factor of mask size (two dimensional size)

    Returns:

    """
    assert len(np.unique(image_array)) <= 2, 'input is not binary array!'
    assert np.max(image_array) <= 1
    x, y = np.where(image_array == np.max(image_array))
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    size = np.asarray([xmax - xmin, ymax - ymin])
    if ero_factor:
        erosion_size = int(np.sqrt(size[0] * size[1]) / ero_factor)
        image_array = dilate(image_array, -erosion_size)
    if blur_factor:
        image_array = gaussian(image_array, sigma=size / blur_factor)
    return image_array


def dilate(binary_array, dilation_kernel_size):
    """

    Args:
        binary_array:
        dilation_kernel_size: an integer. Erode if negative, dilate if positive.
            This kernel size is diameter

    Returns:
        Dilated or eroded binary array
    """
    if dilation_kernel_size == 0:
        return binary_array
    if dilation_kernel_size < 0:
        dilation_kernel_size = -dilation_kernel_size
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        binary_array = cv2.erode(binary_array.astype(np.uint8), kernel, iterations=1)
    else:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        binary_array = cv2.dilate(binary_array.astype(np.uint8), kernel, iterations=1)
    binary_array = binary_array.astype(np.bool)
    return binary_array


def clean_bg_component(prob_array, threshold=0.5, anchor_patch=10):
    """Keep the central connected component and set the rest of the pixels to 0

    Note that the current algorithm requires that the ceneral pixel

    Args:
        prob_array: a probability array with values between 0 and 1
        anchor_patch: collect all non-zero labels in this center patch and find connected component.
            Sometimes the center of the patch does not lie in any foreground connected component mask,
            and this serves as a temporary solution.

    Returns:
        masked_prob_array: the masked prob array
    """
    binary_array = prob_array > threshold
    labeled_mask_array = measure.label(binary_array, connectivity=2)
    y_c, x_c = np.array(binary_array.shape) // 2
    central_component_idx = np.unique(
        labeled_mask_array[(y_c - anchor_patch // 2):(y_c + anchor_patch // 2),
        (x_c - anchor_patch // 2):(x_c + anchor_patch // 2)])
    central_component_idx = [x for x in central_component_idx if x]
    center_mask = np.zeros_like(binary_array)
    if not central_component_idx:
        # If cannot find central connected component, return the original input
        return prob_array, labeled_mask_array
    for idx in central_component_idx:
        center_mask[labeled_mask_array == idx] = 1
    masked_prob_array = prob_array * center_mask
    return masked_prob_array, labeled_mask_array


def opening(binary_array, open_kernel_size):
    """

    Args:
        binary_array:
        open_kernel_size(int): Closing if negative, opening if positive

    Returns:

    """
    if open_kernel_size == 0:
        return binary_array
    if open_kernel_size < 0:
        open_kernel_size = -open_kernel_size
        kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        binary_array = cv2.dilate(binary_array.astype(np.uint8), kernel, iterations=1)
        binary_array = cv2.erode(binary_array.astype(np.uint8), kernel, iterations=1)
    else:
        kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        binary_array = cv2.erode(binary_array.astype(np.uint8), kernel, iterations=1)
        binary_array = cv2.dilate(binary_array.astype(np.uint8), kernel, iterations=1)
    binary_array = binary_array.astype(np.bool)
    return binary_array


class DicomCorrector(object):
    def __init__(self, dicom_path, level):
        """Correct dicom grayscale according to LUT
        Ref: http://dicom.nema.org/medical/dicom/2017a/output/chtml/part03/sect_C.11.html
        There 3 stages or transforms within the DICOM rendering pipeline with regards to applying Lookup tables that
        can alter input values for rendering. Used within these stages are 4 types of lookup table (LUT) which can be
        found within DICOM images are part of the standard, and one further type which exists in DicomObjects. These
        together with a number of other Pixel data modifiers are used within the pipeline to produce a flexible
        rendering chain.

        Args:
            dicom_path:
            level: gamma correction levels, could be `softer`, `normal` or `harder`
        """
        self._dicom_path = dicom_path
        try:
            self._ds = dicom.read_file(dicom_path)
            self._image_array = self._ds.pixel_array
        except:
            print("Dicom reading error")
            # Add preamble manually
            fp = BytesIO()
            fp.write(b'\x00' * 128)
            fp.write(b'DICM')
            # Add the contents of the file
            f = open(dicom_path, 'rb')
            fp.write(f.read())
            f.close()
            fp.seek(0)
            # Read the dataset:
            self._ds = dicom.read_file(fp, force=True)
            import pydicom.uid
            self._ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
            self._image_array = self._ds.pixel_array
        self._level = level
        # original array before correction
        self._raw_array = self._image_array.copy()

    @staticmethod
    def look_up_value(lut_sequence, image_array, level):
        """

        Args:
            voi_lut_module: An element of VOILUTSequence
            image_array(np.array):

        Returns:

        """
        for i in range(lut_sequence.VM):
            lut_descriptor = lut_sequence[i][0x28, 0x3002].value
            lut_explanation = lut_sequence[i][0x28, 0x3003].value
            lut_data = lut_sequence[i][0x28, 0x3006].value
            num_entries = lut_descriptor[0]
            offset = lut_descriptor[1]
            if lut_explanation.lower() == level.lower():
                image_array = np.clip(image_array, offset, num_entries + offset - 1)
                image_array = image_array - offset
                lut = np.asarray(lut_data)
                image_array = lut[image_array.astype(int)]
                return image_array

    def voi_lut_windowing(self, win_center, win_width, fun_type="linear"):
        """VOI LUT function is LINEAR

        Args:
            win_center:
            win_width:
            fun_type:

        Returns:

        """
        assert fun_type.lower() in ['linear', 'sigmoid', 'linear_exact']
        print('Using windowing type `{}`'.format(fun_type))
        if fun_type.lower() == "linear":
            lower_bound = win_center - 0.5 - (win_width - 1) / 2
            upper_bound = win_center - 0.5 + (win_width - 1) / 2
            self._image_array[self._image_array <= lower_bound] = lower_bound
            self._image_array[self._image_array > upper_bound] = upper_bound
            self._image_array[(self._image_array > lower_bound) * (self._image_array <= upper_bound)] = \
                ((self._image_array[(self._image_array > lower_bound) * (self._image_array <= upper_bound)]
                  - (win_center - 0.5)) / (win_width - 1) + 0.5) * (upper_bound - lower_bound) + lower_bound
        elif fun_type.lower() == "sigmoid":
            bits_stored = self._ds[0x28, 0x101].value
            output_range = 2**bits_stored
            self._image_array = output_range / (1 + np.exp(-4*(self._image_array-win_center))/win_width)
        elif fun_type.lower() == "linear_exact":
            lower_bound = win_center - win_width / 2
            upper_bound = win_center + win_width / 2
            self._image_array[self._image_array <= lower_bound] = lower_bound
            self._image_array[self._image_array > upper_bound] = upper_bound
            self._image_array[(self._image_array > lower_bound)*(self._image_array<=upper_bound)] = \
                (self._image_array - win_center) / win_width * (upper_bound - lower_bound) + lower_bound

    def modality_lut(self):
        """This function transforms the manufacturer dependent pixel values into pixel values which are meaningful for
        the modality and which are manufacturer independent.
        Returns:

        """
        try:
            modality_lut_sequence = self._ds[0x28, 0x3000]
            self._image_array = DicomCorrector.look_up_value(modality_lut_sequence, self._image_array, self._level)
        except:
            try:
                print("Use rescaling to do the modality lut")
                intercept = self._ds[0x28, 0x1052].value
                slope = self._ds[0x28, 0x1053].value
                self._image_array = self._image_array * slope + intercept
            except:
                print("Unable to do the modaligy lut", self._dicom_path)

    def voi_lut(self):
        """The Value Of Interest(VOI) LUT transformation transforms the modality pixel values into pixel values which
        are meaningful for the user or the application.

        Args:
            level:

        Returns:

        """
        try:
            voi_lut_sequence = self._ds[0x28, 0x3010]
            self._image_array = DicomCorrector.look_up_value(voi_lut_sequence, self._image_array, self._level)
        except:
            print("Render the gray scale into window")
            try:
                win_center_list = self._ds[0x28, 0x1050].value
                win_width_list = self._ds[0x28, 0x1051].value
                if self._ds[0x28, 0x1051].VM == 1:
                    win_center = win_center_list
                    win_width = win_width_list
                else:
                    zipped = zip(win_width_list, win_center_list)
                    zipped_sorted = sorted(zipped, key=lambda t: t[0])
                    if self._level.lower() == "softer":
                        win_center = zipped_sorted[0][1]
                        win_width = zipped_sorted[0][0]
                    elif self._level.lower() == "normal":
                        win_center = zipped_sorted[1][1]
                        win_width = zipped_sorted[1][0]
                    elif self._level.lower() == "harder":
                        win_center = zipped_sorted[2][1]
                        win_width = zipped_sorted[2][0]
                    else:
                        raise KeyError("Input level error, should be softer, normal or harder")
                print('Level `{}` not found'.format(self._level))
                try:
                    function_type = self._ds[0x28, 0x1056]
                    self.voi_lut_windowing(win_center, win_width, function_type)
                except:
                    self.voi_lut_windowing(win_center, win_width)
            except:
                print("Unable to do the voi lut", self._dicom_path)

    def presentation_lut(self):
        """The Presentation LUT transformation transforms the pixel values into P-Values, a device independent perceptually
        linear space.

        Returns:

        """
        try:
            presentation_lut_shape = self._ds[0x2050, 0x0020].value
            if presentation_lut_shape.lower == "invert" or presentation_lut_shape.lower() == "inverse":
                self._image_array = np.max(self._image_array) - self._image_array
        except:
            print("Use photometric interpretation to check invert", self._dicom_path)
            try:
                photometric_interpretation = self._ds[0x28, 0x4].value
                if photometric_interpretation == "MONOCHROME1":
                    self._image_array = np.max(self._image_array) - self._image_array
            except:
                print("Unable to do the presentation lut", self._dicom_path)

    def correct(self):
        """The main function to correct dicom pixel_value error.

        Returns:

        """
        try:
            self.modality_lut()
            self.voi_lut()
            self.presentation_lut()
            return self._image_array
        except:
            print("Correction Error", self._dicom_path)
