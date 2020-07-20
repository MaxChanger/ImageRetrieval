"""This file contains utility functions used for bbox"""

import logging
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage import measure

# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 150

from projects.drutils import fileio
# from projects.drutils import patch  #when doing froc in object detector
from projects.drutils import augmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def calc_intersection(bbox_1, bbox_2):
    """Calculate the IOU (Intersection Over Union) given two set of bbox coordinates

    All coordinates are in the order of (ymin, xmin, ymax, xmax), following the convention in `tf_example_decoder.py`

    Args:
        bbox_1: coordinates of bbox1
        bbox_2: coordinates of bbox2

    Returns:
        x_intersection: the intersected x length
        y_intersection: the intersected y length
    """
    assert len(bbox_1) == len(bbox_2)
    if len(bbox_1) == 4:
        ymin, xmin, ymax, xmax = bbox_1
        ymin_2, xmin_2, ymax_2, xmax_2 = bbox_2
        assert ymax >= ymin and ymax_2 >= ymin_2 and xmax >= xmin and xmax_2 >= xmin_2, 'Please check coordinate input!'

        x_intersection = max(min(xmax, xmax_2) - max(xmin, xmin_2), 0)
        y_intersection = max(min(ymax, ymax_2) - max(ymin, ymin_2), 0)
        return (x_intersection, y_intersection)
    elif len(bbox_1) == 2:
        xmin, xmax = bbox_1
        xmin_2, xmax_2 = bbox_2
        assert xmax >= xmin and xmax_2 >= xmin_2, 'Please check coordinate input!'

        x_intersection = max(min(xmax, xmax_2) - max(xmin, xmin_2), 0)
        return x_intersection
    else:
        raise ValueError('Input bbox size must be 2 or 4.')


def calc_iou_1d(bbox_1, bbox_2):
    """Calculate the 1-dimensional IOU (Intersection Over Union) given two set of bbox coordinates

    All coordinates are in the order of (ymin, xmin, ymax, xmax), following the convention in `tf_example_decoder.py`

    Args:
        bbox_1: coordinates of bbox1
        bbox_2: coordinates of bbox2

    Returns:
        iou: the intersection over union
    """
    xmin, xmax = bbox_1
    xmin_2, xmax_2 = bbox_2
    assert xmax >= xmin and xmax_2 >= xmin_2, 'Please check coordinate input!'

    x_intersection = calc_intersection(bbox_1, bbox_2)
    area_intersection = x_intersection
    area_1 = xmax - xmin
    area_2 = xmax_2 - xmin_2
    area_union = area_1 + area_2 - area_intersection
    if area_union == 0:
        return 0
    iou = area_intersection / area_union

    return iou


def calc_iou(bbox_1, bbox_2):
    """Calculate the IOU (Intersection Over Union) given two set of bbox coordinates

    All coordinates are in the order of (ymin, xmin, ymax, xmax), following the convention in `tf_example_decoder.py`

    Args:
        bbox_1: coordinates of bbox1
        bbox_2: coordinates of bbox2

    Returns:
        iou: the intersection over union
    """
    ymin, xmin, ymax, xmax = bbox_1
    ymin_2, xmin_2, ymax_2, xmax_2 = bbox_2
    assert ymax > ymin and ymax_2 > ymin_2 and xmax > xmin and xmax_2 > xmin_2, 'Please check coordinate input!'

    x_intersection, y_intersection = calc_intersection(bbox_1, bbox_2)
    area_intersection = x_intersection * y_intersection
    area_1 = (xmax - xmin) * (ymax - ymin)
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)
    area_union = area_1 + area_2 - area_intersection
    iou = area_intersection / area_union

    return iou


def calc_avg_iou(predictions, labels):
    """Compute average bbox IOU over a number of batches

    Args:
        predictions: Predictions matrix [num_batches, batch_size, 4 or 2]
        labels: Labels matrix [num_batches, batch_size, 4 or 2]

    Returns:
        average IOU value
    """
    num_batches, batch_size, num_coordinates = predictions.shape
    assert num_coordinates == 4 or num_coordinates == 2

    iou_per_sample = np.zeros((num_batches, batch_size))
    for batch_id in range(num_batches):
        for image_id in range(batch_size):
            pred_array = np.array(predictions[batch_id, image_id, :])
            label_array = np.array(labels[batch_id, image_id, :])
            if num_coordinates == 4:
                iou_per_sample[batch_id, image_id] = calc_iou(pred_array, label_array)
            elif num_coordinates == 2:
                iou_per_sample[batch_id, image_id] = calc_iou_1d(pred_array, label_array)

    return np.mean(iou_per_sample)


def calc_ios(bbox_1, bbox_2):
    """Calculate intersection over small ratio

    This is a variant of more commonly used IoU (intersection over union) metric

    All coordinates are in the order of (ymin, xmin, ymax, xmax)

    Args:
        bbox_1:
        bbox_2:

    Returns:

    """

    def cal_area(bbox):
        # calculate the area for a bbox in format (y_min, x_min, y_max, x_max)
        return max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)

    ymin_1, xmin_1, ymax_1, xmax_1 = bbox_1
    ymin_2, xmin_2, ymax_2, xmax_2 = bbox_2
    x_min = max(xmin_1, xmin_2)
    y_min = max(ymin_1, ymin_2)
    x_max = min(xmax_1, xmax_2)
    y_max = min(ymax_1, ymax_2)
    area_intersection = cal_area([y_min, x_min, y_max, x_max])
    area_small = min(cal_area(bbox_1), cal_area(bbox_2))
    ios = area_intersection / area_small
    return ios


def is_overlapped(pred_bbox, gt_bbox,
                  overlap_method='gt_center_in_pred',
                  iou_threshold=0, relative_lateral_threshold=0.9):
    """To tell if there is overlap between perd_bbox and gt_bbox

    TODO: 3D version in evaluation tools sigmaLU
    TODO: switching between 2D/3D based on the input shape
    TODO: add assertion of cases tested

    All coordinates are in the of (ymin, xmin, ymax, xmax), following the convention in `tf_example_decoder.py`

    Args:
        pred_bbox: four predicted coordinates
        gt_bbox: four GT coordinates
        overlap_method: defaults to `gt_center_in_pred`, could be
            gt_center_in_pred:
            pred_center_in_gt:
            iou:
            relative_lateral_overlap:
    Return:
        is_overlapped: True if is_overlapped, False otherwise
    """
    ymin, xmin, ymax, xmax = pred_bbox
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = gt_bbox
    xcenter = (xmin + xmax) // 2
    ycenter = (ymin + ymax) // 2
    gt_xcenter = (gt_xmin + gt_xmax) // 2
    gt_ycenter = (gt_ymin + gt_ymax) // 2
    if overlap_method == 'gt_center_in_pred':
        is_overlapped = (xmin <= gt_xcenter <= xmax and ymin <= gt_ycenter <= ymax)
    elif overlap_method == 'pred_center_in_gt':
        is_overlapped = (gt_xmin <= xcenter <= gt_xmax and gt_ymin <= ycenter <= gt_ymax)
    elif overlap_method == 'either_center_in_other':
        is_overlapped = ((xmin <= gt_xcenter <= xmax and ymin <= gt_ycenter <= ymax) or
                         (gt_xmin <= xcenter <= gt_xmax and gt_ymin <= ycenter <= gt_ymax))
    elif overlap_method == 'both_center_in_other':
        is_overlapped = ((xmin <= gt_xcenter <= xmax and ymin <= gt_ycenter <= ymax) and
                         (gt_xmin <= xcenter <= gt_xmax and gt_ymin <= ycenter <= gt_ymax))
    elif overlap_method == 'iou':
        is_overlapped = (calc_iou(pred_bbox, gt_bbox) >= iou_threshold)
    elif overlap_method == 'relative_lateral_overlap':
        x_intersection, y_intersection = calc_intersection(pred_bbox, gt_bbox)
        is_overlapped = (x_intersection / min(xmax - xmin, gt_xmax - gt_xmin) > relative_lateral_threshold) and (
                         y_intersection / min(ymax - ymin, gt_ymax - gt_ymin) > relative_lateral_threshold)
    else:
        raise ValueError('Unknown overlap_method {}'.format(overlap_method))

    return is_overlapped


def is_bbox_list_overlapped(pred_bbox_list, gt_bbox_list, gt_class_list=[], **kwargs):
    """Check if the input bb overlaps with the list of groundtruth (GT) bbox.

    Args:
      pred_bbox_list:
      gt_bbox_list:
      gt_class_list:

    Return:
      is_pred_bbox_correct_list: a list of set containing the index of overlapped GT bbox
      is_gt_bbox_covered_list: a list of set containing the index of overlapped pred bbox
    """
    is_pred_bbox_correct_list = np.array([set() for _ in range(len(pred_bbox_list))])
    is_gt_bbox_covered_list = np.array([set() for _ in range(len(gt_bbox_list))])

    if not gt_class_list:
        gt_class_list = [-1 for _ in range(len(gt_bbox_list))]
    assert len(gt_class_list) == len(gt_bbox_list)
    for idx_gt, (gt_bbox, gt_class) in enumerate(zip(gt_bbox_list, gt_class_list)):
        for idx_pred, pred_bbox in enumerate(pred_bbox_list):
            if is_overlapped(pred_bbox, gt_bbox, **kwargs):
                is_gt_bbox_covered_list[idx_gt].add(idx_pred) # TODO: change this to pred classes
                is_pred_bbox_correct_list[idx_pred].add(idx_gt)

    # is_pred_bbox_correct_list = is_pred_bbox_correct_list.astype(bool)
    # is_gt_bbox_covered_list = is_gt_bbox_covered_list.astype(bool)

    return is_pred_bbox_correct_list, is_gt_bbox_covered_list


def draw_bb_and_test(image, rects, label='', color=(0, 255, 0),
                     line_thickness=2, font_scale=0.5, font_thickness=2):
    """ 
    Draw bounding box and add text to image

    Args:
      image: input image
      rects: bbox coordinates in the order of (x, y, w, h)
      label: label text to annotate bbox
      color: label color
      line_thickness: bbox line thickness
      font_scale: label text scale factor
      font_thickness: label text thickness

    Returns:
      image: the modified image
    """
    assert len(image.shape) == 3, 'images are expected to have 3 channels'
    x, y, w, h = rects
    # loop over the cat faces and draw a rectangle surrounding each
    cv2.rectangle(image, (x, y), (x + w, y + h), color, line_thickness)
    cv2.putText(image, "{}".format(label), (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    return image


def overlay_bbox_on_grayscale_image(image, rects, label='', color=(0, 255, 0), order='xywh', **kwargs):
    """
    Convert grayscale image to RGB and then overlay bbox

    Args:
        image: grayscale image
        rects: bbox coordinates in the order of (x, y, w, h)
        order: order of bbox, 'xywh' or 'yminxmin'
    Return:
        image: the overlaid image with num_channels
    """
    line_thickness = kwargs.get('line_thickness', 5)
    font_scale = kwargs.get('font_scale', 2)
    font_thickness = kwargs.get('font_thickness', 5)
    if len(image.shape) == 2:
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.dstack([image] * 3)
    assert len(image.shape) == 3, 'image must have 3 channels'
    assert len(color) == 3, 'color must have 3 channels'

    if order == 'yminxmin':
        rects = convert_bbox_yminxmin_to_xywh(rects)

    # Draw bbox and annotation
    image = draw_bb_and_test(image, rects=rects,
                             label=label, color=color,
                             line_thickness=line_thickness,
                             font_scale=font_scale,
                             font_thickness=font_thickness)
    return image


def overlay_bbox_list_on_image(image, rects_list, label_list=[], color=(0, 255, 0), order='yminxmin', **kwargs):
    """Overlay a list of bboxes on top of an image

    Args:
        image:
        rects_list:
        label_list:
        color:

    Returns:
        image: rgb image with bbox overlay
    """
    n_bbox = len(rects_list)
    if len(label_list) == 0:
        label_list = ['' for _ in range(n_bbox)]
    elif len(label_list) == 1:
        label = label_list[0]
        label_list = [label for _ in range(n_bbox)]
    else:
        assert len(rects_list) == len(label_list), 'number of labels should agree with number of bboxes!'
    if len(image.shape) == 2:
        image = np.dstack([image] * 3)
    for idx, (rects, label) in enumerate(zip(rects_list, label_list)):
        if not rects:
            continue
        image = overlay_bbox_on_grayscale_image(image, rects, label=label, color=color, order=order, **kwargs)
    return image


def create_mask_with_bbox_list(image, rects_list):
    """Create binary mask array with bbox list

    This function fills the inside of a bbox.
    It is different from overlay_bbox_list_on_image which only shows the edge of the bbox

    Args:
        image: numpy array of the canvas
        rects_list: list of int in the order of (x, y, w, h)

    """
    for x, y, w, h in rects_list:
        # loop over the cat faces and draw a rectangle surrounding each
        # use thickness = -1 to fill the inside
        cv2.rectangle(image, (x, y), (x + w, y + h), color=255, thickness=-1)
    return image


def show_bbox(bbox_dict, raw_image_path_dict, image_pred_path_dict=None, n_demo=0, output_dir='', gt_only=False):
    """Overlay bbox coordinate to original image and show GT and pred side by side.

    Prediction overlay uses prediction probability map if image_pred_path_dict is not None, otherwise use raw image

    Args:
        bbox_dict: a dict with image name as key. Each key corresponds to another dict with the following keys
            'pred_bbox_list': input to is_bbox_list_overlapped()
            'gt_bbox_list': input to is_bbox_list_overlapped()
            'pred_box_correct': output of is_bbox_list_overlapped()
            'gt_box_covered': output of is_bbox_list_overlapped()
        raw_image_path_dict: a dict with image name as key. The corresponding value is the path to the raw image to
            overlay
        image_pred_path_dict: optional, a dict with image name as key. The corresponding value is the path to the pred
            results to overlay. Default to None, and if specified, use it to showcase pred result on the RHS of the
            stack image
        n_demo: number of times to run demo
        gt_only: boolean, whether to show gt only (no prediction) <TODO> Not tested yet

    Returns:
        None
    """
    colors = {
        "tp": (0, 255, 0), # green
        "fp": (255, 0, 0), # blue in BGR
        "fn": (0, 0, 255), # red in BGR
    }
    for idx, key in enumerate(bbox_dict.keys()):
        if key.startswith('_'):
            continue
        pred_bbox_list = bbox_dict[key]['pred_bbox_list']
        is_pred_bbox_correct_list = bbox_dict[key]['pred_box_correct']
        gt_bbox_list = bbox_dict[key]['gt_bbox_list']
        is_gt_bbox_covered_list = bbox_dict[key]['gt_box_covered']
        bbox_list_tp = [bbox for bbox, bool in zip(pred_bbox_list, is_pred_bbox_correct_list) if bool]
        bbox_list_fp = [bbox for bbox, bool in zip(pred_bbox_list, is_pred_bbox_correct_list) if not bool]
        bbox_list_fn = [bbox for bbox, bool in zip(gt_bbox_list, is_gt_bbox_covered_list) if not bool]
        bbox_list_tp_gt = [bbox for bbox, bool in zip(gt_bbox_list, is_gt_bbox_covered_list) if bool]

        image_path = raw_image_path_dict[key]
        image = fileio.load_image_to_array(image_path, np.uint8)
        if image_pred_path_dict:
            image_pred_path = image_pred_path_dict[key]
            # this can be a list of up to 3 elements to populate BGR channels
            if isinstance(image_pred_path, (list, tuple)) and len(image_pred_path) > 1:
                image_overlay = np.dstack([image] * 3)
                for idx_ch, single_pred_path in enumerate(image_pred_path):
                    logging.debug('assembling channel {}'.format(idx_ch))
                    image_pred = fileio.load_image_to_array(single_pred_path, np.uint8)
                    # generate overlay in green channel (low prob in magenta color)
                    logging.debug('before crop_or_pad {} {}'.format(image_pred.shape, image.shape))
                    image_pred = augmentation.crop_or_pad(image_pred, image.shape)
                    logging.debug('after crop_or_pad {} {}'.format(image_pred.shape, image.shape))
                    image_proba = np.where(image_pred > 0, image_pred, image) # as a single channel
                    image_overlay[:, :, idx_ch] = image_proba
                image_pred = image_overlay
            else:
                image_pred = fileio.load_image_to_array(image_pred_path, np.uint8)
                # generate overlay in green channel (low prob in magenta color)
                logging.debug('before crop_or_pad {} {}'.format(image_pred.shape, image.shape))
                image_pred = augmentation.crop_or_pad(image_pred, image.shape)
                logging.debug('after crop_or_pad {} {}'.format(image_pred.shape, image.shape))
                image_proba = np.where(image_pred > 0, image_pred, image) # as a single channel
                image_overlay = np.dstack([image, image_proba, image])
                image_pred = image_overlay
        else:
            image_pred = image
        image_overlay_pred = overlay_bbox_list_on_image(image_pred, bbox_list_tp, color=colors['tp'])
        image_overlay_pred = overlay_bbox_list_on_image(image_overlay_pred, bbox_list_fp, color=colors['fp'])
        image_overlay_gt = overlay_bbox_list_on_image(image, bbox_list_tp_gt, color=colors['tp'])
        image_overlay_gt = overlay_bbox_list_on_image(image_overlay_gt, bbox_list_fn, color=colors['fn'])
        if idx < n_demo:
            fig, ax = plt.subplots(1, 2, figsize=(16, 10))
            ax = np.atleast_2d(ax)
            ax[0, 0].imshow(image_overlay_gt)
            ax[0, 1].imshow(image_overlay_pred)
            plt.show()
        # stack image and image_overlay side by side
        # image_rgb = np.dstack([image] * 3)
        logging.info('Processing key: {}'.format(key))
        if output_dir:
            if not gt_only:
                image_stack = np.hstack([image_overlay_gt, image_overlay_pred])
            else:
                image_stack = image_overlay_gt
            image_stack_path = os.path.join(output_dir, os.path.basename(image_path))
            fileio.maybe_make_new_dir(output_dir)
            cv2.imwrite(image_stack_path, image_stack)
        else:
            logging.warning('No output_dir specified. Skip key: {}'.format(key))


def convert_bbox_xywh_to_yminxmin(cv2_rects):
    """Convert cv2_rects (x, y, w, h) to bbox_coord (ymin, xmin, ymax, xman)

    Args:
        cv2_rects:

    Returns:
        bbox_coord
    """
    x, y, w, h = cv2_rects
    ymin, xmin, ymax, xmax = y, x, y + h, x + w
    tf_bbox_coord = (ymin, xmin, ymax, xmax)
    return tf_bbox_coord


def convert_bbox_yminxmin_to_xywh(tf_bbox_coord):
    """Convert tf_bbox_coord (ymin, xmin, ymax, xman) to cv2_rects (x, y, w, h)

    Args:
        tf_bbox_coord:

    Returns:
        cv2_rects
    """
    ymin, xmin, ymax, xmax = tf_bbox_coord
    y, x, h, w = ymin, xmin, ymax - ymin, xmax - xmin
    cv2_rects = (x, y, w, h)
    return cv2_rects


def save_annotated_image(image, filename, label='', newdir=''):
    """ Save image to newdir with the same basename in filename """
    if newdir:
      basename = os.path.basename(filename)
      if label:
          file, ext = os.path.splitext(basename)
          basename = file + '_' + label + ext
      newfilename = os.path.join(newdir, basename)
    else:
      newfilename = filename
    dirname = os.path.dirname(newfilename)
    fileio.maybe_make_new_dir(dirname)
    # print(newfilename)
    cv2.imwrite(newfilename, image)


def imshow_patch(image, rects, margin=200, origin='upper'):
    """
    Show image patch of interest 

    Args:
      image:
      rects:
      margin:
      origin:

    Returns:
      image_path:
    """
    x, y, w, h = rects
    height, width = image.shape[:2]
    xmin, xmax = max(0, x-margin), min(x+w+margin, width)
    ymin, ymax = max(0, y-margin), min(y+h+margin, height)
    image_patch = image[ymin:ymax, xmin:xmax]
    plt.gcf()
    if len(image.shape) == 2:
        plt.imshow(image, origin=origin, cmap='gray')
    else:
        plt.imshow(image, origin=origin)
    plt.xlim((xmin, xmax))
    if origin == 'upper':
        ymin, ymax = ymax, ymin
    plt.ylim((ymin, ymax))

    return image_patch


def get_largest_bbox_coord_from_pixel_array(image_array, max_val=255, threshold=127, min_cnt_size=None):
    """
    Convert binary bbox to bbox coordinates.

    Note that this assumes that the script only gets the LARGEST connected component in image_array.
    The use of this function should be DEPRECATED. For general purpose conversion, use
    `get_bbox_coord_list_from_binary_array` instead.

    Args:
        mask_array: input binary mask in the form of a numpy array
        max_val: optional, default to 255 (8-bit)
        threshold: optional, defaults to 127
        min_cnt_size: if not None, assert contour size > min_cnt_size
    Returns:
        rects: bbox coordinates in the order of (x, y, w, h)
    """
    # Get contours and return the largest
    ret, thresh = cv2.threshold(image_array, threshold, max_val, 0)
    image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = max(contours, key=cv2.contourArea)
    if min_cnt_size is not None:
        assert cv2.contourArea(cnt) > 100, 'max contour are is less than 100 pixels'

    # Get coordinates for minimum circumscribing bounding box
    rects = cv2.boundingRect(cnt)
    return rects


def get_bbox_coord_list_from_binary_array(binary_array,
                                          ignore_pred_under=0,
                                          dilation_kernel_size=0,
                                          bbox_dilation_ratio=1,
                                          bbox_dilation_size=0):
    """Convert a binary array to a list of bbox coordinates in the order (ymin, xmin, ymax, xmax)

        Connected component analysis with output = cv2.connectedComponentsWithStats():

        Labels = output[1] is an array with the same shape as the input binary array, with each component
            labeled with a different integer (BG is 0).

        Stats = output[2] is a matrix of the stats that the function calculates. It has a length equal
        to the number of labels and a width equal to the number of stats.

        It can be used with the OpenCV documentation for it:

        Statistics output for each label, including the background label, see below for available statistics.
        Statistics are accessed via stats[label, COLUMN] where available columns are defined below.

        cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box
            in the horizontal direction.
        cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box
            in the vertical direction.
        cv2.CC_STAT_WIDTH The horizontal size of the bounding box
        cv2.CC_STAT_HEIGHT The vertical size of the bounding box
        cv2.CC_STAT_AREA The total area (in pixels) of the connected component

    Args:
        binary_array: objects are marked with 1
        ignore_pred_under: pixel count threshold below which to discard the predicted component
        dilation_kernel_size: size of kernel to dilate the binary mask with
        bbox_dilation_ratio: bbox_new = bbox * bbox_dilation_ratio + bbox_dilation_size
        bbox_dilation_size: bbox_new = bbox * bbox_dilation_ratio + bbox_dilation_size

    Returns:
        bbox_coord_list:
        area_list: a list of connected component areas
    """

    assert binary_array.dtype == np.bool
    binary_array = binary_array.astype(np.uint8)
    # dilate binary mask to connect neightboring components
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        binary_array = cv2.dilate(binary_array, kernel, iterations=1)
    # connected component analysis
    bbox_coord_list = []
    area_list = []
    output = cv2.connectedComponentsWithStats(binary_array)
    stats = output[2]
    for idx, stat in enumerate(stats):
        # in the order of (x, y, w, h, area)
        x, y, w, h, area = stat
        # skip background component (always the first row)
        if idx == 0:
            if not (x == 0 and y == 0):
                logging.warning(
                    'The background component did not start at top left corner but at x={}, y={}!'.format(x, y))
            continue
        if area < ignore_pred_under:
            continue
        # dilate bbox
        x_center = x + w // 2
        y_center = y + h // 2
        w = w * bbox_dilation_ratio + 2 * bbox_dilation_size
        h = h * bbox_dilation_ratio + 2 * bbox_dilation_size
        x = x_center - w // 2
        y = y_center - h // 2
        ymin, xmin, ymax, xmax = y, x, y + h, x + w
        # convert to integers
        ymin, xmin, ymax, xmax = [int(item) for item in (ymin, xmin, ymax, xmax )]
        bbox_coord_list.append((ymin, xmin, ymax, xmax))
        area_list.append(area)
    if area_list:
        # sort by area_list in descending order, the largest bbox is bbox_coord_list[0]
        area_list, bbox_coord_list = list(zip(*sorted(zip(area_list, bbox_coord_list), reverse=True)))
    return bbox_coord_list, area_list


def get_bbox_coord_for_largest_cc_in_binary_array(binary_array, **kwargs):
    bbox = get_bbox_coord_list_from_binary_array(binary_array, **kwargs)[0][0]
    return bbox


def get_largest_foreground_mask(image_array, background_value='auto'):
    """Find the largest foreground connected component

    Connected component anlaysis with output = cv2.connectedComponentsWithStats():

        Labels = output[1] is an array with the same shape as the input binary array, with each component
            labeled with a different integer (BG is 0).

    Args:
        image_array: binary array where background is 0

    Returns:
        fg_mask_array: boolean numpy array. True for largest foreground connected component
    """
    if background_value == 'auto':
        # set to 20 percentile of the image
        lower_clip = np.percentile(image_array, 5)
        upper_clip = np.percentile(image_array, 30)
        if np.abs(upper_clip - lower_clip) / np.max(image_array) < 0.02:
            background_value = upper_clip
        else:
            logging.warning('difference 5th and 30th percentile is {}\nManually inspect this image'.format(
                np.abs(upper_clip - lower_clip)))
            background_value = lower_clip
    else:
        assert isinstance(background_value, np.int)
    binary_array = image_array > background_value
    output = cv2.connectedComponentsWithStats(binary_array.astype(np.uint8))
    stats = output[2]
    if len(stats) > 1 :
        # if there are at least two components returned
        # find the idx of the largest fg component by area (excluding 0th row, i.e., the BG)
        idx = np.argmax(stats[1:, -1]) + 1
        fg_mask_array = (output[1] == idx)
    else:
        logging.debug('Only one component in the image. Check raw image!')
        fg_mask_array = None
    return fg_mask_array


def get_ar(bbox):
    """Get aspect ratio of bbox"""
    ymin, xmin, ymax, xmax = bbox
    width, height = xmax - xmin, ymax - ymin
    ar = max(width, height) / min(width, height)
    return ar


def large_ar_suppression(boxes, ar_threshold=2):
    """Filter out bbox with aspect ratio larger than ar_threshold"""
    return [bbox for bbox in boxes if get_ar(bbox) <= ar_threshold]


def get_minmax_size(bbox):
    """Get aspect ratio of bbox"""
    ymin, xmin, ymax, xmax = bbox
    width, height = xmax - xmin, ymax - ymin
    min_size = min(width, height)
    max_size = max(width, height)
    return min_size, max_size


def non_max_suppression_fast(boxes, threshold=0.5, option='union', max_iterations=1):
    """ NMS to combine bboxes

    Adapted from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        boxes: in the order of (ymin, xmin, ymax, xmax)
        overlapThresh:
        option: method to postprocess the bbox coordinates
            'union': find the bbox for the union of the overlapping boxes
            'original': find the original bbox, from right to left

    Returns:

    """
    def concate_list(arrays, concateidx):
        """method to help track the resource of combined bounding boxes

       Args:
           arrays: list of list, represent the indices
           concateidx: indices of list to be merged

       Returns: merged flat list

        """


        result = []
        for idx in concateidx:
            result.extend(arrays[idx])
        return result

    merged_boxes_sources = [[i] for i in list(range(len(boxes)))]

    for i_iter in range(max_iterations):
        num_bbox_before_nms = len(boxes)
        # if there are no boxes, return an empty list
        if num_bbox_before_nms == 0:
            return [], []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        boxes = np.array(boxes).astype("float")

        # grab the coordinates of the bounding boxes
        # x1, y1 == xmin, ymin
        # x2, y2 == xmax, ymax
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        merged_boxes = []
        new_merged_boxes_sources = []
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            # NB. use the area of the moving box as overlap denominator
            # <TODO> add union area calculation
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list
            idxs_idx_to_delete = np.concatenate(([last],
                                                 np.where(overlap > threshold)[0]))
            if option == 'union':
                # return the bbox of the union
                xx1 = np.min(x1[idxs[idxs_idx_to_delete]])
                yy1 = np.min(y1[idxs[idxs_idx_to_delete]])
                xx2 = np.max(x2[idxs[idxs_idx_to_delete]])
                yy2 = np.max(y2[idxs[idxs_idx_to_delete]])
                merged_boxes.append((yy1, xx1, yy2, xx2))
                # merged_boxes_sources.append(idxs[idxs_idx_to_delete])
                new_merged_boxes_sources.append(concate_list(merged_boxes_sources, idxs[idxs_idx_to_delete]))

            elif option == 'original':
                merged_boxes.append(boxes[i])
                merged_boxes_sources.append(i)
            else:
                raise ValueError('Unsupported option {}'.format(option))
            idxs = np.delete(idxs, idxs_idx_to_delete)

        merged_boxes = np.array(merged_boxes).astype(np.int)
        # the original bbox coord
        boxes = merged_boxes
        merged_boxes_sources = new_merged_boxes_sources
        num_bbox_after_nms = len(boxes)
        # no bbox has been merged in this iteration
        if num_bbox_before_nms == num_bbox_after_nms:
            logging.debug('Finish NMS at {} out of {} requested iterations'.format(i_iter + 1, max_iterations))
            return boxes, merged_boxes_sources

    return boxes, merged_boxes_sources


def calculate_union_area(boxes):
    """ calculate the union area of several bounding boxes

    Args:
        boxes: list of bounding boxes, each one in the order of (ymin, xmin, ymax, xmax)

    Returns: union area

    """
    # convert to np array if the input is a list
    boxes = np.array(boxes)
    width = max(boxes[:, 3])
    height = max(boxes[:, 2])
    canvas = np.zeros([width + 1, height + 1])
    for i in range(len(boxes)):
        canvas[boxes[i, 1]:boxes[i, 3] + 1, boxes[i, 0]:boxes[i, 2] + 1] = 1
    return np.sum(canvas)


def _get_valid_length(line_scan):
    """Helper function for calculating valid length in one line_scan.

    Used in calculate_union_area_v2
    """
    sum_length = 0
    acc = 0
    last_x = 0
    for current_x in sorted(line_scan):
        if acc > 0:
            sum_length += current_x - last_x
        acc += line_scan[current_x]
        last_x = current_x
    return sum_length


def calculate_union_area_v2(boxes):
    """Calculate the union area of several bounding boxes

    This algorithm is inspired by numerical integration.
    Scan a line through the whole image. Calculate the 'valid length (height)'
    of each scanning position, and the intervals (width) during which the
    'valid length' stays the same.

    Args:
        boxes: list of bounding boxes, each one in the order of (ymin, xmin, ymax, xmax)

    Returns: union area

    """
    # convert to np array if the input is a list
    boxes = np.array(boxes)

    START = 1
    END = -START

    # key: y axis of the changing line
    # value list of tuple(x axis,status(beginning/ending of a meeting) )
    boundary = {}

    for box in boxes:
        y0, x0, y1, x1 = box

        if y0 not in boundary:
            boundary[y0] = []
        if y1 + 1 not in boundary:
            boundary[y1 + 1] = []

        # starting and ending of a bounding box are 'changing lines'
        # since in our case, area means number of pixels
        # and [x0,x1],[y0,y1] are inclusive,
        # so '+1' is needed for x1 and y1

        # in line y0, a meeting starts at x0 and ends at x1
        boundary[y0].append((x0, START))
        boundary[y0].append((x1 + 1, END))

        # in line y1 + 1, there will be no more meeting
        # the effect needs to be negated
        boundary[y1 + 1].append((x0, END))
        boundary[y1 + 1].append((x1 + 1, START))

    # valid length in each line is equivalent to
    # 'meeting scheduling' interview problem.
    # previous line's y value with a changing line scan
    # first value does not matter
    # as long as valid_length is set to 0 at first

    last_y = -1
    valid_length = 0
    area_sum = 0

    # one line scan over the 2d space.
    # key is x
    # value is summation of status(1/-1)
    # 1 means beginning of bounding box
    # -1 means ending of bounding box
    # changes over y

    line_scan = {}

    for current_y in sorted(boundary):
        #  valid length stay the same for [last_y, current_y]
        area_sum += (current_y - last_y) * valid_length
        last_y = current_y

        # update the status of line scan
        for pair in boundary[current_y]:
            x = pair[0]
            status = pair[1]
            line_scan[x] = line_scan.get(x, 0) + status
            if not line_scan[x]:
                del line_scan[x]

        valid_length = _get_valid_length(line_scan)

    return area_sum


def calc_cluster(bboxes, threshold=0, max_iterations=100):
    """Calculate clusters given a list of bbox coordinates. Return

    Args:
        threshold: merging overlap. Defaults to 0, meaning that if two bbox are merged if they touch
        max_iterations: this should be set to a sufficiently high number to ensure merged cluster is clean
        bboxes: list of bounding boxes, each one in the order of (ymin, xmin, ymax, xmax)
            and represents a single calcification

    Returns:
        merged_source_list: list of list of bbox idx (corresponding to the input list)
        union_area_list: list of areas of each merged cluster
        count_list: list of counts in each merged cluster
        density_list: list of number density

    """
    # TODO select better nms parameters
    bboxes = np.array(bboxes)
    merged_bbox_list, merged_source_list = non_max_suppression_fast(bboxes, threshold=threshold, max_iterations=max_iterations)
    union_area_list = []
    count_list = []
    density_list = []
    for merged_idxs in merged_source_list:
        union_area = calculate_union_area(bboxes[merged_idxs])
        union_area_list.append(union_area)
    for idx in range(len(merged_source_list)):
        count = len(merged_source_list[idx])
        density = count / union_area_list[idx]
        count_list.append(count)
        density_list.append(density)
    return merged_bbox_list, merged_source_list, union_area_list, count_list, density_list


def bbox_list_intersect(bbox_list_list, threshold=0.1):
    """

    Args:
        bbox_list_list: a list of bbox_list. Each bbox_list comes form one source, so len(bbox_list_list) is the
            number of sources

    Returns:

    """
    # print(bbox_list_list)
    flat_bbox_list = [x for xx in bbox_list_list for x in xx]
    bbox_list, source_list = non_max_suppression_fast(flat_bbox_list, threshold=threshold)
    # print(source_list)
    class_len_list = [len(bbox_list) for bbox_list in bbox_list_list]

    def map_index_to_source_class(index, class_len_list):
        """Convert an index in a concatenated list to the list index

        For example, index 5 in a concatenated list from 3 lists with length [3, 1, 4] belongs to the
        third list, thus returns 2.
        """
        cumsum_list = np.cumsum(class_len_list)
        assert index < np.max(cumsum_list) and index >= 0, 'index is {}'.format(index)
        return min(np.argwhere([index < cumsum for cumsum in np.cumsum(class_len_list)]))[0]

    source_list = [[map_index_to_source_class(x, class_len_list) for x in xx] for xx in source_list]
    # print(source_list)
    bbox_list_intersected = []
    for bbox_coord, source in zip(bbox_list, source_list):
        if len(set(source)) == len(class_len_list):
            bbox_list_intersected.append(bbox_coord)

    # print(bbox_list_intersected)
    return bbox_list_intersected


def combine_mask_images(mask_search_path1, mask_search_path2, mask_output_dir, method='intersection'):
    """Read in masks, find intersection or union of annotation and write out to file

    Notes: This function currently only supports two sources. To extend to multiple sources:
        1. Change argument to a list of search paths
        2. Call PairedDictGenerator recursively

    Args:
        mask_search_path1:
        mask_search_path2:
        mask_output_dir:
        method: 'intersection' or 'union'
    """
    assert method in ['intersection', 'union']
    join = 'inner' if method == 'intersection' else 'outer'

    generator = patch.PairedDictGenerator(mask_search_path1, mask_search_path2, output_dir=mask_output_dir,
                                          mask_suffix='.png', image_suffix='.png')
    paired_dict = generator.get_paired_image_and_mask(key_names=('mask1', 'mask2'), join=join)

    keys = sorted(paired_dict.keys())
    for key in tqdm(keys[:]):
        # read in mask arrays
        mask_array_list = []
        for dataset in paired_dict[key].keys():
            mask_path = paired_dict[key][dataset]
            if os.path.isfile((mask_path)):
                mask_array = plt.imread(mask_path, -1)
            else:
                continue
            mask_array_list.append(mask_array)

        assert mask_array_list, 'no mask found for key {}'.format(key)
        # generate union mask
        canvas = None
        for mask_array in mask_array_list:
            try:
                if method == 'intersection':
                    canvas *= mask_array.astype(np.bool)
                else:
                    canvas += mask_array.astype(np.bool)
            except:
                canvas = mask_array.astype(np.bool)

        # write generated mask to file
        mask_output_path = os.path.join(mask_output_dir, '{}_{}.png'.format(key, method))
        cv2.imwrite(mask_output_path, (canvas * 255).astype(np.uint8))


def clip_to_boundary(bbox, canvas_shape):
    """Clip bbox coordinates to canvas shape

    Args:
        bbox:
        canvas_shape:

    Returns:

    """
    ymin, xmin, ymax, xmax = bbox
    assert len(canvas_shape) == 2, 'canvas shape {} is not 2D!'.format(canvas_shape)
    height, width = canvas_shape

    # crop to boundary
    ymin = max(ymin, 0)
    xmin = max(xmin, 0)
    ymax = min(ymax, height)
    xmax = min(xmax, width)
    assert ymax - ymin > 1 and xmax - xmin > 1, 'Bbox too small, invalid crop!'
    bbox = (ymin, xmin, ymax, xmax)
    return bbox


def crop_by_bbox(image_array, bbox):
    """

    Args:
        image_array: 2d or 3d array
        bbox: in the order of (ymin, xmin, ymax, xmax)

    Returns:

    """
    canvas_shape = image_array.shape[:2]
    bbox = clip_to_boundary(bbox, canvas_shape=canvas_shape)
    ymin, xmin, ymax, xmax = bbox

    image_array = image_array[ymin:ymax, xmin:xmax, ...]
    return image_array


def poly_approx(contours, approx_tol=0.01):
    """Approximate contours with polygon

    Args:
        contours:
        approx_tol:

    Returns:

    """
    poly_contours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_tol * peri, True)
        poly_contours.append(approx)
    return poly_contours


def is_point_inside_contour(point, contour):
    """Tell if a point is inside a contour

    Args:
        point: in the order of (x, y) per cv2
        contour:

    Returns:

    """

    is_inside = cv2.pointPolygonTest(contour, point, measureDist=False) > 0
    return is_inside


def fill_holes(image, thresh=0.001):
    """Fill holes in a binary image

    Args:
        image:
        thresh:

    Returns:

    """
    from skimage.morphology import reconstruction

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image

    filled = reconstruction(seed, mask, method='erosion')
    #     filled = (filled > thresh).astype(np.uint8)
    return filled
