"""
This file contains utility functions used to convert between data format
"""

import re
import numpy as np

def parse_coord_string(bbox_coord):
    """Get coord numbers from bbox_coord

    bbox_coord is in the format of `(x,w,w,h)`

    Args:
        bbox_coord:

    Returns:
        coord: a numpy array of coord
    """
    fields = bbox_coord.replace(')', '').replace('(', '').split(',')
    coord = np.array([int(field) for field in fields])
    return coord