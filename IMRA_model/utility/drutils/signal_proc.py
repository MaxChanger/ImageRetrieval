"""Functions for 1d signal processing"""

import numpy as np
import scipy

def smooth_mean(y, window=11, mode='valid'):
    box = np.ones(window) / window
    if mode == 'valid':
        y = np.pad(y, pad_width=(window)//2, mode='reflect')
    y_smooth = np.convolve(y, box, mode=mode)
    return y_smooth


def smooth_med(y, window=11):
    y_smooth = scipy.signal.medfilt(y, kernel_size=window)
    return y_smooth


def my_smooth(y, window=11):
    y_smooth = smooth_mean(smooth_med(y, window=window))
    return y_smooth