"""Utility functions for tensorflow"""

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class FlagsObjectView(object):
    """For tensorflow 1.5 and above"""
    def __init__(self, FLAGS):
        self.__dict__ = FLAGS.flag_values_dict()
