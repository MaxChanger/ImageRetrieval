"""This file contains utility functions used for configuration"""

import json
import tensorflow as tf
import os
FLAGS = tf.app.flags.FLAGS


def maybe_load_from_json_and_overwrite(json_filepath):
  if json_filepath:
    assert os.path.isfile(json_filepath), "invalid json file!"
    with open(json_filepath) as json_data:
      d = json.load(json_data)
      print('Keys loaded from json files')
      for key, val in sorted(d.items()):
        # skip keys starting with `_` (used for commenting)
        if not key.startswith('_'):
          FLAGS.__flags[key] = val
          print('\t{:40s} : {}'.format(key, val))


