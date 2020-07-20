import os
import json
from projects.drutils import fileio
from projects.drutils.tf_utils import FlagsObjectView
from distutils.version import StrictVersion
import tensorflow as tf
import  tensorflow.contrib.image

def setup_gpu(cuda_device_id):
    # Set up GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device_id)


def set_session_keras(allow_growth=True, log_device_placement=False, fraction=0.8):
    """Set default (global) session for keras

    Args:
        allow_growth:
        log_device_placement:

    Returns:
        None
    """
    import tensorflow as tf
    import keras
    config = tf.ConfigProto()
    # Allow GPU growth
    config.gpu_options.allow_growth = allow_growth  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    # Log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    config.log_device_placement = log_device_placement
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_session_tf(allow_growth=True, log_device_placement=False, fraction=1.0):
    """Get a well-behaved session for tensorflow

    Usage:
        Replace the normal
        ```
        with tf.Session() as sess:
            # run graph
        ```
        with
        ```
        with gpu.get_session_tf() as sess:
            # run graph
        ```

    Args:
        allow_growth:
        log_device_placement:

    Returns:
        sess: a configured tf session
    """
    import tensorflow as tf
    # Allow GPU growth
    config = tf.ConfigProto()
    # Allow GPU growth
    config.gpu_options.allow_growth = allow_growth  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    # Log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    config.log_device_placement = log_device_placement
    sess = tf.Session(config=config)
    return sess


def log_flags(flags, logdir, log_name='config.json'):
    """Log tf FLAGS to json"""
    fileio.maybe_make_new_dir(logdir)
    config_log = os.path.join(logdir, log_name)
    if flags is None:
        config_dict = {}
    else:
        # for tensorflow 1.5 and above
        if StrictVersion(tf.__version__) >= StrictVersion('1.5.0'):
            flags = FlagsObjectView(flags)
        config_dict = flags.__dict__
    with open(config_log, 'w') as f:
        json.dump(config_dict, f, indent=1, sort_keys=True)


def load_graph_ckpt(filepath, gpu_memory_fraction=1.0):
    """Load model from tensorflow ckpt file

    There should be 3 files starting with `filepath`
        .meta (contains the graph)
        .index
        .data (contains the weight)
    """
    sess = get_session_tf(gpu_memory_fraction)
    saver = tf.train.import_meta_graph(filepath + '.meta')
    saver.restore(sess, filepath)
    print('Graph loaded from ckpt.')
    return sess

def load_new_graph_ckpt(filepath, gpu_memory_fraction=1.0):
    """Load model from a specific tensorflow ckpt file

    There should be 3 files starting with `filepath`
        .meta (contains the graph)
        .index
        .data (contains the weight)
    """
    g = tf.Graph()
    with g.as_default():
        sess = get_session_tf(gpu_memory_fraction)
        saver = tf.train.import_meta_graph(filepath + '.meta')
        saver.restore(sess, filepath)
    print('Graph loaded from ckpt.')
    return sess

def load_tf_checkpoint(model_path):
    """Load model from "latest" checkpoint

    Args:
        model_path: Checkpoint path

    Returns:
        Current session

    """
    import tensorflow as tf
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    meta_file = checkpoint_path + '.meta'
    g = tf.Graph()
    with g.as_default():
        sess = get_session_tf()
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint_path)
    return sess


def load_tf_frozen_graph(frozen_graph_filename):
    """Load model from frozen graph

    Args:
        frozen_graph_filename: File name of the frozen graph

    Returns:
        A TensorFlow graph containing the loaded model
    """
    import tensorflow as tf
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
        sess = get_session_tf()
    return sess
