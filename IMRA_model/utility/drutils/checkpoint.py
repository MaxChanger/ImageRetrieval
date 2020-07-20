"""This file contains utility functions used for manipulating checkpoint files"""
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os


def genenerate_pb(sess, save_dir, ckpt_name, pb_file_name, output_node_name):
    """Generate .pb model file for loading in C++.

    This function gets the network structure from sess and restores from checkpoint stored in ckpt_name.

    Args:
        sess:
        save_dir:
        ckpt_name:
        pb_file_name:
        output_node_name:

    Returns:
        None
    """
    tf.train.write_graph(sess.graph_def, save_dir, pb_file_name + '.pbtxt', as_text=True)

    g_new = tf.Graph()
    with g_new.as_default():
        input_graph = os.path.join(save_dir, pb_file_name + '.pbtxt')
        input_checkpoint = os.path.join(save_dir, ckpt_name)
        output_graph = os.path.join(save_dir, pb_file_name+'.pb')
        freeze_graph.freeze_graph(input_graph=input_graph,
                                  input_checkpoint=input_checkpoint,
                                  output_graph=output_graph,
                                  output_node_names=output_node_name,
                                  input_saver='',
                                  input_binary=False,
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  clear_devices='True',
                                  initializer_nodes='')


def rename_variables_in_checkpoint(checkpoint_dir, replace_from, replace_to, add_prefix, renamed_checkpoint_dir='', dry_run=False):
    """Load varibales from a checkpoint file, rename them and save to a new file

    Args:
      checkpoint_dir: directory containing the old checkpoint file
      dry_run: if True, perform dry run without renameing

    Returns:
        None

    Usage:
        python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir --replace_from=scope1 \
            --replace_to=scope1/model --add_prefix=abc/
            will rename the variable scope1/Variable1 to abc/scope1/model/Variable1.

    Note:
        This function only helps to restore the variable values, but it does not contain any of the graph structure.
        The graoph has to be rebult from scratch.
    """
    g = tf.Graph()
    with g.as_default():
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

        with tf.Session() as sess:
            new_name_list = []

            # latest = tf.train.latest_checkpoint(checkpoint_dir)
            # saver = tf.train.import_meta_graph(latest + '.meta', input_map=None, import_scope='')
            # saver.restore(sess, latest)
            # writer = tf.summary.FileWriter('newmodel', sess.graph)
            # saver.save(sess, os.path.join(checkpoint_dir, 'newmodel.ckpt'), global_step=9)

            # print all nodes
            # for node in tf.get_default_graph().as_graph_def().node:
            #     pass
                # print('*')
                # print(node.name)
            print('load variable from {}'.format(checkpoint_dir))
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir)[:]:
                # Load the variable

                var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

                # Set the new name
                new_name = var_name
                if None not in [replace_from, replace_to]:
                    new_name = new_name.replace(replace_from, replace_to)
                if add_prefix:
                    new_name = add_prefix + new_name

                # dump new names to a text file
                new_name_list.append(new_name)
                with open('/tmp/tmp.txt', 'w') as f_out:
                    f_out.write('\n'.join(new_name_list))

                if dry_run:
                    print('%s would be renamed to %s.' % (var_name, new_name))
                else:
                    print('Renaming %s to %s.' % (var_name, new_name))
                    # Rename the variable
                    # `load_variable` does not add tensor to `GraphKeys.GLOBAL_VARIABLES`
                    # but `Variable()` or `get_variable()` do
                    var = tf.Variable(var, name=new_name)

            if not dry_run:
                # Save the variables
                # print('***global vars: {}'.format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, None)))
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                print('Saving to {}'.format(checkpoint.model_checkpoint_path))
                # print('saver._var_list is {}'.format(saver._var_list))
                if not renamed_checkpoint_dir:
                    renamed_checkpoint_dir = os.path.join(checkpoint_dir, 'renamed')
                os.makedirs(renamed_checkpoint_dir, exist_ok=True)
                writer = tf.summary.FileWriter(renamed_checkpoint_dir, sess.graph).close()
                renamed_checkpoint_path = os.path.join(renamed_checkpoint_dir, 'renamed_checkpoint')
                # os.makedirs(renamed_checkpoint_path)
                saver.save(sess, renamed_checkpoint_path)
