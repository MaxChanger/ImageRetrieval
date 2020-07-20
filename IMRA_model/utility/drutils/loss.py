"""This file contains utility functions used for loss"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

EPSILON = 0.00001

def loss(logits, labels, type='xentropy', loss_params={}):
    """Loss function

    Args:
        logits: prediction before sigmoid
        labels: one-hot encoded prediction
        type: type of loss, can be one of the following
            'xentropy':
            'bounded_xentropy':
            'weighted_xentropy':
            'l2_distance':
            'pos_weighted_l2_distance':
            'subclass_weighted_xentropy'"
            'focal_loss':
            'dice':
            'dice_2_class':
            'single_class': single class, single label prediction (mutually exclusive)
        loss_params: dictionary with the following keys:
            'xentropy_epsilon'
            'focal_loss_gamma_p'
            'focal_loss_gamma_n'
            'focal_loss_alpha'
            'pos_weight_factor'
            'position_weight_flag'
    Returns:
        loss_: A float tensor containing the overall loss value

    Note:
        For mutually-exclusive multi-label classification, use softmax cross entropy. This is NOT currently implemented.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    """
    # use sigmoid_cross_entropy_with_logits for multi-label classification
    # get loss params
    xentropy_epsilon = loss_params.get('xentropy_epsilon', 0.01)
    focal_loss_gamma_p = loss_params.get('focal_loss_gamma_p', 2)
    focal_loss_gamma_n = loss_params.get('focal_loss_gamma_n', 2)
    focal_loss_alpha = loss_params.get('focal_loss_alpha', 0.25)
    pos_weight_factor = loss_params.get('pos_weight_factor', 1)
    labels = tf.to_float(labels, name='ToFloat')  # to accommodate `sigmoid_cross_entropy_with_logits`

    if type == 'xentropy':
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    elif type == 'bounded_xentropy':
        cross_entropy = bounded_sigmoid_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                  epsilon=xentropy_epsilon, name='xentropy')
    elif type == 'weighted_xentropy':
        num_total = tf.to_float(tf.size(labels))
        num_positives = tf.reduce_sum(labels)
        num_negatives = num_total - num_positives
        pos_weight = pos_weight_factor * num_negatives / (num_positives + EPSILON)
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,
                                                                 pos_weight=pos_weight, name='xentropy')
        cross_entropy = num_total / num_negatives * cross_entropy
    elif type == 'subclass_weighted_xentropy':
        num_total = tf.to_float(tf.shape(labels)[0])
        num_positives = tf.reduce_sum(labels, axis=0)
        num_negatives = num_total - num_positives
        pos_weight = pos_weight_factor * num_negatives / (num_positives + EPSILON)
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,
                                                                 pos_weight=pos_weight, name='xentropy')
        cross_entropy = num_total / num_negatives * cross_entropy

        print('pos weight dimension {}'.format(pos_weight.get_shape()))
        print('labels dimension {}'.format(labels.get_shape()))
        print('cross_entropy dimension {}'.format(cross_entropy.get_shape()))
    elif type == 'focal_loss':
        cross_entropy = focal_loss(prediction_tensor=logits, target_tensor=labels,
                                   alpha=focal_loss_alpha,
                                   gamma_p=focal_loss_gamma_p,
                                   gamma_n=focal_loss_gamma_n)
        cross_entropy = tf.reshape(cross_entropy, [-1])
        cross_entropy = tf.map_fn(nan_guard, cross_entropy)
    elif type == 'xentropy':
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='xentropy')
    elif type == 'l2_distance':
        # this is not by definition `cross_entropy`, but we use it anyway for loss calc
        cross_entropy = (labels - logits) ** 2

    elif type == 'pos_weighted_l2_distance':
        # NB. add weighs
        # slice logits and labels where labels is greater than -100
        position_weight_dict = {
            1: 1,
            2: [0, 0, 1, 1],
            3: [0, 0, 1, 1, 0],
            4: [0, 0, 1, 1, 1, 1, 1], # no muscle
            5: [0, 0, 0, 0, 1, 1, 0], # mlo nipple only
        }
        position_weight_list = position_weight_dict[loss_params.get('position_weight_flag', 1)]
        mask = tf.cast(tf.greater(labels, -100), tf.float32)
        cross_entropy = ((labels - logits) * position_weight_list * mask) ** 2
        # add to tensorboard
        loss_muscle = tf.reduce_mean(cross_entropy[:2])
        loss_nipple_cc = tf.reduce_mean(cross_entropy[2:4])
        loss_nipple_mlo = tf.reduce_mean(cross_entropy[4:6])
        loss_density = tf.reduce_mean(cross_entropy[6:])
        tf.summary.scalar('loss_muscle', loss_muscle)
        tf.summary.scalar('loss_nipple_cc', loss_nipple_cc)
        tf.summary.scalar('loss_nipple_mlo', loss_nipple_mlo)
        tf.summary.scalar('loss_density', loss_density)
        # tf.summary.value.add(tag='loss_muscle', simple_value=avg_valid_loss)

    elif type == 'l4_distance':
        cross_entropy = (labels - logits) ** 4

    elif type == 'l2_4_distance':
        def l2_4_loss_fn(d):
            # This maps scalar tensor (rank-0 tensor) to scalar
            return tf.case(
                pred_fn_pairs=[
                    (tf.abs(d) <= 1, lambda: 2 * (d ** 2))],
                default=lambda: d ** 4 + 1, exclusive=True)
        difference = labels - logits
        orig_shape = tf.shape(difference) # NB. a.get_shape() is equivalent to a.shape and returns static shape
        difference = tf.reshape(difference, [-1])
        cross_entropy = tf.map_fn(l2_4_loss_fn, difference)
        cross_entropy = tf.reshape(cross_entropy, orig_shape)
    elif type == 'flat_bottom_l3_distance':
        def l3_loss_fn(d):
            # This maps scalar tensor (rank-0 tensor) to scalar
            return tf.case(
                pred_fn_pairs=[
                    (d > 1, lambda: (d - 1) ** 3),
                    (d < -1, lambda: -(d + 1) ** 3)],
                default=lambda: tf.constant(0, dtype=tf.float32), exclusive=True)
        difference = labels - logits
        orig_shape = tf.shape(difference) # NB. a.get_shape() is equivalent to a.shape and returns static shape
        difference = tf.reshape(difference, [-1])
        cross_entropy = tf.map_fn(l3_loss_fn, difference)
        cross_entropy = tf.reshape(cross_entropy, orig_shape)
    elif type == 'l4_log_loss':
        def l4_log_loss_fn(d):
            # This maps scalar tensor (rank-0 tensor) to scalar
            return tf.case(
                pred_fn_pairs=[
                    (tf.abs(d) <= 1, lambda: (d ** 4)/12)],
                default=lambda: -tf.log((4-tf.abs(d))/4) + 1/12 - tf.log(4/3), exclusive=True)
        difference = labels - logits
        orig_shape = tf.shape(difference) # NB. a.get_shape() is equivalent to a.shape and returns static shape
        difference = tf.reshape(difference, [-1])
        cross_entropy = tf.map_fn(l4_log_loss_fn, difference)
        cross_entropy = tf.reshape(cross_entropy, orig_shape)
    elif type == 'dice':
        predictions = tf.nn.sigmoid(logits)
        dice_numerator_1 = tf.reduce_sum(tf.multiply(labels, predictions), axis=0) + EPSILON
        dice_denominator_1 = tf.reduce_sum(predictions, axis=0) + tf.reduce_sum(labels, axis=0) + EPSILON
        dice_numerator_2 = tf.reduce_sum(tf.multiply((1 - labels), (1 - predictions)), axis=0) + EPSILON
        dice_denominator_2 = tf.reduce_sum((1 - predictions), axis=0) + tf.reduce_sum((1 - labels),
                                                                                      axis=0) + EPSILON
        cross_entropy = 1 - (tf.div(dice_numerator_1, dice_denominator_1)
                             - pos_weight_factor * tf.div(dice_numerator_2, dice_denominator_2))
    elif type == 'dice_2_class':
        predictions = tf.nn.softmax(logits)
        dice_numerator_1 = tf.reduce_sum(tf.multiply(labels, tf.slice(predictions, [0,0], [-1, 1])), axis=0) + EPSILON
        dice_denominator_1 = (tf.reduce_sum(tf.slice(predictions, [0,0], [-1, 1]), axis=0)
                              + tf.reduce_sum(labels, axis=0) + EPSILON)
        dice_numerator_2 = tf.reduce_sum(tf.multiply((1 - labels), tf.slice(predictions, [0,1], [-1, 1])), axis=0) + EPSILON
        dice_denominator_2 = (tf.reduce_sum(tf.slice(predictions, [0,1], [-1, 1]), axis=0)
                              + tf.reduce_sum((1 - labels), axis=0) + EPSILON)
        cross_entropy = 1 - (tf.div(dice_numerator_1, dice_denominator_1)
                             - pos_weight_factor * tf.div(dice_numerator_2, dice_denominator_2))
    else:
        raise ValueError('Unkown loss funciton type {}'.format(type))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    # tf.summary.scalar('loss', loss_) # add to summary when loss() is called
    return loss_


def focal_loss(prediction_tensor, target_tensor, alpha=0.5, gamma_p=2.0, gamma_n=2.0):
    """Compute focal loss for predictions

    Multi-labels Focal loss formula:
        FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
             ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Ref: https://arxiv.org/pdf/1708.02002.pdf

    Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        alpha: A scalar tensor for focal loss alpha hyper-parameter
        gamma: A scalar tensor for focal loss gamma hyper-parameter

    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    # when target_tensor == 1, pos_p_sub = 1 - sigmoid_p
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    # when target_tensor == 0, neg_p_sub = sigmoid_p
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma_p) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma_n) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent


def bounded_sigmoid_cross_entropy_with_logits(
        labels=None,
        logits=None,
        epsilon=0.0,
        name=None):
    """Computes sigmoid cross entropy given `logits` with a `bound` (b) away from 0 and 1 when computing log(x).

    Modified based on tf.nn.sigmoid_cross_entropy_with_logits()
        For brevity, let `x = logits`, `z = labels`.  The logistic loss is
              z * -log(b + sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x) + b)
            = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
            = (1 - z) * x + log(1 + exp(-x))
            = x - x * z + log(1 + exp(-x))
        For x < 0, to avoid overflow in exp(-x), we reformulate the above
              x - x * z + log(1 + exp(-x))
            = log(exp(x)) - x * z + log(1 + exp(-x))
            = - x * z + log(1 + exp(x))
        Hence, to ensure stability and avoid overflow, the implementation uses this
        equivalent formulation
            max(x, 0) - x * z + log(1 + exp(-abs(x)))
        `logits` and `labels` must have the same type and shape.

    Args:
      labels: A `Tensor` of the same type and shape as `logits`.
      logits: A `Tensor` of type `float32` or `float64`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
      logistic losses.

    Raises:
      ValueError: If `logits` and `labels` do not have the same shape.
    """

    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        labels = ops.convert_to_tensor(labels, name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

        # The logistic loss formula from above is
        #   x - x * z + log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   -x * z + log(1 + exp(x))
        # Note that these two expressions can be combined into the following:
        #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # To allow computing gradients at zero, we define custom versions of max and
        # abs functions.
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        # relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)
        out_negative = math_ops.add(
            math_ops.log1p(math_ops.exp(neg_abs_logits)),
            math_ops.add(
                -labels * math_ops.log(math_ops.add(epsilon, math_ops.exp(neg_abs_logits))),
                -(1 - labels) * math_ops.log1p(epsilon * math_ops.exp(neg_abs_logits))
            )
        )
        out_positive = math_ops.add(
            math_ops.log1p(math_ops.exp(neg_abs_logits)),
            math_ops.add(
                -labels * math_ops.log1p(epsilon * math_ops.exp(neg_abs_logits)),
                -(1 - labels) * math_ops.log(math_ops.add(epsilon, math_ops.exp(neg_abs_logits)))
            )
        )
        return array_ops.where(cond, out_positive, out_negative)


def nan_guard(x):
    """Replace NaN with 0

    Args:
        x: a Tensor
    Returns:
        A Tensor, replaced with 0 if NaN
    """
    return tf.cond(tf.is_nan(x),
                   lambda: tf.zeros_like(x),
                   lambda: x)


def none_guard(x):
    """Replace None with a constant Tensor 0

    Args:
        x: a Tensor
    Returns:
        A Tensor, replaced with 0 if NaN
    """
    return tf.cond(lambda: (x is None),
                   lambda: tf.constant(0),
                   lambda: x)
