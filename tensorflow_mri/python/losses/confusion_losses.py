# Copyright 2022 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Confusion losses.

This module contains loss functions derived from the confusion matrix for
classification and segmentation problems.
"""

import abc
import dataclasses

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


_CONFUSION_LOSS_INTRO_DOCSTRING = """
  Inputs `y_true` and `y_pred` are expected to have shape `[..., num_classes]`,
  with channel `i` containing labels/predictions for class `i`. `y_true[..., i]`
  is 1 if the element represented by `y_true[...]` is a member of class `i` and
  0 otherwise. `y_pred[..., i]` is the predicted probability, in the range
  `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
  class `i`.

  This class further assumes that inputs `y_true` and `y_pred` have shape
  `[batch_size, ..., num_classes]`. The loss is computed for each batch element
  `y_true[i, ...]` and `y_pred[i, ...]`, and then reduced over this dimension
  as specified by argument `reduction`.

  This loss works for binary, multiclass and multilabel classification and/or
  segmentation. In multiclass/multilabel problems, the different classes are
  combined according to the `average` and `class_weights` arguments. Argument
  `average` can take one of the following values:

  * `'micro'`: Calculate the loss globally by counting the total number of
    true positives, true negatives, false positives and false negatives.
  * `'macro'`: Calculate the loss for each label, and return their unweighted
    mean. This does not take label imbalance into account.
  * `'weighted'`: Calculate the loss for each label, and find their average
    weighted by `class_weights`. If `class_weights` is `None`, the classes
    are weighted by support (the number of true instances for each label).
    This alters 'macro' to account for label imbalance.

"""


_CONFUSION_LOSS_ARGS_DOCSTRING = """
    average: A `str`. The class averaging mode. Valid values are `'micro'`,
      `'macro'` and `'weighted'`. Defaults to `'macro'`. See above for details
      on the different modes.
    class_weights: A `list` of `float` values. The weights for each class.
      Must have length equal to the number of classes. This parameter is only
      relevant if `average` is `'weighted'`. Defaults is `None`.
    reduction: A value in `tf.keras.losses.Reduction`_.
      The type of loss reduction.
    name: A `str`. The name of the loss instance.

  .. _tf.keras.losses.Reduction: https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction

"""


@dataclasses.dataclass
class ConfusionMatrix():
  true_positives: tf.Tensor
  true_negatives: tf.Tensor
  false_positives: tf.Tensor
  false_negatives: tf.Tensor


@api_util.export("losses.ConfusionLoss")
class ConfusionLoss(tf.keras.losses.Loss):
  """Abstract base class for losses derived from the confusion matrix.

  A confusion matrix is a table that reports the number of true positives,
  false positives, true negatives and false negatives.

  This provides a base class for losses that are calculated based on the
  values of the confusion matrix.

  This class's `call` method computes the confusion matrix and then calls
  method `result`. Subclasses are expected to implement this method to
  compute the loss value based on the confusion matrix. Then, the average
  is computed according to the configuration.

  This class exposes the attributes `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` for use by subclasses. Each of these
  is a list containing one value for each class.

  This loss may be used for binary, multi-class and multi-label classification.

  Args:
  """
  def __init__(self,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='confusion_loss'):
    super().__init__(reduction=reduction, name=name)
    self.average = check_util.validate_enum(
        average, {'micro', 'macro', 'weighted'}, 'average')
    if class_weights is not None:
      self.class_weights = class_weights
    else:
      self.class_weights = None

  def call(self, y_true, y_pred): # pylint: disable=missing-function-docstring
    y_true = tf.convert_to_tensor(y_true, name='y_true')
    y_pred = tf.convert_to_tensor(y_pred, name='y_pred')
    y_true = tf.cast(y_true, y_pred.dtype)
    # Reduce over all axes except batch and channels.
    axis = tf.range(1, tf.rank(y_pred) - 1)
    # Compute confusion matrix. Note that these are relaxations where `y_pred`
    # has support in [0, 1], as opposed to {0, 1}. Each component of the
    # confusion matrix has shape `[batch_size, num_classes]`.
    confusion_matrix = ConfusionMatrix(
        true_positives=tf.reduce_sum(y_pred * y_true, axis=axis),
        true_negatives=tf.reduce_sum((1 - y_pred) * (1 - y_true), axis=axis),
        false_positives=tf.reduce_sum(y_pred * (1 - y_true), axis=axis),
        false_negatives=tf.reduce_sum((1 - y_pred) * y_true, axis=axis)
    )
    # If averaging is micro, we need to do it before computing the loss.
    if self.average == 'micro':
      confusion_matrix = ConfusionMatrix(
          true_positives=tf.reduce_sum(
              confusion_matrix.true_positives, -1, keepdims=True),
          true_negatives=tf.reduce_sum(
              confusion_matrix.true_negatives, -1, keepdims=True),
          false_positives=tf.reduce_sum(
              confusion_matrix.false_positives, -1, keepdims=True),
          false_negatives=tf.reduce_sum(
              confusion_matrix.false_negatives, -1, keepdims=True)
      )
    # Compute the loss.
    loss = self._call(confusion_matrix)
    # Compute result and average.
    loss = self._average(loss, confusion_matrix)
    return loss

  @abc.abstractmethod
  def _call(self, confusion_matrix):
    """Compute the loss value from the confusion matrix.

    This method must be implemented by subclasses.

    Args:
      confusion_matrix: A `ConfusionMatrix` instance. The confusion matrix
        is a `dataclass` with fields `true_positives`, `true_negatives`,
        `false_positives` and `false_negatives`. Each of these is a `tf.Tensor`
        of shape `[batch_size, num_classes]`.

    Returns:
      A `tf.Tensor` of shape `[batch_size, num_classes]` containing the loss
      value for class and batch element.
    """
    raise NotImplementedError("_call must be implemented in subclasses.")

  def _average(self, class_values, confusion_matrix):
    """Average the class values according to the specified configuration.

    Args:
      class_values: A `tf.Tensor` of shape `[batch_size, num_classes]`, or
        `[batch_size, 1]` if `average` is `'micro'`.
      confusion_matrix: A `ConfusionMatrix` instance. The confusion matrix
        is a `dataclass` with fields `true_positives`, `true_negatives`,
        `false_positives` and `false_negatives`. Each of these is a `tf.Tensor`
        of shape `[batch_size, num_classes]`.

    Returns:
      A `tf.Tensor` of shape `[batch_size]`. The averaged result.

    Raises:
      ValueError: If `self.average` has an invalid value.
    """
    if self.average == 'micro':
      # In this case the confusion matrix has been averaged over all classes,
      # so nothing to do here.
      class_average = class_values[..., 0]

    elif self.average == 'macro':
      class_average = tf.math.reduce_mean(class_values, axis=-1)

    elif self.average == 'weighted':
      if self.class_weights is not None:
        # Use user-specified class weights.
        class_weights = tf.convert_to_tensor(self.class_weights)
      else:
        # Weight by support, accounting for class imbalance.
        true_instances = confusion_matrix.true_positives + \
            confusion_matrix.false_negatives
        class_weights = tf.math.divide_no_nan(
            true_instances,
            tf.reduce_sum(true_instances, axis=-1, keepdims=True))
      # Weighted average.
      class_average = tf.math.reduce_sum(
          class_values * class_weights, axis=-1)

    else:
      raise ValueError(f"Unknown average mode: {self.average}")

    return class_average

  def get_config(self):
    config = {
        'average': self.average,
        'class_weights': self.class_weights}
    base_config = super().get_config()
    return {**config, **base_config}


@api_util.export("losses.FocalTverskyLoss")
@tf.keras.utils.register_keras_serializable(package='MRI')
class FocalTverskyLoss(ConfusionLoss):
  r"""Focal Tversky loss function.

  The focal Tversky loss is computed as:

  .. math::
    L = \left ( 1 - \frac{\mathrm{TP} + \epsilon}{\mathrm{TP} + \alpha \mathrm{FP} + \beta \mathrm{FN} + \epsilon} \right ) ^ \gamma

  This loss allows control over the relative importance of false positives and
  false negatives through the `alpha` and `beta` parameters, which may be useful
  in imbalanced classes. Additionally, the `gamma` exponent can be used to
  shift the focus towards difficult examples.

  Args:
    alpha: A `float`. Weight given to false positives. Defaults to 0.3.
    beta: A `float`. Weight given to false negatives. Defaults to 0.7.
    gamma: A `float`. The focus parameter. A lower value increases the
      importance given to difficult examples. Default is 0.75.
    epsilon: A `float`. A smoothing factor. Defaults to 1e-5.

  Notes:
    [1] and [2] use inverted notations for the :math:`\alpha` and :math:`\beta`
    parameters. Here we use the notation of [1]. Also note that [2] refers to
    :math:`\gamma` as :math:`\frac{1}{\gamma}`.

  References:
    [1] Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017, September).
      Tversky loss function for image segmentation using 3D fully convolutional
      deep networks. In International workshop on machine learning in medical
      imaging (pp. 379-387). Springer, Cham.
    [2] Abraham, N., & Khan, N. M. (2019, April). A novel focal tversky loss
      function with improved attention u-net for lesion segmentation. In 2019
      IEEE 16th international symposium on biomedical imaging (ISBI 2019)
      (pp. 683-687). IEEE.
  """  # pylint: disable=line-too-long
  def __init__(self,
               alpha=0.3,
               beta=0.7,
               gamma=0.75,
               epsilon=1e-5,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='focal_tversky_loss'):
    super().__init__(average=average,
                     class_weights=class_weights,
                     reduction=reduction,
                     name=name)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.epsilon = epsilon

  def _call(self, confusion_matrix):
    numerator = confusion_matrix.true_positives
    denominator = confusion_matrix.true_positives + \
        self.alpha * confusion_matrix.false_positives + \
        self.beta * confusion_matrix.false_negatives
    index = (numerator + self.epsilon) / (denominator + self.epsilon)
    return tf.math.pow(1.0 - index, self.gamma)

  def get_config(self):
    config = {
        'alpha': self.alpha,
        'beta': self.beta,
        'gamma': self.gamma,
        'epsilon': self.epsilon}
    base_config = super().get_config()
    return {**config, **base_config}


@api_util.export("losses.TverskyLoss")
@tf.keras.utils.register_keras_serializable(package='MRI')
class TverskyLoss(FocalTverskyLoss):
  r"""Tversky loss function.

  The Tversky loss is computed as:

  .. math::
    L = \left ( 1 - \frac{\mathrm{TP} + \epsilon}{\mathrm{TP} + \alpha \mathrm{FP} + \beta \mathrm{FN} + \epsilon} \right )

  Args:
    alpha: A `float`. Weight given to false positives. Defaults to 0.3.
    beta: A `float`. Weight given to false negatives. Defaults to 0.7.
    epsilon: A `float`. A smoothing factor. Defaults to 1e-5.
  """  # pylint: disable=line-too-long
  def __init__(self,
               alpha=0.3,
               beta=0.7,
               epsilon=1e-5,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='tversky_loss'):
    super().__init__(alpha=alpha,
                     beta=beta,
                     gamma=1.0,
                     epsilon=epsilon,
                     average=average,
                     class_weights=class_weights,
                     reduction=reduction,
                     name=name)

  def get_config(self):
    base_config = super().get_config()
    base_config.pop('gamma')
    return base_config


@api_util.export("losses.F1Loss", "losses.DiceLoss")
@tf.keras.utils.register_keras_serializable(package='MRI')
class F1Loss(TverskyLoss):
  r"""F1 loss function (aka Dice loss).

  The F1 loss is computed as:

  .. math::
    L = \left ( 1 - \frac{\mathrm{TP} + \epsilon}{\mathrm{TP} + \frac{1}{2} \mathrm{FP} + \frac{1}{2} \mathrm{FN} + \epsilon} \right )

  Args:
    epsilon: A `float`. A smoothing factor. Defaults to 1e-5.
  """  # pylint: disable=line-too-long
  def __init__(self,
               epsilon=1e-5,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='f1_loss'):
    super().__init__(alpha=0.5,
                     beta=0.5,
                     epsilon=epsilon,
                     average=average,
                     class_weights=class_weights,
                     reduction=reduction,
                     name=name)

  def get_config(self):
    base_config = super().get_config()
    base_config.pop('alpha')
    base_config.pop('beta')
    return base_config


@api_util.export("losses.IoULoss", "losses.JaccardLoss")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IoULoss(TverskyLoss):
  r"""IoU loss function (aka Jaccard loss).

  The IoU loss is computed as:

  .. math::
    L = \left ( 1 - \frac{\mathrm{TP} + \epsilon}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN} + \epsilon} \right )

  Args:
    epsilon: A `float`. A smoothing factor. Defaults to 1e-5.
  """  # pylint: disable=line-too-long
  def __init__(self,
               epsilon=1e-5,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='iou_loss'):
    super().__init__(alpha=1.0,
                     beta=1.0,
                     epsilon=epsilon,
                     average=average,
                     class_weights=class_weights,
                     reduction=reduction,
                     name=name)

  def get_config(self):
    base_config = super().get_config()
    base_config.pop('alpha')
    base_config.pop('beta')
    return base_config


def _update_docstring(docstring):
  """Updates the docstring of a loss function.

  Args:
    docstring: A `str`. The docstring of the loss function.

  Returns:
    A `str`. The updated docstring.
  """
  doclines = docstring.splitlines()
  args_index = doclines.index("  Args:")
  args_end_index = args_index
  while doclines[args_end_index].strip() != "":
    args_end_index += 1
  doclines[args_end_index:args_end_index] = \
      _CONFUSION_LOSS_ARGS_DOCSTRING.splitlines()
  doclines[args_index-1:args_index-1] = \
      _CONFUSION_LOSS_INTRO_DOCSTRING.splitlines()
  return '\n'.join(doclines)


ConfusionLoss.__doc__ = _update_docstring(ConfusionLoss.__doc__)
FocalTverskyLoss.__doc__ = _update_docstring(FocalTverskyLoss.__doc__)
TverskyLoss.__doc__ = _update_docstring(TverskyLoss.__doc__)
F1Loss.__doc__ = _update_docstring(F1Loss.__doc__)
IoULoss.__doc__ = _update_docstring(IoULoss.__doc__)
