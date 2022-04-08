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
    average: A `str`. The class averaging mode. Valid values are `'micro'`,
      `'macro'` and `'weighted'`. Defaults to `'macro'`. See Notes for details
      on the different modes.
    class_weights: A `list` of `float` values. The weights for each class.
      Only relevant if `average` is `'weighted'`. Defaults to `None`.
    reduction: Type of reduction to apply.
    name: String name of the loss instance.

  Notes:
    * `'micro'`: Calculate the loss globally by counting the total true
      positives, true negatives, false positives and false negatives.
    * `'macro'`: Calculate the loss for each label, and return their unweighted
      mean. This does not take label imbalance into account.
    * `'weighted'`: Calculate the loss for each label, and find their average
      weighted by `class_weights`. If `class_weights` is `None`, the classes
      are weighted by support (the number of true instances for each label).
      This alters 'macro' to account for label imbalance.
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
      self.class_weights = tf.convert_to_tensor(class_weights)
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

    Returns:
      A `tf.Tensor` of shape `[batch_size]`. The averaged result.
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
        class_weights = self.class_weights
      else:
        # Weight by support, accounting for class imbalance.
        true_instances = confusion_matrix.true_positives + \
            confusion_matrix.false_negatives
        class_weights = tf.math.divide_no_nan(
            true_instances, tf.reduce_sum(true_instances, axis=-1))
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
  """Loss based on Tversky index with focusing.

  Args:
    alpha: A `float`. Weight given to false positives. Must be between 0.0 and
      1.0. Default is 0.3.
    beta: A `float`. Weight given to false negatives. Must be between 0.0 and
      1.0. Default is 0.7.
    gamma: A `float`. Focus parameter. Default is 1.33.
    epsilon: A `float`. A smoothing factor.
    average: Type of averaging to be performed on data. Acceptable values
      are `None`, `micro`, `macro` and `weighted`. Default value is None.
    reduction: Type of reduction to apply.
    name: String name of the loss instance.

  References:
    [1] Abraham, N., & Khan, N. M. (2019, April). A novel focal tversky loss
      function with improved attention u-net for lesion segmentation. In 2019
      IEEE 16th international symposium on biomedical imaging (ISBI 2019)
      (pp. 683-687). IEEE.
  """
  def __init__(self,
               alpha=0.3,
               beta=0.7,
               gamma=1.33,
               epsilon=1e-5,
               average='macro',
               class_weights=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='focal_tversky_loss'):
    super().__init__(average=average,
                     class_weights=class_weights,
                     reduction=reduction,
                     name=name)
    # Check inputs.
    if alpha < 0.0 or alpha > 1.0:
      raise ValueError("alpha value must be in range [0, 1].")
    if beta < 0.0 or beta > 1.0:
      raise ValueError("beta value must be in range [0, 1].")
    if gamma < 1.0 or gamma > 3.0:
      raise ValueError("gamma value must be in range [1, 3].")
    # Set attributes.
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
    return tf.math.pow(1.0 - index, 1.0 / self.gamma)

  def get_config(self):
    config = {
        'alpha': self.alpha,
        'beta': self.beta,
        'gamma': self.gamma,
        'epsilon': self.epsilon}
    base_config = super().get_config()
    return {**config, **base_config}
