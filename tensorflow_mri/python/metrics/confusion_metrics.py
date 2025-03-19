# Copyright 2021 University College London. All Rights Reserved.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Confusion metrics.

This module contains metrics derived from the confusion matrix for
classification and segmentation problems.
"""
# Some of the code in this file is adapted from
# tensorflow_addons/metrics/f_scores.py, to support a much wider range of uses
# and metrics.

import abc

import tensorflow as tf

from tensorflow_mri.python.util import api_util


_CONFUSION_METRIC_INTRO_DOCTRING = """
  Inputs `y_true` and `y_pred` are expected to have shape `[..., num_classes]`,
  with channel `i` containing labels/predictions for class `i`. `y_true[..., i]`
  is 1 if the element represented by `y_true[...]` is a member of class `i` and
  0 otherwise. `y_pred[..., i]` is the predicted probability, in the range
  `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
  class `i`.

  This metric works for binary, multiclass and multilabel classification. In
  multiclass/multilabel problems, this metric can be used to measure performance
  globally or for a specific class.

  With the default configuration, this metric will:

  * If `num_classes == 1`, assume a binary classification problem with a
    threshold of 0.5 and return the confusion metric.
  * If `num_classes >= 2`, assume a multiclass classification problem where
    the class with the highest probability is selected as the prediction,
    compute the confusion metric for each class and return the unweighted mean.

  See the Parameters and Notes for other configurations.

"""

_CONFUSION_METRIC_NOTES_DOCTRING = """
  Notes:
    This metric works for binary, multiclass and multilabel classification.

    * For **binary** tasks, set `num_classes` to 1, and optionally, `threshold`
      to the desired value (default is 0.5 if unspecified). The value of
      `average` is irrelevant.
    * For **multiclass** tasks, set `num_classes` to the number of possible
      labels and set `average` to the desired mode. `threshold` should be left
      as `None`.
    * For **multilabel** tasks, set `num_classes` to the number of possible
      labels, set `threshold` to the desired value in the range `(0.0, 1.0)` (or
      provide a list of length `num_classes` to specify a different threshold
      value for each class), and set `average` to the desired mode.

    In multiclass/multilabel problems, this metric can be used to measure
    performance globally or for a specific class. For a specific class, set
    `class_id` to the desired value. For a global measure, set `class_id` to
    `None` and `average` to the desired averaging method. `average` can take
    the following values:

    * `None`: Scores for each class are returned.
    * `'micro'`: Calculate metrics globally by counting the total true
      positives, true negatives, false positives and false negatives.
    * `'macro'`: Calculate metrics for each label, and return their unweighted
      mean. This does not take label imbalance into account.
    * `'weighted'`: Calculate metrics for each label, and find their average
      weighted by support (the number of true instances for each label). This
      alters 'macro' to account for label imbalance.

"""

_CONFUSION_METRIC_ARGS_DOCSTRING = """
    num_classes: Number of unique classes in the dataset. If this value is not
      specified, it will be inferred during the first call to `update_state`
      as `y_pred.shape[-1]`.
    class_id: Integer class ID for which metrics should be reported. This must
      be in the half-open interval [0, num_classes). If `None`, a global average
      metric is returned as defined by `average`. Defaults to `None`.
    average: Type of averaging to be performed on data. Valid values are `None`,
      `'micro'`, `'macro'` and `'weighted'`. Defaults to `'macro'`. See Notes
      for details on the different modes. This parameter is ignored if
      `class_id` is not `None`.
    threshold: Elements of `y_pred` above threshold are considered to be 1, and
      the rest 0. A list of length `num_classes` may be provided to specify a
      threshold for each class. If threshold is `None`, the argmax is converted
      to 1, and the rest 0. Defaults to `None` if `num_classes >= 2` (multiclass
      classification) and 0.5 if `num_classes == 1` (binary classification).
      This parameter is required for multilabel classification.
"""

@api_util.export("metrics.ConfusionMetric")
class ConfusionMetric(tf.keras.metrics.Metric): # pylint: disable=abstract-method
  """Abstract base class for metrics derived from the confusion matrix.

  This class maintains a confusion matrix in its state and updates it with every
  call to `update_state`. Subclasses must implement the `_result` method to
  compute the desired metric. `_result` is called during `result`.

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='confusion_matrix',
               dtype=None):
    super().__init__(name=name, dtype=dtype)
    # Check inputs.
    if class_id is not None:
      if not isinstance(class_id, int):
        raise TypeError(
          "Argument `class_id` must be a Python int.")
    if average not in (None, 'micro', 'macro', 'weighted'):
      raise ValueError((
        "Unknown average mode: {}. Valid values are: "
        "[None, 'micro', 'macro', 'weighted']").format(average))
    if threshold is not None:
      if isinstance(threshold, float):
        threshold = [threshold]
      for t in threshold:
        if t >= 1.0 or t <= 0.0:
          raise ValueError(
            "Argument `threshold` must be in the open interval (0, 1).")

    # Set attributes.
    self.num_classes = num_classes
    self.class_id = class_id
    self.average = average
    self.threshold = threshold

    # If user informed us about number of classes, build layer.
    self._built = False
    if self.num_classes:
      self._build()

  def _build(self, input_shape=None):
    """Initialize weights.

    Args:
      input_shape: Input shape. Used only if `num_classes` has not been
        set. The static channel dimension must be defined.

    Raises:
      ValueError: If `class_id` is outside range [0, num_classes).
    """
    # Get number of classes if not already set.
    self.num_classes = self.num_classes or input_shape[-1]

    # If number of classes is 1 and threshold was not provided, set to 0.5.
    if self.num_classes == 1:
      self.threshold = self.threshold or 0.5

    # Check `class_id` argument now that we know the number of classes.
    if self.class_id is not None:
      if self.class_id < 0 or self.class_id >= self.num_classes:
        raise ValueError(
          f"Argument `class_id` must be in the range "
          f"[0, num_classes), but got: {self.class_id}")

    # If we are returning the metric for a single class or the 'micro' average,
    # we only need to store one confusion matrix. Otherwise, we need to store
    # one confusion matrix per class.
    if self.class_id is not None or self.average == 'micro':
      self.init_shape = []
    else:
      self.init_shape = [self.num_classes]

    # Initialize the confusion matrix entries and the number of true instances.
    def _zero_wt_init(name):
      return self.add_weight(name,
                             shape=self.init_shape,
                             initializer='zeros',
                             dtype=self.dtype)

    self.true_positives = _zero_wt_init('true_positives')
    self.true_negatives = _zero_wt_init('true_negatives')
    self.false_positives = _zero_wt_init('false_positives')
    self.false_negatives = _zero_wt_init('false_negatives')
    self.true_instances = _zero_wt_init('true_instances')
    self._built = True

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update confusion matrix entries.

    Args:
      y_true: The ground truth labels. Must have shape `[..., num_classes]`,
        where `y_true[..., i]` is 1 if the element represented by `y_true[...]`
        is a member of class `i` and 0 otherwise.
      y_pred: The predictions. Must have shape `[..., num_classes]`, where
        `y_pred[..., i]` is the predicted probability, in the range
        `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member
        of class `i`.
      sample_weight: The predictions are weighted by `sample_weight`. If
        `sample_weight` is `None`, weights default to 1. Use a `sample_weight`
        of 0 to mask values.
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Build layer if not built yet.
    if not self._built:
      self._build(tf.TensorShape(y_pred.shape))

    # Convert input probabilities to 0s and 1s.
    if self.threshold is None:
      # Threshold is `None`, so multiclass problem. Compute argmax and convert
      # to one-hot representation.
      y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1),
                          self.num_classes, on_value=True, off_value=False)
    else:
      # Binary or multilabel classification.
      y_pred = y_pred > self.threshold

    # Cast to metric type.
    y_true = tf.cast(y_true, self.dtype)
    y_pred = tf.cast(y_pred, self.dtype)

    # Update values.
    def _weighted_sum(val, sample_weight):
      if sample_weight is not None:
        val = tf.math.multiply(val, sample_weight)
      axis = None if self.average == 'micro' else tf.range(tf.rank(val)-1)
      if self.class_id is not None:
        val = val[..., self.class_id]
      return tf.reduce_sum(val, axis=axis)

    self.true_positives.assign_add(
        _weighted_sum(y_pred * y_true, sample_weight))
    self.true_negatives.assign_add(
        _weighted_sum((1 - y_pred) * (1 - y_true), sample_weight))
    self.false_positives.assign_add(
        _weighted_sum(y_pred * (1 - y_true), sample_weight))
    self.false_negatives.assign_add(
        _weighted_sum((1 - y_pred) * y_true, sample_weight))
    self.true_instances.assign_add(
        _weighted_sum(y_true, sample_weight))

  def result(self): # pylint: disable=missing-function-docstring
    # Compute metric. This must be implemented by subclasses.
    value = self._result()
    # Average values.
    if self.average == 'weighted':
      weights = tf.math.divide_no_nan(
          self.true_instances, tf.reduce_sum(self.true_instances))
      value = tf.math.reduce_sum(value * weights)
    elif self.average in ('micro', 'macro'):
      value = tf.math.reduce_mean(value)
    return value

  @abc.abstractmethod
  def _result(self):
    """Compute the desired metric from the state variables.

    This method must be implemented by subclasses.
    """
    raise NotImplementedError("Must be implemented in subclasses.")

  def get_config(self):
    config = {
      'num_classes': self.num_classes,
      'class_id': self.class_id,
      'average': self.average,
      'threshold': self.threshold}
    base_config = super().get_config()
    return {**base_config, **config}

  def reset_state(self):
    reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
    self.true_positives.assign(reset_value)
    self.true_negatives.assign(reset_value)
    self.false_positives.assign(reset_value)
    self.false_negatives.assign(reset_value)
    self.true_instances.assign(reset_value)


@api_util.export("metrics.Accuracy")
@tf.keras.utils.register_keras_serializable(package="MRI")
class Accuracy(ConfusionMetric):
  r"""Computes accuracy.

  Estimates how often predictions match labels.

  .. math::
    \textrm{accuracy} = \frac{\textrm{TP} + \textrm{TN}}{\textrm{TP} + \textrm{TN} + \textrm{FP} + \textrm{FN}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """ # pylint: disable=line-too-long
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='acc',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

  def _result(self):
    """Computes accuracy from confusion matrix state variables."""
    return tf.math.divide_no_nan(
        self.true_positives + self.true_negatives,
        self.true_positives + self.true_negatives + \
        self.false_positives + self.false_negatives)


@api_util.export("metrics.TruePositiveRate", "metrics.Recall",
                 "metrics.Sensitivity")
@tf.keras.utils.register_keras_serializable(package="MRI")
class TruePositiveRate(ConfusionMetric):
  r"""Computes the true positive rate (TPR).

  The true positive rate (TPR), also called sensitivity or recall, is the
  proportion of correctly predicted positives among all positive instances.

  .. math::
    \textrm{TPR} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='tpr',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

  def _result(self):
    """Computes the TPR from confusion matrix state variables."""
    return tf.math.divide_no_nan(
      self.true_positives,
      self.true_positives + self.false_negatives)


@api_util.export("metrics.TrueNegativeRate", "metrics.Specificity",
                 "metrics.Selectivity")
@tf.keras.utils.register_keras_serializable(package="MRI")
class TrueNegativeRate(ConfusionMetric):
  r"""Computes the true negative rate (TNR).

  The true negative rate (TNR), also called specificity or selectivity, is the
  proportion of correctly predicted negatives among all negative instances.

  .. math::
    \textrm{TNR} = \frac{\textrm{TN}}{\textrm{TN} + \textrm{FP}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='tnr',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

  def _result(self):
    """Computes the TNR from confusion matrix state variables."""
    return tf.math.divide_no_nan(
      self.true_negatives,
      self.true_negatives + self.false_positives)


@api_util.export("metrics.PositivePredictiveValue", "metrics.Precision")
@tf.keras.utils.register_keras_serializable(package="MRI")
class PositivePredictiveValue(ConfusionMetric):
  r"""Computes the positive predictive value (PPV).

  The positive predictive value (PPV), also called precision, is the proportion
  of correctly predicted positives among all positive calls.

  .. math::
    \textrm{PPV} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='ppv',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

  def _result(self):
    """Computes the PPV from confusion matrix state variables."""
    return tf.math.divide_no_nan(
      self.true_positives,
      self.true_positives + self.false_positives)


@api_util.export("metrics.NegativePredictiveValue")
@tf.keras.utils.register_keras_serializable(package="MRI")
class NegativePredictiveValue(ConfusionMetric):
  r"""Computes the negative predictive value (NPV).

  The negative predictive value (NPV) is the proportion of correctly predicted
  negatives among all negative calls.

  .. math::
    \textrm{NPV} = \frac{\textrm{TN}}{\textrm{TN} + \textrm{FN}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='npv',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

  def _result(self):
    """Computes NPV from confusion matrix state variables."""
    return tf.math.divide_no_nan(
      self.true_negatives,
      self.true_negatives + self.false_negatives)


@api_util.export("metrics.TverskyIndex")
@tf.keras.utils.register_keras_serializable(package="MRI")
class TverskyIndex(ConfusionMetric):
  r"""Computes Tversky index.

  The Tversky index is an asymmetric similarity measure [1]_. It is a
  generalization of the F-beta family of scores and the IoU.

  .. math::
    \textrm{TI} = \frac{\textrm{TP}}{\textrm{TP} + \alpha * \textrm{FP} + \beta * \textrm{FN}}

  Args:
    alpha: A `float`. The weight given to false positives. Defaults to 0.5.
    beta: A `float`. The weight given to false negatives. Defaults to 0.5.
    name: String name of the metric instance.
    dtype: Data type of the metric result.

  References:
    .. [1] Tversky, A. (1977). Features of similarity. Psychological review,
      84(4), 327.
  """ # pylint: disable=line-too-long
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               alpha=0.5,
               beta=0.5,
               name='tversky_index',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     name=name,
                     dtype=dtype)

    if alpha < 0.0 or alpha > 1.0:
      raise ValueError("`alpha` must be in range [0, 1].")
    if beta < 0.0 or beta > 1.0:
      raise ValueError("`beta` must be in range [0, 1].")
    self.alpha = alpha
    self.beta = beta

  def _result(self):
    """Computes Tversky index from confusion matrix state variables."""
    return tf.math.divide_no_nan(
      self.true_positives,
      self.true_positives + \
      self.alpha * self.false_positives + \
      self.beta * self.false_negatives)

  def get_config(self):
    config = {
      'alpha': self.alpha,
      'beta': self.beta}
    base_config = super().get_config()
    return {**base_config, **config}


@api_util.export("metrics.FBetaScore")
@tf.keras.utils.register_keras_serializable(package="MRI")
class FBetaScore(TverskyIndex):
  r"""Computes F-beta score.

  The F-beta score is the weighted harmonic mean of precision and recall.

  .. math::
    F_{\beta} = (1 + \beta^2) * \frac{\textrm{precision} * \textrm{precision}}{(\beta^2 \cdot \textrm{precision}) + \textrm{recall}}

  Args:
    beta: A `float`. Determines the weight of precision and recall in harmonic
      mean, such that recall is `beta` times as important as precision.
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """ # pylint: disable=line-too-long
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               beta=1.0,
               name='fbeta_score',
               dtype=None):
    if beta <= 0.0:
      raise ValueError("`beta` value should be greater than zero.")
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     alpha=1.0 / (1.0 + beta ** 2),
                     beta=beta ** 2 / (1.0 + beta ** 2),
                     name=name,
                     dtype=dtype)
    # We add underscore to avoid conflict with parent's beta.
    self.beta_ = beta

  def get_config(self):
    config = {
      'beta': self.beta_}
    base_config = super().get_config()
    base_config.pop('alpha')
    base_config.pop('beta')
    return {**base_config, **config}


@api_util.export("metrics.F1Score", "metrics.DiceIndex")
@tf.keras.utils.register_keras_serializable(package="MRI")
class F1Score(FBetaScore):
  r"""Computes F-1 score.

  The F-1 score is the harmonic mean of precision and recall.

  .. math::
    F_1 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """  # pylint: disable=line-too-long
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='f1_score',
               dtype=None):
    super().__init__(num_classes=num_classes,
                     class_id=class_id,
                     average=average,
                     threshold=threshold,
                     beta=1.0,
                     name=name,
                     dtype=dtype)

  def get_config(self):
    base_config = super().get_config()
    base_config.pop('beta')
    return base_config


@api_util.export("metrics.IoU", "metrics.JaccardIndex")
@tf.keras.utils.register_keras_serializable(package="MRI")
class IoU(TverskyIndex):
  r"""Computes the intersection-over-union (IoU) metric.

  Also known as Jaccard index.

  .. math::
    \textrm{IoU} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP} + \textrm{FN}}

  Args:
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               num_classes=None,
               class_id=None,
               average='macro',
               threshold=None,
               name='iou',
               dtype=None):
    super().__init__(num_classes=num_classes,
             class_id=class_id,
             average=average,
             threshold=threshold,
             alpha=1.0,
             beta=1.0,
             name=name,
             dtype=dtype)

  def get_config(self):
    base_config = super().get_config()
    base_config.pop('alpha')
    base_config.pop('beta')
    return base_config


def _update_docstring(docstring):
  doclines = docstring.splitlines()
  args_index = doclines.index("  Args:")
  doclines[args_index+1:args_index+1] = \
      _CONFUSION_METRIC_ARGS_DOCSTRING.splitlines()
  doclines[args_index-1:args_index-1] = \
      _CONFUSION_METRIC_INTRO_DOCTRING.splitlines()
  doclines.extend(_CONFUSION_METRIC_NOTES_DOCTRING.splitlines())
  return '\n'.join(doclines)


ConfusionMetric.__doc__ = _update_docstring(ConfusionMetric.__doc__)
Accuracy.__doc__ = _update_docstring(Accuracy.__doc__)
TruePositiveRate.__doc__ = _update_docstring(TruePositiveRate.__doc__)
TrueNegativeRate.__doc__ = _update_docstring(TrueNegativeRate.__doc__)
PositivePredictiveValue.__doc__ = _update_docstring(
    PositivePredictiveValue.__doc__)
NegativePredictiveValue.__doc__ = _update_docstring(
    NegativePredictiveValue.__doc__)
TverskyIndex.__doc__ = _update_docstring(TverskyIndex.__doc__)
FBetaScore.__doc__ = _update_docstring(FBetaScore.__doc__)
F1Score.__doc__ = _update_docstring(F1Score.__doc__)
IoU.__doc__ = _update_docstring(IoU.__doc__)
