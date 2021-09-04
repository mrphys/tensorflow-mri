# Copyright 2021 University College London. All Rights Reserved.
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
"""Tests for module `confusion_metrics`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.metrics import confusion_metrics
from tensorflow_mri.python.utils import test_utils


class ConfusionMetricTest(test_utils.TestCase):
  """Tests for confusion metrics."""

  names = [
    'Accuracy',
    'TruePositiveRate',
    'TrueNegativeRate',
    'PositivePredictiveValue',
    'NegativePredictiveValue',
    'Precision',
    'Recall',
    'Sensitivity',
    'Specificity',
    'Selectivity',
    'TverskyIndex',
    'FBetaScore',
    'F1Score',
    'IoU'
  ]

  @parameterized.parameters(*names)
  def test_binary_metric(self, name): # pylint: disable=missing-param-doc
    """Test binary metric."""
    metric = getattr(confusion_metrics, name)

    y_true = [[1, 1, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    y_pred = [[0.4, 0.7, 0.3],
              [0.9, 0.2, 0.8],
              [0.6, 0.8, 0.1],
              [0.2, 0.3, 0.9]]

    y_pred = tf.expand_dims(y_pred, -1)
    y_true = tf.expand_dims(y_true, -1)

    tp, tn, fp, fn = 4, 5, 2, 1
    result = self._compute_result(name, tp, tn, fp, fn)

    kwargs = self._get_kwargs(name)
    m = metric(**kwargs)
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

    # Check serialization.
    m = metric.from_config(m.get_config())
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

  @parameterized.parameters(*names)
  def test_binary_metric_custom_threshold(self, name): # pylint: disable=missing-param-doc
    """Test binary metric with a custom threshold."""
    metric = getattr(confusion_metrics, name)

    y_true = [[1, 1, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    y_pred = [[0.4, 0.7, 0.3],
              [0.9, 0.2, 0.8],
              [0.6, 0.8, 0.1],
              [0.2, 0.3, 0.9]]

    y_pred = tf.expand_dims(y_pred, -1)
    y_true = tf.expand_dims(y_true, -1)

    tp, tn, fp, fn = 3, 6, 1, 2
    result = self._compute_result(metric.__name__, tp, tn, fp, fn)

    kwargs = self._get_kwargs(metric.__name__)
    m = metric(threshold=0.75, **kwargs)
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

    # Check serialization.
    m = metric.from_config(m.get_config())
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

  @parameterized.product(name=names,
                         class_id=[None, 0, 1, 2],
                         average=[None, 'macro', 'micro'])
  def test_multiclass_metric(self, name, class_id, average): # pylint: disable=missing-param-doc
    """Test multiclass metric."""
    metric = getattr(confusion_metrics, name)

    y_true = [[1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 0, 1]]
    y_pred = [[0.8, 0.1, 0.1],
              [0.2, 0.5, 0.3],
              [0.2, 0.7, 0.1],
              [0.1, 0.3, 0.6],
              [0.1, 0.3, 0.6],
              [0.7, 0.1, 0.2],
              [0.5, 0.2, 0.3],
              [0.1, 0.4, 0.5]]

    tps, tns, fps, fns = (3, 1, 2), (4, 5, 5), (0, 1, 1), (1, 1, 0)
    if class_id is None:
      if average in (None, 'macro', 'weighted'):
        result = tuple(
            self._compute_result(
                name, tp, tn, fp, fn) for tp, tn, fp, fn in zip(
                    tps, tns, fps, fns))
        if average == 'macro':
          result = sum(result) / len(result)
        if average == 'weighted':
          pass
      elif average == 'micro':
        tp, tn, fp, fn = map(sum, (tps, tns, fps, fns))
        result = self._compute_result(name, tp, tn, fp, fn)
    else:
      result = self._compute_result(name,
                                    tps[class_id], tns[class_id],
                                    fps[class_id], fns[class_id])

    kwargs = self._get_kwargs(name)
    m = metric(num_classes=3, class_id=class_id, average=average, **kwargs)
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

    # Check serialization.
    m = metric.from_config(m.get_config())
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), result)

  @parameterized.parameters(*names)
  def test_metric_reset(self, name): # pylint: disable=missing-param-doc
    """Test metric reset."""
    metric = getattr(confusion_metrics, name)
    y_true = tf.concat([tf.ones([4, 1]), tf.zeros([4, 1])], 0)
    y_pred = y_true
    m = metric()
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(), 1.0)
    m.reset_state()
    self.assertAllClose(m.result(), 0.0)

  def _get_kwargs(self, name):
    kwargs = {}
    if name == 'TverskyIndex':
      kwargs.update({'alpha': 0.3, 'beta': 0.7})
    if name == 'FBetaScore':
      kwargs.update({'beta': 0.5})
    return kwargs

  def _compute_result(self, name, tp, tn, fp, fn):
    return {
      'Accuracy': (tp + tn) / (tp + tn + fp + fn),
      'TruePositiveRate': tp / (tp + fn),
      'TrueNegativeRate': tn / (tn + fp),
      'PositivePredictiveValue': tp / (tp + fp),
      'NegativePredictiveValue': tn / (tn + fn),
      'Precision': tp / (tp + fp),
      'Recall': tp / (tp + fn),
      'Sensitivity': tp / (tp + fn),
      'Specificity': tn / (tn + fp),
      'Selectivity': tn / (tn + fp),
      'TverskyIndex': tp / (tp + 0.3 * fp + 0.7 * fn),
      'FBetaScore': 1.25 * tp / (1.25 * tp + fp + 0.25 * fn),
      'F1Score': tp / (tp + 0.5 * fp + 0.5 * fn),
      'IoU': tp / (tp + fp + fn)
    }[name]
