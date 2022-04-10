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
"""Tests for module `confusion_losses`."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmri

from tensorflow_mri.python.util import test_util


class ConfusionLossTest(test_util.TestCase):
  """Tests for module confusion losses."""
  names = [
      'FocalTverskyLoss',
      'TverskyLoss',
      'F1Loss',
      'IoULoss'
  ]

  shapes = [
      [2, 4, 3],
      [4, 3],
      [3, 6, 6, 4]
  ]

  average = [
      'micro',
      'macro',
      'weighted'
  ]

  default_args = {
      'FocalTverskyLoss': {
          'alpha': 0.3,
          'beta': 0.7,
          'gamma': 0.75,
          'epsilon': 1e-5
      },
      'TverskyLoss': {
          'alpha': 0.3,
          'beta': 0.7,
          'epsilon': 1e-5
      },
      'F1Loss': {
          'epsilon': 1e-5
      },
      'IoULoss': {
          'epsilon': 1e-5
      }
  }

  args = {
      'FocalTverskyLoss': {
          'alpha': 0.2,
          'beta': 0.8,
          'gamma': 0.5,
          'epsilon': 1e-3
      },
      'TverskyLoss': {
          'alpha': 0.2,
          'beta': 0.8,
          'epsilon': 1e-3
      },
      'F1Loss': {
          'epsilon': 1e-3
      },
      'IoULoss': {
          'epsilon': 1e-3
      }
  }

  @parameterized.product(name=names, shape=shapes, average=average,
                         use_args=[True, False],
                         use_class_weights=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_confusion_loss(self, name, shape, average,  # pylint: disable=missing-param-doc
                          use_args, use_class_weights):
    """Tests a confusion loss."""
    if use_class_weights:
      class_weights = np.random.default_rng(7).uniform(0, 1, shape[-1])
    else:
      class_weights = None

    loss = getattr(tfmri.losses, name)

    y_true = np.random.default_rng(4).binomial(1, 0.5, shape)
    y_pred = np.random.default_rng(4).uniform(0, 1, shape)

    cm = self._compute_confusion_matrix(y_true, y_pred, average)
    expected = self._compute_loss(name, cm, use_args)
    expected = self._compute_average(expected, cm, average, class_weights)
    expected = np.mean(expected, axis=0)

    args = self.args[name] if use_args else {}
    args['average'] = average
    args['class_weights'] = class_weights
    l = loss(**args)
    self.assertAllClose(expected, l(y_true, y_pred))

    # Check serialization.
    l = loss.from_config(l.get_config())
    self.assertAllClose(expected, l(y_true, y_pred))


  def _compute_confusion_matrix(self, y_true, y_pred, average):
    if average == 'micro':
      axis = tuple(range(1, y_true.ndim))
    else:
      axis = tuple(range(1, y_true.ndim - 1))
    tp = np.sum(y_true * y_pred, axis=axis)
    tn = np.sum((1 - y_true) * (1 - y_pred), axis=axis)
    fp = np.sum((1 - y_true) * y_pred, axis=axis)
    fn = np.sum(y_true * (1 - y_pred), axis=axis)
    return tp, tn, fp, fn


  def _compute_loss(self, name, cm, use_args):  # pylint: disable=missing-function-docstring
    tp, _, fp, fn = cm
    args = self.args[name] if use_args else self.default_args[name]
    eps = args['epsilon']
    if name == 'FocalTverskyLoss':
      return (1.0 - ((tp + eps) / \
          (tp + args['alpha'] * fp + args['beta'] * fn + eps))) ** args['gamma']
    if name == 'TverskyLoss':
      return 1.0 - ((tp + eps) / \
          (tp + args['alpha'] * fp + args['beta'] * fn + eps))
    if name == 'F1Loss':
      return 1.0 - ((tp + eps) / (tp + 0.5 * fp + 0.5 * fn + eps))
    if name == 'IoULoss':
      return 1.0 - ((tp + eps) / (tp + 1.0 * fp + 1.0 * fn + eps))
    raise ValueError(f"Invalid loss name: {name}")


  def _compute_average(self, value, cm, average, class_weights):  # pylint: disable=missing-function-docstring
    tp, _, _, fn = cm
    if average == 'micro':
      return value
    if average == 'macro':
      return np.mean(value, axis=-1)
    if average == 'weighted':
      if class_weights is None:
        class_weights = (tp + fn) / np.sum(tp + fn, axis=-1, keepdims=True)
      return np.sum(value * class_weights, axis=-1)
    raise ValueError(f"Invalid average mode: {average}")


if __name__ == '__main__':
  tf.test.main()
