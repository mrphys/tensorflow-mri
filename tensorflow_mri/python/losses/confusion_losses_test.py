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
      'FocalTverskyLoss'
  ]

  @parameterized.parameters(names)
  def test_confusion_loss(self, name):
    """Tests a confusion loss."""
    loss = getattr(tfmri.losses, name)

    y_true = np.array([[1, 1, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    y_pred = np.array([[0.4, 0.7, 0.3],
                      [0.9, 0.2, 0.8],
                      [0.6, 0.8, 0.1],
                      [0.2, 0.3, 0.9]])

    l = loss()

    cm = _compute_confusion_matrix(y_true, y_pred)
    expected = _compute_loss(name, *cm)

    expected = np.mean(expected, axis=-1)
    expected = np.mean(expected, axis=0)
    self.assertAllClose(expected, l(y_true, y_pred))


def _compute_confusion_matrix(y_true, y_pred):
  tp = y_true * y_pred
  tn = (1 - y_true) * (1 - y_pred)
  fp = (1 - y_true) * y_pred
  fn = y_true * (1 - y_pred)
  return tp, tn, fp, fn


def _compute_loss(name, tp, tn, fp, fn):
  return {
      'TverskyLoss': 1.0 - ((tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5)),
      'FocalTverskyLoss': (1.0 - ((tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5))) ** (1 / 1.33)
  }[name]


if __name__ == '__main__':
  tf.test.main()
