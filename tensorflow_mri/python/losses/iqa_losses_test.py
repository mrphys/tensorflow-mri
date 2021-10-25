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
"""Tests for module `iqa_losses`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.losses import iqa_losses
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import test_util


class SSIMLossTest(test_util.TestCase):
  """Tests for SSIM loss."""

  losses = [
    (iqa_losses.StructuralSimilarityLoss,
     iqa_losses.ssim_loss, image_ops.ssim),
    (iqa_losses.MultiscaleStructuralSimilarityLoss,
     iqa_losses.ssim_multiscale_loss, image_ops.ssim_multiscale)
  ]

  @parameterized.product(loss=losses)
  @test_util.run_in_graph_and_eager_modes
  def test_loss(self, loss): # pylint: disable=missing-param-doc
    """Test the loss function."""
    loss_cls, loss_fn, ref_fn = loss
    y_true, y_pred = self._random_images()

    ref = 1.0 - ref_fn(y_true, y_pred)

    # Test function.
    result_fn = loss_fn(y_true, y_pred)
    self.assertAllClose(result_fn, ref)

    # Test class.
    loss_obj = loss_cls()
    result_cls = loss_obj(y_true, y_pred)
    self.assertAllClose(result_cls, tf.math.reduce_mean(ref))

  def _random_images(self):
    """Generate random images."""
    y_true = tf.random.uniform((4, 192, 192, 2), dtype=tf.float32)
    y_pred = tf.random.uniform((4, 192, 192, 2), dtype=tf.float32)
    return y_true, y_pred



if __name__ == '__main__':
  tf.test.main()
