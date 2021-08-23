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
"""Tests for module `iqa_metrics`."""

import tensorflow as tf

from tensorflow_mri.python.metrics import iqa_metrics
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.utils import test_utils


class IQAMetricTest(tf.test.TestCase):
  """Tests for IQA metrics."""

  pairs = [
    (iqa_metrics.PeakSignalToNoiseRatio, image_ops.psnr),
    (iqa_metrics.StructuralSimilarity, image_ops.ssim),
    (iqa_metrics.MultiscaleStructuralSimilarity, image_ops.ssim_multiscale)
  ]

  @test_utils.parameterized_test(pair=pairs)
  def test_mean_metric(self, pair):
    """Test mean metric."""
    metric, fn = pair
    y_true, y_pred = self._random_images()
    m = metric()
    m.update_state(y_true, y_pred)
    self.assertAllClose(m.result(),
                        tf.math.reduce_mean(fn(y_true, y_pred)))


  def _random_images(self):

    rng = tf.random.Generator.from_seed(0)
    y_true = rng.uniform((4, 192, 192, 8))
    y_pred = rng.uniform((4, 192, 192, 8))
    return y_true, y_pred


if __name__ == '__main__':
  tf.test.main()
