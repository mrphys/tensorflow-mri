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
"""Tests for module `math_ops`."""

import itertools

import tensorflow as tf

from tensorflow_mri.python.ops import math_ops


class ScaleMinmaxTest(tf.test.TestCase):
  """Tests for function `scale_by_min_max`."""

  def test_scale_by_min_max(self):
    """Test function `scale_by_min_max`."""

    # Create a random tensor.
    x = tf.random.uniform((5, 3, 4), dtype=tf.dtypes.float32)

    # Create a few parameters to test.
    params = {
      'output_min': (0.0, 1.4, 10.0),
      'output_max': (1.0, 1.5, 16.0)
    }

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        if p['output_min'] >= p['output_max']:
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = math_ops.scale_by_min_max(x, **p)
          continue

        # Test op.
        y = math_ops.scale_by_min_max(x, **p)

        self.assertAllClose(tf.reduce_min(y), p['output_min'])
        self.assertAllClose(tf.reduce_max(y), p['output_max'])


  def test_scale_by_min_max_complex(self):
    """Test function `scale_by_min_max` with complex numbers."""
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

    # Create a random complex tensor.
    x = tf.dtypes.complex(
      tf.random.uniform((5, 3, 4), minval=-1.0, maxval=1.0),
      tf.random.uniform((5, 3, 4), minval=-1.0, maxval=1.0))

    # Create a few parameters to test.
    params = {
      'output_min': (0.0, 1.4, 10.0),
      'output_max': (1.0, 1.5, 16.0)
    }

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        if p['output_min'] >= p['output_max']:
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = math_ops.scale_by_min_max(x, **p)
          continue

        # Test op.
        y = math_ops.scale_by_min_max(x, **p)

        # Check magnitude was scaled.
        self.assertAllClose(
          tf.reduce_min(tf.math.abs(y)), p['output_min'])
        self.assertAllClose(
          tf.reduce_max(tf.math.abs(y)), p['output_max'])

        # Check phase did not change. Weight by magnitude, as for very
        # small magnitudes, phase is noisy.
        mag = tf.math.abs(y)
        self.assertAllClose(
          tf.math.angle(y) * mag, tf.math.angle(x) * mag)
        self.assertAllClose(
          tf.math.angle(y) * mag, tf.math.angle(x) * mag)


if __name__ == '__main__':
  tf.test.main()
