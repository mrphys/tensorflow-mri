# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Tests for normalization layers."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.layers import normalization
from tensorflow_mri.python.util import test_util


class NormalizedTest(test_util.TestCase):
  @test_util.run_all_execution_modes
  def test_normalized_dense(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4)))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

    # test config
    model.get_config()

    # check whether the model variables are present in the
    # trackable list of objects
    checkpointed_object_ids = {
        id(o) for o in trackable_util.list_objects(model)
    }
    for v in model.variables:
      self.assertIn(id(v), checkpointed_object_ids)


if __name__ == '__main__':
  tf.test.main()
