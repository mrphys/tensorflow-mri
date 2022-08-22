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

import tensorflow as tf


class GraphLikeModel(tf.keras.Model):
  """A model with graph-like structure.

  Adds a method `functional` that returns a functional model with the same
  structure as the current model. Functional models have some advantages over
  subclassing as described in
  https://www.tensorflow.org/guide/keras/functional#when_to_use_the_functional_api.
  """
  def functional(self, inputs):
    return tf.keras.Model(inputs, self.call(inputs))
