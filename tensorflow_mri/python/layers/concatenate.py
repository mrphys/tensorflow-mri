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

from tensorflow_mri.python.ops import array_ops


@tf.keras.utils.register_keras_serializable(package="MRI")
class ResizeAndConcatenate(tf.keras.layers.Layer):
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          f"Layer {self.__class__.__name__} expects a list of inputs. "
          f"Received: {inputs}")

    ref = inputs[0]
    others = inputs[1:]
    others = [tf.ensure_shape(
                  array_ops.resize_with_crop_or_pad(tensor, tf.shape(ref)),
                  ref.shape) for tensor in others]

    return tf.concat([ref] + others, axis=self.axis)
