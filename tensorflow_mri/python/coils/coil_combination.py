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

from tensorflow_mri.python.util import api_util


@api_util.export("coils.combine_coils")
def combine_coils(images, maps=None, coil_axis=-1, keepdims=False, name=None):
  """Combines a multicoil image into a single-coil image.

  Supports sum of squares (when `maps` is `None`) and adaptive combination.

  Args:
    images: A `Tensor`. The input images.
    maps: A `Tensor`. The Wcoil sensitivity maps. This argument is optional.
      If `maps` is provided, it must have the same shape and type as
      `images`. In this case an adaptive coil combination is performed using
      the specified maps. If `maps` is `None`, a simple estimate of `maps`
      is used (ie, images are combined using the sum of squares method).
    coil_axis: An `int`. The coil axis. Defaults to -1.
    keepdims: A `boolean`. If `True`, retains the coil dimension with size 1.
    name: A name for the operation. Defaults to "combine_coils".

  Returns:
    A `Tensor`. The combined images.

  References:
    1. Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P. and
      Mueller, O.M. (1990), The NMR phased array. Magn Reson Med, 16:
      192-225. https://doi.org/10.1002/mrm.1910160203

    2. Bydder, M., Larkman, D. and Hajnal, J. (2002), Combination of signals
      from array coils using image-based estimation of coil sensitivity
      profiles. Magn. Reson. Med., 47: 539-548.
      https://doi.org/10.1002/mrm.10092
  """
  with tf.name_scope(name or "combine_coils"):
    images = tf.convert_to_tensor(images)
    if maps is not None:
      maps = tf.convert_to_tensor(maps)

    if maps is None:
      combined = tf.math.sqrt(
          tf.math.reduce_sum(images * tf.math.conj(images),
                            axis=coil_axis, keepdims=keepdims))

    else:
      combined = tf.math.divide_no_nan(
          tf.math.reduce_sum(images * tf.math.conj(maps),
                            axis=coil_axis, keepdims=keepdims),
          tf.math.reduce_sum(maps * tf.math.conj(maps),
                            axis=coil_axis, keepdims=keepdims))

    return combined
