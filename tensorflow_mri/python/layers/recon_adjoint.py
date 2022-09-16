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
"""Adjoint reconstruction layer."""

import string

import tensorflow as tf

from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


class ReconAdjoint(tf.keras.layers.Layer):
  r"""${rank}D adjoint reconstruction layer.

  This layer reconstructs a signal using the adjoint of the specified system
  operator.

  Given measurement data $b$ generated by a linear system $A$ such that
  $Ax = b$, this function estimates the corresponding signal $x$ as
  $x = A^H b$, where $A$ is the specified linear operator.

  ```{note}
  This function is part of the family of
  [universal operators](https://mrphys.github.io/tensorflow-mri/guide/universal/),
  a set of functions and classes designed to work flexibly with any linear
  system.
  ```

  ```{seealso}
  This is the Keras layer equivalent of `tfmri.recon.adjoint_universal`.
  ```

  ## Inputs

  This layer's `call` method expects the following inputs:

  - data: A `tf.Tensor` of real or complex dtype. The measurement data $b$.
    Its shape must be compatible with `operator.range_shape`.
  - operator: A `tfmri.linalg.LinearOperator` representing the system operator
    $A$. Its range shape must be compatible with `data.shape`.

  ```{attention}
  Both `data` and `operator` should be passed as part of the first positional
  `inputs` argument, either as as a `tuple` or as a `dict`, in order to take
  advantage of this argument's special rules. For more information, see
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#call.
  ```

  ## Outputs

  This layer's `call` method returns a `tf.Tensor` containing the reconstructed
  signal. Has the same dtype as `data` and shape
  `batch_shape + operator.domain_shape`. `batch_shape` is the result of
  broadcasting the batch shapes of `data` and `operator`.

  Args:
    expand_channel_dim: A `boolean`. Whether to expand the channel dimension.
      If `True`, output has shape `batch_shape + operator.domain_shape + [1]`.
      If `False`, output has shape `batch_shape + operator.domain_shape`.
      Defaults to `True`.
    reinterpret_complex: A `boolean`. Whether to reinterpret a complex-valued
      output image as a dual-channel real image. Defaults to `False`.
    **kwargs: Keyword arguments to be passed to base layer
      `tf.keras.layers.Layer`.
  """
  def __init__(self,
               rank,
               expand_channel_dim=False,
               reinterpret_complex=False,
               **kwargs):
    super().__init__(**kwargs)
    self.rank = rank  # Currently unused.
    self.expand_channel_dim = expand_channel_dim
    self.reinterpret_complex = reinterpret_complex

  def call(self, inputs):  # pylint: arguments-differ
    data, operator = parse_inputs(inputs)
    image = recon_adjoint.recon_adjoint(data, operator)
    if self.expand_channel_dim:
      image = tf.expand_dims(image, axis=-1)
    if self.reinterpret_complex and image.dtype.is_complex:
      image = math_ops.view_as_real(image, stacked=False)
    return image

  def get_config(self):
    base_config = super().get_config()
    config = {
        'expand_channel_dim': self.expand_channel_dim,
        'reinterpret_complex': self.reinterpret_complex
    }
    return {**base_config, **config}


def parse_inputs(inputs):
  def _parse_inputs(data, operator):
    return data, operator
  if isinstance(inputs, tuple):
    return _parse_inputs(*inputs)
  if isinstance(inputs, dict):
    return _parse_inputs(**inputs)
  raise ValueError('inputs must be a tuple or dict')


@api_util.export("layers.ReconAdjoint2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ReconAdjoint2D(ReconAdjoint):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("layers.ReconAdjoint3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ReconAdjoint3D(ReconAdjoint):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


ReconAdjoint2D.__doc__ = string.Template(
    ReconAdjoint.__doc__).safe_substitute(rank=2)
ReconAdjoint3D.__doc__ = string.Template(
    ReconAdjoint.__doc__).safe_substitute(rank=3)


ReconAdjoint2D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)
ReconAdjoint3D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)