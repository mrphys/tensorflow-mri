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

from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util
from tensorflow_mri.python.util import keras_util


class ReconAdjoint(linear_operator_layer.LinearOperatorLayer):
  r"""${rank}-D adjoint reconstruction layer.

  This layer reconstructs a signal using the adjoint of the specified system
  operator.

  This layer can use the same operator instance in each invocation or
  instantiate a new operator in each invocation, depending on whether the
  operator remains constant or depends on the inputs to the layer.

  - If you wish to use the same operator instance during each call, initialize
    the layer by setting `operator` to be an instance of a linear operator.
    Then the call `inputs` are simply the input to the operator's `transform`
    method (usually, the *k*-space data).

  - If you wish to instantiate a new operator during each call (e.g., because
    the operator itself depends on the layer's inputs), initialize the layer by
    setting `operator` to be a function that returns an instance of a linear
    operator (or a string if you wish to use one of the built-in operators).
    In this case the call `inputs` must be a `dict` containing both the inputs
    to the operator's `transform` method (specified by `input_indices`) and the
    any other inputs needed by the `operator` function to instantiate the
    linear operator.

  Args:
    expand_channel_dim: A `boolean`. Whether to expand the channel dimension.
      If `True`, the output has shape `[*batch_shape, ${dim_names}, 1]`.
      If `False`, the output has shape `[*batch_shape, ${dim_names}]`.
      Defaults to `True`.
    reinterpret_complex: A `boolean`. Whether to reinterpret a complex-valued
      output image as a dual-channel real image. Defaults to `False`.
    operator: A `tfmri.linalg.LinearOperator`, or a callable that returns a
      `tfmri.linalg.LinearOperator`, or a `str` containing the name of one
      of the built-in linear operators. The system operator.

      - If `operator` is a `tfmri.linalg.LinearOperator`, the operator will be
        used as is during each invocation of the layer's `call` method.
      - If `operator` is a generic callable, it will be called during each
        invocation of the layer's `call` method to construct a new
        `tfmri.linalg.LinearOperator`. The callable will be passed all of the
        arguments in `inputs` except `kspace_index`.
      - If `operator` is a `str`, it must be the name of one of the built-in
        linear operators. See the `tfmri.linalg` module for a list of built-in
        operators. The operator will be constructed during each invocation of
        `call` using the arguments in `inputs` except `kspace_index`.

      Defaults to `'MRI'`, which creates a new `tfmri.linalg.LinearOperatorMRI`
      during each invocation of `call`.
    kspace_index: A `str`. The key of `inputs` containing the *k*-space data.
      Defaults to `None`, which takes the first element of `inputs`.
  """
  def __init__(self,
               rank,
               expand_channel_dim=True,
               reinterpret_complex=False,
               operator='MRI',
               kspace_index=None,
               **kwargs):
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)
    self.rank = rank
    self.expand_channel_dim = expand_channel_dim
    self.reinterpret_complex = reinterpret_complex

  def call(self, inputs):
    kspace, operator = self.parse_inputs(inputs)
    image = recon_adjoint.recon_adjoint(kspace, operator)
    if self.expand_channel_dim:
      image = tf.expand_dims(image, axis=-1)
    if self.reinterpret_complex:
      image = math_ops.view_as_real(image, stacked=False)
    return image

  def get_config(self):
    config = {
        'expand_channel_dim': self.expand_channel_dim,
        'reinterpret_complex': self.reinterpret_complex
    }
    base_config = super().get_config()
    input_indices = base_config.pop('input_indices')
    config['kspace_index'] = (
        input_indices[0] if input_indices is not None else None)
    return {**config, **base_config}


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


ReconAdjoint2D.__doc__ = string.Template(ReconAdjoint.__doc__).substitute(
    rank=2, dim_names='height, width')
ReconAdjoint3D.__doc__ = string.Template(ReconAdjoint.__doc__).substitute(
    rank=3, dim_names='depth, height, width')


ReconAdjoint2D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)
ReconAdjoint3D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)
