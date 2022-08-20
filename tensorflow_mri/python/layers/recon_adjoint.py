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
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util
from tensorflow_mri.python.util import keras_util


DOCSTRING = string.Template(
  """${rank}-D adjoint reconstruction layer.

  This layer reconstructs a signal using the adjoint of the specified system
  operator.

  This layer's `inputs` differ depending on whether `operator` is a class or an
  instance.

  - If `operator` is a class, then `inputs` must be a `dict` containing both
    the inputs to the operator's constructor (e.g., *k*-space mask, trajectory,
    coil sensitivities, etc...) and the input to the operator's `transform`
    method (usually, the *k*-space data). The value at `kspace_index` will be
    passed to the operator's `transform` method. Any other values in `inputs`
    will be passed to the operator's constructor.

  - If `operator` is an instance, then `inputs` only contains the input to the
    operator's `transform` method (usually, the *k*-space data). In this case,
    `inputs` must be a `tf.Tensor` which will be passed unmodified to the
    operator's `transform` method.

  Args:
    expand_channel_dim: A `boolean`. Whether to expand the channel dimension.
      If `True`, the output has shape `[*batch_shape, ${dim_names}, 1]`.
      If `False`, the output has shape `[*batch_shape, ${dim_names}]`.
      Defaults to `True`.
    operator: A subclass of `tfmri.linalg.LinearOperator` or an instance
      thereof. The system operator. This object may be a class or an instance.

      - If `operator` is a class, then a new instance will be created during
        each evaluation of `call`. The constructor will be passed the arguments
        in `inputs` except `kspace_index`.
      - If `operator` is an instance, then it will be used as is.

      Defaults to `tfmri.linalg.LinearOperatorMRI`.
    kspace_index: A `str`. The key of `inputs` containing the *k*-space data.
      Defaults to `None`, which takes the first element of `inputs`.
  """)


class ReconAdjoint(linear_operator_layer.LinearOperatorLayer):
  def __init__(self,
               rank,
               expand_channel_dim=True,
               operator=linear_operator_mri.LinearOperatorMRI,
               kspace_index=None,
               **kwargs):
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)
    self.rank = rank
    self.expand_channel_dim = expand_channel_dim

  def call(self, inputs):
    kspace, operator = self.parse_inputs(inputs)
    image = recon_adjoint.recon_adjoint(kspace, operator)
    if self.expand_channel_dim:
      image = tf.expand_dims(image, axis=-1)
    return image

  def get_config(self):
    config = {
        'expand_channel_dim': self.expand_channel_dim
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


ReconAdjoint2D.__doc__ = DOCSTRING.substitute(
    rank=2, dim_names='height, width')
ReconAdjoint3D.__doc__ = DOCSTRING.substitute(
    rank=3, dim_names='depth, height, width')


ReconAdjoint2D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)
ReconAdjoint3D.__signature__ = doc_util.get_nd_layer_signature(ReconAdjoint)
