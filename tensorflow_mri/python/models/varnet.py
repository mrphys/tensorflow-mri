# Copyright 2022 University College London. All Rights Reserved.
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

from tensorflow_mri.python.layers import kspace_scaling
from tensorflow_mri.python.util import api_util


class VarNet(tf.keras.Model):
  def __init__(self,
               rank,
               kspace_index=None,
               scale_kspace=True,
               **kwargs):
    super().__init__(**kwargs)
    self.rank = rank
    self.kspace_index = kspace_index
    self.scale_kspace = scale_kspace
    if self.scale_kspace:
      self._kspace_scaling_layer = kspace_scaling.KSpaceScaling(
          kspace_index=self.kspace_index)
    else:
      self._kspace_scaling_layer = None

  def call(self, inputs):
    if self.scale_kspace:
      kspace = self._kspace_scaling_layer(inputs)

    if self.scale_kspace:
      sensitivities = CoilSensitivityEstimation()
    kwargs['sensitivities'] = CoilSensitivities()({'kspace': kspace, **kwargs})

    zfill = ReconAdjoint()({'kspace': kspace, **kwargs})

    image = zfill
    for i in range(num_iterations):
      image = tfmri.models.UNet2D(
          filters=[32, 64, 128],
          kernel_size=3,
          activation=tfmri.activations.complex_relu,
          out_channels=1,
          dtype=tf.complex64,
          name=f'reg_{i}')(image)
      image = tfmri.layers.LeastSquaresGradientDescent(
          operator=tfmri.linalg.LinearOperatorMRI,
          dtype=tf.complex64,
          name=f'lsgd_{i}')(
              {'x': image, 'b': kspace, **kwargs})

    outputs = {'zfill': zfill, 'image': image}
    return tf.keras.Model(inputs=inputs, outputs=outputs)

  def parse_inputs(self, inputs):
    if isinstance(inputs, dict):
      kspace = inputs[self.kspace_index]
      args = ()
      kwargs = {k: inputs[k] for k in inputs.keys() if k != self.kspace_index}
    elif isinstance(inputs, tuple):
      kspace = inputs[0]
      args = inputs[1:]
      kwargs = {}
    else:
      raise TypeError(
          f"inputs must be a dict or a tuple, but got type: {type(inputs)}")
    return kspace, args, kwargs


@api_util.export("models.VarNet1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class VarNet1D(VarNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@api_util.export("models.VarNet2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class VarNet2D(VarNet):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("models.VarNet3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class VarNet3D(VarNet):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)
