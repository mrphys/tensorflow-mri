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

import numpy as np
import tensorflow as tf
import warnings

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.layers import coil_sensitivities
from tensorflow_mri.python.layers import data_consistency
from tensorflow_mri.python.layers import kspace_scaling
from tensorflow_mri.python.layers import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import keras_util
from tensorflow_mri.python.util import layer_util
from tensorflow_mri.python.util import model_util


class VarNet(model_util.GraphLikeModel):
  def __init__(self,
               rank,
               num_iterations=10,
               calib_region=0.1 * np.pi,
               reg_network='UNet',
               reg_network_kwargs=None,
               sens_network='UNet',
               sens_network_kwargs=None,
               compress_coils=True,
               coil_compression_kwargs=None,
               scale_kspace=True,
               estimate_sensitivities=True,
               view_complex_as_real=False,
               return_multicoil=False,
               return_zerofilled=False,
               return_sensitivities=False,
               kspace_index=None,
               **kwargs):
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(**kwargs)
    self.rank = rank
    self.num_iterations = num_iterations
    self.calib_region = calib_region
    self.reg_network = reg_network
    self.reg_network_kwargs = reg_network_kwargs or {}
    self.sens_network = sens_network
    self.sens_network_kwargs = sens_network_kwargs or {}
    self.compress_coils = compress_coils
    self.coil_compression_kwargs = coil_compression_kwargs or {}
    self.scale_kspace = scale_kspace
    self.estimate_sensitivities = estimate_sensitivities
    self.view_complex_as_real = view_complex_as_real
    self.return_zerofilled = return_zerofilled
    self.return_multicoil = return_multicoil
    self.return_sensitivities = return_sensitivities
    self.kspace_index = kspace_index

    if self.compress_coils:
      coil_compression_kwargs = _get_default_coil_compression_kwargs()
      coil_compression_kwargs.update(self.coil_compression_kwargs)
      self._coil_compression_layer = layer_util.get_nd_layer(
          'CoilCompression', self.rank)(
              calib_region=self.calib_region,
              coil_compression_kwargs=coil_compression_kwargs,
              kspace_index=self.kspace_index)

    if self.scale_kspace:
      self._kspace_scaling_layer = kspace_scaling.KSpaceScaling(
          calib_region=self.calib_region,
          kspace_index=self.kspace_index)
    else:
      self._kspace_scaling_layer = None

    if self.estimate_sensitivities:
      self._coil_sensitivities_layer = layer_util.get_nd_layer(
          'CoilSensitivityEstimation', self.rank)(
              calib_region=self.calib_region,
              sens_network=self.sens_network,
              sens_network_kwargs=self.sens_network_kwargs,
              kspace_index=self.kspace_index)

    self._recon_adjoint_layer = recon_adjoint.ReconAdjoint(
        kspace_index=self.kspace_index)

    lsgd_layer_class = data_consistency.LeastSquaresGradientDescent
    lsgd_layers_kwargs = {}

    reg_network_class = model_util.get_nd_model(self.reg_network, rank)
    reg_network_kwargs = dict(
        filters=[32, 64, 128],
        kernel_size=3,
        activation=complex_activations.complex_relu,
        out_channels=2 if self.view_complex_as_real else 1,
        dtype=tf.float32 if self.view_complex_as_real else tf.complex64
    )

    self._lsgd_layers = [
        lsgd_layer_class(**lsgd_layers_kwargs, name=f'lsgd_{i}')
        for i in range(self.num_iterations)]
    self._reg_layers = [
        reg_network_class(**reg_network_kwargs, name=f'reg_{i}')
        for i in range(self.num_iterations)]

  def call(self, inputs):
    x = inputs

    if 'image_shape' in x:
      image_shape = x['image_shape']
      if image_shape.shape.rank == 2:
        warnings.warn(
            f"Layer {self.name} got a batch of image shapes. "
            f"It is not possible to reconstruct images with "
            f"different shapes in the same batch. "
            f"If the input batch has more than one element, "
            f"only the first image shape will be used. "
            f"It is up to you to verify if this behavior is correct.")
        x['image_shape'] = tf.ensure_shape(
            image_shape[0], image_shape.shape[1:])

    if self.compress_coils:
      x['kspace'] = self._coil_compression_layer(x)

    if self.scale_kspace:
      x['kspace'] = self._kspace_scaling_layer(x)

    if self.estimate_sensitivities:
      x['sensitivities'] = self._coil_sensitivities_layer(x)

    zfill = self._recon_adjoint_layer(x)

    image = zfill
    for lsgd, reg in zip(self._lsgd_layers, self._reg_layers):
      image = reg(image)
      image = lsgd({'image': image, **x})

    outputs = {'image': image}

    if self.return_zerofilled:
      outputs['zerofilled'] = zfill
    if self.return_multicoil:
      outputs['multicoil'] = (
          tf.expand_dims(image, -(self.rank + 2)) *
          tf.expand_dims(x['sensitivities'], -1))
    if self.return_sensitivities:
      outputs['sensitivities'] = x['sensitivities']

    return outputs


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


def _get_default_coil_compression_kwargs():
  return {
      'out_coils': 12
  }
