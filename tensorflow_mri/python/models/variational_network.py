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
from tensorflow_mri.python.util import model_util


class VarNet(tf.keras.Model):
  def __init__(self,
               rank,
               num_iterations=10,
               calib_region=0.1 * np.pi,
               reg_network='UNet',
               reg_network_kwargs=None,
               scale_kspace=True,
               estimate_sensitivities=True,
               view_complex_as_real=False,
               kspace_index=None,
               **kwargs):
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(**kwargs)
    self.rank = rank
    self.num_iterations = num_iterations
    self.calib_region = calib_region
    self.reg_network = reg_network
    self.reg_network_kwargs = reg_network_kwargs or {}
    self.scale_kspace = scale_kspace
    self.estimate_sensitivities = estimate_sensitivities
    self.view_complex_as_real = view_complex_as_real
    self.kspace_index = kspace_index

    if self.scale_kspace:
      self._kspace_scaling_layer = kspace_scaling.KSpaceScaling(
          calib_region=self.calib_region,
          kspace_index=self.kspace_index)
    else:
      self._kspace_scaling_layer = None

    if self.estimate_sensitivities:
      self._coil_sensitivities_layer = (
          coil_sensitivities.CoilSensitivityEstimation(
              calib_region=self.calib_region,
              kspace_index=self.kspace_index)
      )

    self._recon_adjoint_layer = recon_adjoint.ReconAdjoint(
        kspace_index=self.kspace_index)

    lsgd_layer_class = data_consistency.LeastSquaresGradientDescent
    lsgd_layers_kwargs = {}

    reg_network_class = model_util.get_nd_model(self.reg_network, rank)
    reg_network_kwargs = dict(
        filters=[32, 64, 128, 256],
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

    if self.scale_kspace:
      x['kspace'] = self._kspace_scaling_layer(x)

    if self.estimate_sensitivities:
      x['sensitivities'] = self._coil_sensitivities_layer(x)

    zfill = self._recon_adjoint_layer(x)

    image = zfill
    for lsgd, reg in zip(self._lsgd_layers, self._reg_layers):
      image = reg(image)
      image = lsgd({'image': image, **x})

    outputs = {'zfill': zfill, 'image': image}
    return outputs


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
