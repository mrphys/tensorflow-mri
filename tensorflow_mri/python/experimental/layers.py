# Copyright 2021 University College London. All Rights Reserved.
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
"""Experimental layers."""

import tensorflow as tf

from tensorflow_mri.python.layers import preproc_layers


class KSpaceResamplingWithMotion(preproc_layers.KSpaceResampling):
  """K-space resampling layer with motion.

  Accepts the same arguments as `KSpaceResampling`, plus those defined below.

  Args:
    max_displacement: A list of floats defining the maximum displacement (in
      pixels) along each spatial dimension. If a scalar is given, the same
      displacement will be used in all dimensions. Each element can also be a
      callable which takes no arguments and returns a float. This may be used
      to obtain a different number for each call. Defaults to 0.0.
    views_per_segment: The number of views per segment. All views in the same
      segment share the same motion state. Defaults to 1.
  """
  def __init__(self, *args, **kwargs):
    """Initializes the layer."""
    # Get arguments specific to this subclass.
    self._max_displacement = kwargs.pop('max_displacement', 0.0)
    self._views_per_segment = kwargs.pop('views_per_segment', 1)

    super().__init__(*args, **kwargs)

    if self._phases is not None:
      raise ValueError(f"Layer {self.__class__.__name__} does not support "
                       f"multiple phases.")

    if (isinstance(self._max_displacement, (int, float)) or
        callable(self._max_displacement)):
      self._max_displacement = [self._max_displacement] * self._rank

    if len(self._max_displacement) != self._rank:
      raise ValueError(f"`max_displacement` must be of length equal to the "
                       f"number of spatial dimensions, but got "
                       f"{self._max_displacement}")

    # Number of segments: ceil(views / views_per_segment).
    self._segments = (
        (self._views + self._views_per_segment - 1) // self._views_per_segment)

  def process_kspace(self, kspace):
    """Adds motion to k-space.

    Args:
      kspace: The input k-space.

    Returns:
      The processed k-space.

    Raises:
      ValueError: If `kspace` does not have rank 2.
    """
    # `kspace` should have shape [channels, samples].
    if kspace.shape.rank != 2:
      raise ValueError(f"Expected `kspace` to have rank 2, but got shape: "
                       f"{kspace.shape}")

    channels = kspace.shape[0]
    kspace = tf.reshape(kspace, [channels, self._views, -1])
    points = tf.reshape(self._points, [self._views, -1, self._rank])

    # Get maximum displacement.
    max_displacement = [d() if callable(d) else d
                        for d in self._max_displacement]

    # Get a unique position for each segment (normalized, 0.0 means no
    # displacement, 1.0 means maximum displacement).
    rng = tf.random.get_global_generator()
    positions = rng.uniform([self._segments, 1, 1], minval=0.0, maxval=1.0)

    # Calculate multi-dimensional displacement (in pixels) along each direction
    # for each segment. This will have shape [segments, 1, rank], where rank is
    # the number of spatial dimensions. The singleton dimension is the samples
    # dimension. We rely on broadcasting here, as we apply the same displacement
    # for all the samples in the same view.
    displacements = positions * max_displacement

    # Repeat each element by the number of views per segment, then truncate
    # in case we exceeded the number of views.
    displacements = tf.repeat(displacements,
                              [self._views_per_segment] * self._segments,
                              axis=0)
    displacements = displacements[:self._views, ...]

    # Apply phase modulation.
    kspace *= tf.exp(tf.complex(
        tf.constant(0.0, dtype=tf.as_dtype(self.dtype).real_dtype),
        tf.math.reduce_sum(points * displacements, -1)))

    kspace = tf.reshape(kspace, [channels, -1])

    return kspace

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'max_displacement': self._max_displacement,
        'views_per_segment': self._views_per_segment
    }
    base_config = super().get_config()
    return {**base_config, **config}
