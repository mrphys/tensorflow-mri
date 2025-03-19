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
"""Model utilities."""

from tensorflow_mri.python.models import conv_blocks
from tensorflow_mri.python.models import conv_endec


def get_nd_model(name, rank):
  """Get an N-D model object.

  Args:
    name: A `str`. The name of the requested model.
    rank: An `int`. The rank of the requested model.

  Returns:
    A `tf.keras.Model` object.

  Raises:
    ValueError: If the requested model is unknown to TFMRI.
  """
  try:
    return _ND_MODELS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err


_ND_MODELS = {
    ('ConvBlock', 1): conv_blocks.ConvBlock1D,
    ('ConvBlock', 2): conv_blocks.ConvBlock2D,
    ('ConvBlock', 3): conv_blocks.ConvBlock3D,
    ('UNet', 1): conv_endec.UNet1D,
    ('UNet', 2): conv_endec.UNet2D,
    ('UNet', 3): conv_endec.UNet3D
}
