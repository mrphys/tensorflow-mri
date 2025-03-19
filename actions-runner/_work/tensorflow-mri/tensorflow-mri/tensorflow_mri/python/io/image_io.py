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
"""Image I/O."""

import tensorflow as tf
import tensorflow_io as tfio

from tensorflow_mri.python.util import api_util


@api_util.export("io.encode_gif")
def encode_gif(image, loop_count=0, name=None):
  """Encode a uint8 tensor to gif image with extensions.

  Based on `tfio.image.encode_gif`, but also supports grayscale images and
  looping extension.

  Args:
    image: A uint8 tensor with shape [N, H, W, C], where C must be 1 or 3
      (grayscale, RGB).
    loop_count: Number of times the animation should be looped. Set to 0 for
      endless looping.
    name: A name for the operation.

  Returns:
    A tensor of type `string` with the encoded bytes.

  Raises:
    ValueError: If `loop_count` is not in the range `[0, 65535]`.
  """
  with tf.name_scope(name or "encode_gif"):

    # Check channels.
    channels = tf.shape(image)[-1]
    tf.debugging.Assert(
        tf.math.logical_or(
            tf.math.equal(channels, 1), tf.math.equal(channels, 3)),
        ["`encode_gif` only supports grayscale (channels=1) "
          "and RGB (channels=3) encoding: ", channels])

    # Convert to RGB.
    if tf.math.equal(channels, 1):
      image = tf.image.grayscale_to_rgb(image)

    # Create encoded GIF.
    encoded_image = tfio.image.encode_gif(image)

    # Looping application extension (NETSCAPE 2.0). For details:
    # http://www.vurdalakov.net/misc/gif/netscape-looping-application-extension
    loop_ext = b''
    loop_ext += b'\x21' # Extension introducer.
    loop_ext += b'\xFF' # Extension label: application extension.
    loop_ext += b'\x0B' # Block size.
    loop_ext += b'NETSCAPE' # Application identifier.
    loop_ext += b'2.0' # Application authentication code.
    loop_ext += b'\x03' # Size of data sub-block.
    loop_ext += b'\x01' # Sub-block identifier.
    try:
      loop_ext += loop_count.to_bytes(2, byteorder='little')
    except OverflowError as err:
      raise ValueError((
          "Invalid `loop_count` value: must be between 0 and 65535."
          "Received: {}").format(loop_count)) from err
    loop_ext += b'\x00' # Block terminator.

    # Number of bytes in encoded image.
    num_bytes = tf.strings.length(encoded_image)

    # Add looping application extension to encoded GIF.
    encoded_image = tf.strings.join([
        tf.strings.substr(encoded_image, 0, num_bytes - 1),
        loop_ext,
        tf.strings.substr(encoded_image, num_bytes, 1)])

    return encoded_image
