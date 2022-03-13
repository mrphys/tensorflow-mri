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
"""Tensorboard image summaries."""

import functools

import tensorflow as tf
import tensorboard as tb

from tensorflow_mri.python.io import image_io
from tensorflow_mri.python.util import api_util


@api_util.export("summary.gif")
def gif(name,
        data,
        step=None,
        max_outputs=3,
        loop_count=0,
        description=None):
  """Write an animated GIF image summary.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will
      be this name prefixed by any active name scopes.
    data: A tensor representing pixel data with shape [k, t, h, w, c], where
      k is the number of images, t is the number of frames, h and w are
      the height and width of the images, and c is the number of channels,
      which should be 1 or 3 (grayscale, RGB). Any of the dimensions may
      be statically unknown (i.e., None). Floating point data will be
      clipped to the range [0, 1]. Other data types will be clipped into
      an allowed range for safe casting to uint8, using
      `tf.image.convert_image_dtype`.
    step: Monotonic step value for this summary. If omitted, this defaults
      to `tf.summary.experimental.get_step()`, which must not be None.
    max_outputs: Maximum number of animated images to be emitted at each
      step. When more than `max_outputs` images are provided, the first
      `max_outputs` images will be used and the rest silently discarded.
    loop_count: Number of times the animation should be looped. Set to 0 for
      endless looping.
    description: A string containing a description for this summary.

  Returns:
    True if successful, or False if no summary was emitted because no
    default summary writer was available.
  """
  summary_metadata = tb.plugins.image.metadata.create_summary_metadata(
      display_name=None, description=description)
  summary_scope = (
      getattr(tf.summary.experimental, "summary_scope", None)
      or tf.summary.summary_scope)

  with summary_scope(
      name, "gif_summary", values=[data, max_outputs, step]
  ) as (tag, _):
    # Defer image encoding preprocessing by passing it as a callable to write(),
    # wrapped in a LazyTensorCreator for backwards compatibility, so that we
    # only do this work when summaries are actually written.
    @tb.util.lazy_tensor_creator.LazyTensorCreator
    def lazy_tensor():
      tf.debugging.assert_rank(data, 5)
      tf.debugging.assert_non_negative(max_outputs)
      images = tf.image.convert_image_dtype(data, tf.uint8, saturate=True)
      limited_images = images[:max_outputs]
      encoded_images = tf.map_fn(
          functools.partial(image_io.encode_gif, loop_count=loop_count),
          limited_images,
          dtype=tf.string,
          name="map_encode_gif",
      )
      # Workaround for map_fn returning float dtype for an empty elems
      # input.
      encoded_images = tf.cond(
          tf.shape(input=encoded_images)[0] > 0,
          lambda: encoded_images,
          lambda: tf.constant([], tf.string),
      )
      image_shape = tf.shape(input=images)
      dimensions = tf.stack(
          [
              tf.as_string(image_shape[2], name="width"),
              tf.as_string(image_shape[1], name="height"),
          ],
          name="dimensions",
      )
      return tf.concat([dimensions, encoded_images], axis=0) # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

  # To ensure that image encoding logic is only executed when summaries
  # are written, we pass callable to `tensor` parameter.
  return tf.summary.write(
      tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata)
