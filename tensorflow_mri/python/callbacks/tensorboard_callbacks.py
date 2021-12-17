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
"""Tensorboard callbacks."""

import os

import tensorflow as tf

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.summary import image_summary
from tensorflow_mri.python.util import nest_util


class TensorBoardImages(tf.keras.callbacks.Callback):
  """Keras callback to write image summaries to Tensorboard.

  Supports 2D and 3D images. Inputs are expected to have shape NHWC for 2D
  images or NDHWC for 3D images.

  Subclasses may override the `display_image` method to customize how the images
  to display are generated. This method should accept three arguments (`x`, `y`
  and `y_pred` for a single example) and returns the image to be written to
  TensorBoard for that example. If not specified, the default function will
  concatenate `x`, `y` and `y_pred` horizontally (only suitable if `x`, `y` and
  `y_pred` are all tensors with equal shape and dtype).

  Args:
    x: The input samples. Can be a NumPy array, a TensorFlow tensor, a
      `tf.data.Dataset` or a `tf.keras.utils.Sequence`.
    log_dir: The directory where to save the log files to be parsed by
      TensorBoard.
    images_freq: Frequency (in epochs) at which images will be written to the
      logs. Defaults to 1.
    max_images: Maximum number of images to be written at each step. Defaults
      to 3.
    summary_name: Name for the image summaries. Defaults to `'val_images'`.
    volume_mode: Specifies how to save 3D images. Must be `None`, `'gif'` or an
      integer. If `None` (default), inputs are expected to be 2D images. In
      `'gif'` mode, each 3D volume is stored as an animated GIF. If an integer,
      only the corresponding slice is saved.
  """
  def __init__(self,
               x,
               log_dir='logs',
               images_freq=1,
               max_images=3,
               summary_name='images',
               volume_mode=None):
    """Initialize callback."""
    super().__init__()
    # Set attributes.
    self.x = x
    self.log_dir = log_dir
    self.update_freq = images_freq
    self.max_images = max_images
    self.summary_name = summary_name
    self.volume_mode = volume_mode


  def on_epoch_end(self, epoch, logs=None): # pylint: disable=unused-argument
    """Called at the end of an epoch."""
    # This function is used to save the images when in training mode.
    if epoch % self.update_freq == 0:
      self._write_image_summaries(step=epoch)


  def _write_image_summaries(self, step=0):
    """Write image summaries to TensorBoard logs.

    Args:
      step: Step value.
    """
    # Open writer.
    image_dir = os.path.join(self.log_dir, 'image')
    self.file_writer = tf.summary.create_file_writer(image_dir)

    images = []

    # For each batch.
    for batch in self.x:
      # Extract batch components.
      x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)

      # Use `predict_step` rather than calling the model directly. This
      # might perform additional actions such as resetting the states of
      # stateful layers.
      y_pred = self.model.predict_step(batch)

      # Unbatch.
      x = nest_util.unstack_nested_tensors(x)
      y = nest_util.unstack_nested_tensors(y)
      y_pred = nest_util.unstack_nested_tensors(y_pred)

      # Create display images.
      images.extend(list(map(self.display_image, x, y, y_pred)))

      # Check how many outputs we have processed.
      if len(images) >= self.max_images:
        break

    # Stack all the images.
    images = tf.stack(images)

    # Keep only selected slice, if requested.
    if isinstance(self.volume_mode, int):
      images = images[:, self.volume_mode, ...]

    # Write images.
    with self.file_writer.as_default(step=step):
      if self.volume_mode == 'gif':
        image_summary.gif(self.summary_name,
                          images,
                          max_outputs=self.max_images)
      else:
        tf.summary.image(self.summary_name,
                         images,
                         max_outputs=self.max_images)

    # Close writer.
    self.file_writer.close()


  def display_image(self, x, y, y_pred):
    """Returns the image to be displayed for each example.

    By default, the image is created by concatenating horizontally `x`, `y` and
    `y_pred`.

    Args:
      x: Model input (single example).
      y: Ground truth (single example).
      y_pred: Prediction (single example).

    Returns:
      The image to display.
    """
    x = _canonicalize_single_tensor(x, arg_name='x')
    y = _canonicalize_single_tensor(y, arg_name='y')
    y_pred = _canonicalize_single_tensor(y_pred, arg_name='y_pred')
    return tf.concat([x, y, y_pred], -2)


class TensorBoardImagesRecon(TensorBoardImages):
  """Specializes `TensorBoardImages` for image reconstruction models.

  Displays the ground truth image and the predicted image.

  Args:
    x: The input samples. Can be a NumPy array, a TensorFlow tensor, a
      `tf.data.Dataset` or a `tf.keras.utils.Sequence`.
    log_dir: The directory where to save the log files to be parsed by
      TensorBoard.
    images_freq: Frequency (in epochs) at which images will be written to the
      logs. Defaults to 1.
    max_images: Maximum number of images to be written at each step. Defaults
      to 3.
    summary_name: Name for the image summaries. Defaults to `'val_images'`.
    volume_mode: Specifies how to save 3D images. Must be `None`, `'gif'` or an
      integer. If `None` (default), inputs are expected to be 2D images. In
      `'gif'` mode, each 3D volume is stored as an animated GIF. If an integer,
      only the corresponding slice is saved.
    complex_part: A `str`. One of `'real'`, `'imag'`, `'abs'` or `'angle'`.
      Specifies which part of a complex input should be displayed.
  """
  def __init__(self,
               x,
               log_dir='logs',
               images_freq=1,
               max_images=3,
               summary_name='images',
               volume_mode=None,
               complex_part=None):
    """Initialize callback."""
    super().__init__(x,
                     log_dir=log_dir,
                     images_freq=images_freq,
                     max_images=max_images,
                     summary_name=summary_name,
                     volume_mode=volume_mode)
    self.complex_part = complex_part

  def display_image(self, x, y, y_pred):
    """Returns the image to be displayed for each example."""
    y = _canonicalize_single_tensor(y, arg_name='y')
    y_pred = _canonicalize_single_tensor(y_pred, arg_name='y_pred')

    if y.dtype.is_complex:
      if self.complex_part is None:
        raise ValueError('`complex_part` must be specified for complex inputs.')
      y = image_ops.extract_and_scale_complex_part(
          y, self.complex_part, max_val=1.0)
    if y_pred.dtype.is_complex:
      if self.complex_part is None:
        raise ValueError('`complex_part` must be specified for complex inputs.')
      y_pred = image_ops.extract_and_scale_complex_part(
          y_pred, self.complex_part, max_val=1.0)

    return tf.concat([y, y_pred], -2)


def _canonicalize_single_tensor(arg, arg_name=None):
  """Canonicalizes the input as a single tensor."""
  if not isinstance(arg, (tuple, tf.Tensor)):
    raise ValueError(f"`{arg_name}` must be a single tensor, got: {arg}.")
  if isinstance(arg, tuple):
    if len(arg) != 1 or not isinstance(arg[0], tf.Tensor):
      raise ValueError(f"`{arg_name}` must be a single tensor, got: {arg}.")
    return arg[0]
  return arg
