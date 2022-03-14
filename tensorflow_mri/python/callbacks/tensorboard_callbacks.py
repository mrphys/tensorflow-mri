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
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import nest_util


@api_util.export("callbacks.TensorBoardImages")
class TensorBoardImages(tf.keras.callbacks.Callback):
  """Keras callback to write image summaries to Tensorboard.

  Supports 2D and 3D images. Inputs are expected to have shape NHWC for 2D
  images or NDHWC for 3D images.

  Subclasses may override the `display_image` method to customize how the images
  to display are generated. This method should accept three arguments
  (`features`, `labels` and `predictions` for a single example) and returns the
  image to be written to TensorBoard for that example. Alternatively, the user
  may pass the `display_fn` parameter to override this function.

  The default implementation of `display_image` concatenates the `features`,
  `labels` and `predictions` along axis -2 (horizontally). The concatenation
  axis can be changed with the parameter `concat_axis`. Additionally, the
  features/labels/predictions to display can be selected using the
  `feature_keys`, `label_keys` and `prediction_keys` parameters. By default,
  all features/labels/predictions are displayed.

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
    display_fn: A callable. A function which accepts three arguments
      (features, labels and predictions for a single example) and returns the
      image to be written to TensorBoard. Overrides the default function, which
      concatenates selected features, labels and predictions according to
      `concat_axis`, `feature_keys`, `label_keys`, `prediction_keys` and
      `complex_part`.
    concat_axis: An `int`. The axis along which to concatenate
      features/labels/predictions. Defaults to -2.
    feature_keys: A list of `str` or `int` specifying which features to
      select for display. If `None`, all features are selected. Pass an empty
      list to select no features.
    label_keys: A list of `str` or `int` specifying which labels to select
      for display. If `None`, all labels are selected. Pass an empty list to
      select no labels.
    prediction_keys: A list of `str` or `int` specifying which predictions
      to select for display. If `None`, all predictions are selected. Pass an
      empty list to select no predictions.
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
               display_fn=None,
               concat_axis=-2,
               feature_keys=None,
               label_keys=None,
               prediction_keys=None,
               complex_part=None):
    """Initialize callback."""
    super().__init__()
    self.x = x
    self.log_dir = log_dir
    self.update_freq = images_freq
    self.max_images = max_images
    self.summary_name = summary_name
    self.volume_mode = volume_mode
    self.display_fn = display_fn or self.display_image
    self.concat_axis = concat_axis
    self.feature_keys = feature_keys
    self.label_keys = label_keys
    self.prediction_keys = prediction_keys
    self.complex_part = complex_part

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
      images.extend(list(map(self.display_fn, x, y, y_pred)))

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

  def display_image(self, features, labels, predictions):
    """Returns the image to be displayed for each example.

    By default, the image is created by concatenating horizontally `features`,
    `labels` and `predictions`.

    Args:
      features: Features (model inputs for a single example).
      labels: Labels (ground truth for a single example).
      predictions: Predictions (model outputs for a single example).

    Returns:
      The image to display.
    """
    # Independently concatenate individual features, labels and predictions.
    cat_features = _select_and_concatenate(
        features, self.feature_keys, self.concat_axis, self.complex_part,
        arg_name='features')
    cat_labels = _select_and_concatenate(
        labels, self.label_keys, self.concat_axis, self.complex_part,
        arg_name='labels')
    cat_predictions = _select_and_concatenate(
        predictions, self.prediction_keys, self.concat_axis, self.complex_part,
        arg_name='predictions')

    # Concatenate features, labels and predictions.
    tensors = []
    if cat_features is not None:
      tensors.append(cat_features)
    if cat_labels is not None:
      tensors.append(cat_labels)
    if cat_predictions is not None:
      tensors.append(cat_predictions)
    if tensors:
      return tf.concat(tensors, self.concat_axis)

    return None


def _select_and_concatenate(arg, keys, axis, complex_part, arg_name=None):  # pylint: disable=missing-param-doc
  """Selects and concatenates the tensors for the given keys."""
  if not isinstance(arg, (tuple, dict, tf.Tensor)):
    raise TypeError(
        f"`{arg_name}` must be a tensor, tuple or dict, got: {arg}.")

  # Select specified values and concatenate them.
  if isinstance(arg, (tuple, dict)):
    if keys is None:
      tensors = list(arg.values()) if isinstance(arg, dict) else arg
    else:
      tensors = [arg[key] for key in keys]
    if not tensors:
      return None
    for index, tensor in enumerate(tensors):
      tensors[index] = _prepare_for_concat(tensor, complex_part)
    out = tf.concat(tensors, axis)
  else:  # Input is a tensor, so nothing to select/concatenate.
    out = _prepare_for_concat(arg, complex_part)

  return out


def _prepare_for_concat(tensor, complex_part):  # pylint: disable=missing-param-doc
  """Prepares a tensor for concatenation."""
  if tensor is None:
    return None
  # If tensor is complex, convert to real.
  if tensor.dtype.is_complex:
    if complex_part is None:
      raise ValueError(
          "`complex_part` must be specified for complex inputs.")
    tensor = image_ops.extract_and_scale_complex_part(
        tensor, complex_part, max_val=1.0)
  # Cast to common type (float32).
  return tf.cast(tensor, _CONCAT_DTYPE)


_CONCAT_DTYPE = tf.float32
