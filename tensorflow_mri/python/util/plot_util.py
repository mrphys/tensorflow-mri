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
"""Plotting utilities."""

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import tensorflow as tf


def plot_image_sequence(images,
                        part=None,
                        cmap='gray',
                        fps=20.0):
  """Plots a sequence of images.

  Args:
    images: A 3D `np.ndarray` of shape `[batch, height, width]`.
    part: An optional `str`. The part to display for complex numbers. One of
      `'abs'`, `'angle'`, `'real'` or `'imag'`. Must be specified if `images`
      has complex dtype.
    cmap: A `str` or `matplotlib.colors.Colormap`_. The colormap used to map
      pixel values to colors. Defaults to `'gray'`.
    fps: A `float`. The number of frames per second. Defaults to 20.

  .. _matplotlib.colors.Colormap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
  """
  images = _preprocess_image(images, part=part, expected_ndim=3)
  fig = plt.figure()
  artists = []
  for image in images:
    artist = plt.imshow(image,
                        cmap=cmap,
                        animated=True)
    artist.axes.axis('off')
    artists.append([artist])

  animation = ani.ArtistAnimation(fig, artists,
                                  interval=int(1000 / fps),
                                  repeat_delay=0,
                                  repeat=True,
                                  blit=True)

  plt.show()


def _preprocess_image(image, part=None, expected_ndim=None):
  image = np.asarray(image)
  if expected_ndim is not None:
    if image.ndim != expected_ndim:
      raise ValueError(
          f"Expected input to be {expected_ndim}-D, "
          f"but got shape {image.shape}")

  if np.iscomplexobj(image):
    if part == 'abs':
      image = np.abs(image)
    elif part == 'angle':
      image = np.angle(image)
    elif part == 'real':
      image = np.real(image)
    elif part == 'imag':
      image = np.imag(image)
    elif part is None:
      raise ValueError("Argument `part` must be specified for complex inputs.")
    else:
      raise ValueError(f"Invalid part: {part}")

  return image


def plot_volume_slices(volumes, rows=1, cols=1, subplot_titles=None):
  """Visualize slices of one or more 3D volumes.

  Plots slices of a 3D volume on a 2D grid. If `vol` is a 3-dimensional array,
  slice `i` is `vol[i, :, :]`.

  If multiple input volumes are passed, they must all have the same number of
  slices.

  The generated figure includes a slider to control the currently visible slice.

  Args:
    volumes: A `tf.Tensor` or `np.ndarray`, or a list thereof. Must have rank 3
      (grayscale) or 4 (RGB).
    rows: The number of subplot rows. Defaults to 1.
    cols: The number of subplot columns. Defaults to 1.
    subplot_titles: A list of `str`. Titles of the subplots.

  Returns:
    A `plotly.graph_objects.Figure`.

  Raises:
    ValueError: If any of the input arguments has an invalid value.
  """
  if isinstance(volumes, (tf.Tensor, np.ndarray)):
    volumes = [volumes]

  num_slices = None
  for idx, vol in enumerate(volumes):
    if isinstance(vol, tf.Tensor):
      volumes[idx] = vol.numpy()

    if not isinstance(vol, np.ndarray):
      volumes[idx] = np.asarray(vol)

    if vol.ndim not in (3, 4):
      raise ValueError(
          "Each element of input `volumes` must have 3 or 4 dimensions.")

    if vol.ndim == 3:
      volumes[idx] = np.stack([vol] * 3, axis=-1)

    if num_slices is None:
      num_slices = vol.shape[0]
    else:
      if vol.shape[0] != num_slices:
        raise ValueError(
            "Each element of input `volumes` must have the same number of "
            "slices.")

  # Create figure.
  fig = go.Figure()
  fig = ps.make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

  def _get_row(idx):
    return idx // cols + 1

  def _get_col(idx):
    return idx % cols + 1

  # Add traces, one for each slice
  for idx, vol in enumerate(volumes):
    for slc in range(vol.shape[0]):
      fig.add_trace(go.Image(z=vol[slc, ...],
                             name=f"slice = {slc}",
                             visible=False),
                    row=_get_row(idx), col=_get_col(idx))

  def _get_trace_index(idx, slc):
    return idx * num_slices + slc

  # Make middle slice visible.
  for idx in range(len(volumes)):
    fig.data[_get_trace_index(idx, num_slices // 2)].visible = True

  # Create and add slider.
  steps = []
  for slc in range(num_slices):
    args = [{"visible": [False] * len(fig.data)}]
    step = dict(
        method="update",
        args=args,
    )
    for idx in range(len(volumes)):
      args[0]["visible"][_get_trace_index(idx, slc)] = True
    steps.append(step)

  sliders = [{
      'active': num_slices // 2,
      'currentvalue': {"prefix": "Slice: "},
      'pad': {'t': 50},
      'steps': steps
  }]

  fig.update_layout(
      sliders=sliders
  )

  return fig
