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
"""Plotting utilities."""

import matplotlib as mpl
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.tight_bbox as tight_bbox
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import tensorflow as tf

from tensorflow_mri.python.util import api_util


@api_util.export("plot.image_sequence")
def plot_image_sequence(images,
                        part=None,
                        cmap='gray',
                        norm=None,
                        fps=20.0,
                        fig_size=None,
                        dpi=None,
                        bg_color='dimgray',
                        layout=None,
                        bbox_inches=None,
                        pad_inches=0.1,
                        fig_title=None):
  """Plots a sequence of images.

  Args:
    images: A 3D `np.ndarray` of shape `[time, height, width]` (scalar values,
      mapped to colors using `cmap`) or a 4D `np.ndarray` of shape
      `[time, height, width, 3]` (RGB) or `[time, height, width, 4]` (RGBA).
    part: An optional `str`. The part to display for complex numbers. One of
      `'abs'`, `'angle'`, `'real'` or `'imag'`. Must be specified if `images`
      has complex dtype.
    cmap: A `str` or `matplotlib.colors.Colormap`_. The colormap used to map
      scalar values to colors. This parameter is ignored for RGB(A) data.
      Defaults to `'gray'`.
    norm: A `matplotlib.colors.Normalize`_. Used to scale scalar data to the
      [0, 1] range before mapping to colors using `cmap`. By default, a linear
      scaling mapping the lowest value to 0 and the highest to 1 is used. This
      parameter is ignored for RGB(A) data.
    fps: A `float`. The number of frames per second. Defaults to 20.
    fig_size: A `tuple` of `float`s. Width and height of the figure in inches.
    dpi: A `float`. The resolution of the figure in dots per inch.
    bg_color: A `color`_. The background color.
    layout: A `str`. One of `None`, `'tight'` (default) or `'constrained'`.
      The layout mechanism. See `matplotlib.figure.Figure`_ for details.
    bbox_inches: A `str` or `matplotlib.transforms.Bbox`_. Bounding box in
      inches: only the given portion of the figure is displayed. If `'tight'`,
      try to figure out the tight bbox of the figure.
    pad_inches: A `float`. Amount of padding around the figure when bbox_inches
      is `'tight'`. Defaults to 0.1.
    fig_title: A `str`. The title of the figure.

  Returns:
    A `matplotlib.animation.ArtistAnimation`_ object.

  .. _color: https://matplotlib.org/stable/tutorials/colors/colors.html
  .. _matplotlib.animation.ArtistAnimation: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.ArtistAnimation.html
  .. _matplotlib.colors.Colormap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
  .. _matplotlib.colors.Normalize: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
  .. _matplotlib.transforms.Bbox: https://matplotlib.org/stable/api/transformations.html#matplotlib.transforms.Bbox
  .. _matplotlib.figure.Figure: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
  """
  images = _preprocess_image(images, part=part, expected_ndim=(3, 4))

  fig = plt.figure(figsize=fig_size, dpi=dpi, facecolor=bg_color, layout=layout)

  artists = []
  for image in images:
    artist = plt.imshow(image,
                        cmap=cmap,
                        norm=norm,
                        animated=True)
    artist.axes.axis('off')
    artists.append([artist])

  if fig_title is not None:
    fig.suptitle(fig_title)

  if bbox_inches is not None:
    if bbox_inches == 'tight':
      bbox_inches = fig.get_tightbbox(fig.canvas.get_renderer())
    if pad_inches is None:
      pad_inches = mpl.rcParams['savefig.pad_inches']
    bbox_inches = bbox_inches.padded(pad_inches)
    tight_bbox.adjust_bbox(fig, bbox_inches)
    fig.set_size_inches(bbox_inches.width, bbox_inches.height)

  animation = ani.ArtistAnimation(fig, artists,
                                  interval=int(1000 / fps),
                                  repeat_delay=0,
                                  repeat=True,
                                  blit=True)

  return animation


@api_util.export("plot.tiled_image_sequence")
def plot_tiled_image_sequence(images,
                              part=None,
                              cmap='gray',
                              norm=None,
                              fps=20.0,
                              fig_size=None,
                              dpi=None,
                              bg_color='dimgray',
                              layout=None,
                              bbox_inches=None,
                              pad_inches=0.1,
                              aspect=1.77,  # 16:9
                              grid_shape=None,
                              fig_title=None,
                              subplot_titles=None):
  r"""Plots one or more image sequences in a grid.

  Args:
    images: A 4D `np.ndarray` of shape `[batch, time, height, width]` (scalar
      values, mapped to colors using `cmap`) or a 5D `np.ndarray` of shape
      `[batch, time, height, width, 3]` (RGB) or
      `[batch, time, height, width, 4]` (RGBA).
    part: An optional `str`. The part to display for complex numbers. One of
      `'abs'`, `'angle'`, `'real'` or `'imag'`. Must be specified if `images`
      has complex dtype.
    cmap: A `str` or `matplotlib.colors.Colormap`_. The colormap used to map
      scalar values to colors. This parameter is ignored for RGB(A) data.
      Defaults to `'gray'`.
    norm: A `matplotlib.colors.Normalize`_. Used to scale scalar data to the
      [0, 1] range before mapping to colors using `cmap`. By default, a linear
      scaling mapping the lowest value to 0 and the highest to 1 is used. This
      parameter is ignored for RGB(A) data.
    fps: A `float`. The number of frames per second. Defaults to 20.
    fig_size: A `tuple` of `float`s. Width and height of the figure in inches.
    dpi: A `float`. The resolution of the figure in dots per inch.
    bg_color: A `color`_. The background color.
    layout: A `str`. One of `None`, `'tight'` (default) or `'constrained'`.
      The layout mechanism. See `matplotlib.figure.Figure`_ for details.
    bbox_inches: A `str` or `matplotlib.transforms.Bbox`_. Bounding box in
      inches: only the given portion of the figure is displayed. If `'tight'`,
      try to figure out the tight bbox of the figure.
    pad_inches: A `float`. Amount of padding around the figure when bbox_inches
      is `'tight'`. Defaults to 0.1.
    aspect: A `float`. The desired aspect ratio of the overall figure. Ignored
      if `grid_shape` is specified.
    grid_shape: A `tuple` of `float`s. The number of rows and columns in the
      grid. If `None`, the grid shape is computed from `aspect`.
    fig_title: A `str`. The title of the figure.
    subplot_titles: A `list` of `str`s. The titles of the subplots.

  Returns:
    A `matplotlib.animation.ArtistAnimation`_ object.

  .. _color: https://matplotlib.org/stable/tutorials/colors/colors.html
  .. _matplotlib.animation.ArtistAnimation: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.ArtistAnimation.html
  .. _matplotlib.colors.Colormap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
  .. _matplotlib.colors.Normalize: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
  .. _matplotlib.transforms.Bbox: https://matplotlib.org/stable/api/transformations.html#matplotlib.transforms.Bbox
  .. _matplotlib.figure.Figure: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
  """
  images = _preprocess_image(images, part=part, expected_ndim=(4, 5))
  num_tiles, num_frames, image_rows, image_cols = images.shape[:4]

  # Compute the number of rows and cols for tile.
  if grid_shape is not None:
    grid_rows, grid_cols = grid_shape
  else:
    grid_rows, grid_cols = _compute_grid_shape(
        num_tiles, (image_rows, image_cols), aspect)

  fig, axs = plt.subplots(grid_rows, grid_cols, squeeze=False,
                          figsize=fig_size, dpi=dpi,
                          facecolor=bg_color, layout=layout)

  artists = []
  for frame_idx in range(num_frames):  # For each frame.
    frame_artists = []
    for row, col in np.ndindex(grid_rows, grid_cols):  # For each tile.
      tile_idx = row * grid_cols + col  # Index of current tile.
      ax = axs[row, col]  # Current axes.
      # Set axis properties. This is always done, regardless of whether there's
      # actually anything to display on this tile.
      ax.axis('off')
      if tile_idx >= num_tiles:
        # This tile is empty.
        continue
      # Set the title.
      if subplot_titles is not None:
        ax.set_title(subplot_titles[tile_idx])
      # Get image for this tile and frame.
      image = images[tile_idx, frame_idx, :, :]
      # Render image.
      artist = ax.imshow(image,
                         cmap=cmap,
                         norm=norm,
                         animated=True)
      frame_artists.append(artist)
    artists.append(frame_artists)

  if fig_title is not None:
    fig.suptitle(fig_title)

  if bbox_inches is not None:
    if bbox_inches == 'tight':
      bbox_inches = fig.get_tightbbox(fig.canvas.get_renderer())
      bbox_inches = bbox_inches.padded(pad_inches)
    tight_bbox.adjust_bbox(fig, bbox_inches)
    fig.set_size_inches(bbox_inches.width, bbox_inches.height)

  animation = ani.ArtistAnimation(fig, artists,
                                  interval=int(1000 / fps),
                                  repeat_delay=0,
                                  repeat=True,
                                  blit=True)

  return animation


@api_util.export("plot.tiled_image")
def plot_tiled_image(images,
                     part=None,
                     cmap='gray',
                     norm=None,
                     fig_size=None,
                     dpi=None,
                     bg_color='dimgray',
                     layout=None,
                     bbox_inches=None,
                     pad_inches=0.1,
                     aspect=1.77,  # 16:9
                     grid_shape=None,
                     fig_title=None,
                     subplot_titles=None):
  r"""Plots one or more images in a grid.

  Args:
    images: A 3D `np.ndarray` of shape `[batch, height, width]` (scalar
      values, mapped to colors using `cmap`) or a 4D `np.ndarray` of shape
      `[batch, height, width, 3]` (RGB) or `[batch, height, width, 4]` (RGBA).
    part: An optional `str`. The part to display for complex numbers. One of
      `'abs'`, `'angle'`, `'real'` or `'imag'`. Must be specified if `images`
      has complex dtype.
    cmap: A `str` or `matplotlib.colors.Colormap`_. The colormap used to map
      scalar values to colors. This parameter is ignored for RGB(A) data.
      Defaults to `'gray'`.
    norm: A `matplotlib.colors.Normalize`_. Used to scale scalar data to the
      [0, 1] range before mapping to colors using `cmap`. By default, a linear
      scaling mapping the lowest value to 0 and the highest to 1 is used. This
      parameter is ignored for RGB(A) data.
    fig_size: A `tuple` of `float`s. Width and height of the figure in inches.
    dpi: A `float`. The resolution of the figure in dots per inch.
    bg_color: A `color`_. The background color.
    layout: A `str`. One of `None`, `'tight'` (default) or `'constrained'`.
      The layout mechanism. See `matplotlib.figure.Figure`_ for details.
    bbox_inches: A `str` or `matplotlib.transforms.Bbox`_. Bounding box in
      inches: only the given portion of the figure is displayed. If `'tight'`,
      try to figure out the tight bbox of the figure.
    pad_inches: A `float`. Amount of padding around the figure when bbox_inches
      is `'tight'`. Defaults to 0.1.
    aspect: A `float`. The desired aspect ratio of the overall figure. Ignored
      if `grid_shape` is specified.
    grid_shape: A `tuple` of `float`s. The number of rows and columns in the
      grid. If `None`, the grid shape is computed from `aspect`.
    fig_title: A `str`. The title of the figure.
    subplot_titles: A `list` of `str`s. The titles of the subplots.

  Returns:
    A `list` of `matplotlib.image.AxesImage`_ objects.

  .. _color: https://matplotlib.org/stable/tutorials/colors/colors.html
  .. _matplotlib.colors.Colormap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
  .. _matplotlib.colors.Normalize: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html
  .. _matplotlib.transforms.Bbox: https://matplotlib.org/stable/api/transformations.html#matplotlib.transforms.Bbox
  .. _matplotlib.figure.Figure: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
  .. _matplotlib.image.AxesImage: https://matplotlib.org/stable/api/image_api.html#matplotlib.image.AxesImage
  """
  images = _preprocess_image(images, part=part, expected_ndim=(3, 4))
  num_tiles, image_rows, image_cols = images.shape[:3]

  # Compute the number of rows and cols for tile.
  if grid_shape is not None:
    grid_rows, grid_cols = grid_shape
  else:
    grid_rows, grid_cols = _compute_grid_shape(
        num_tiles, (image_rows, image_cols), aspect)

  fig, axs = plt.subplots(grid_rows, grid_cols, squeeze=False,
                          figsize=fig_size, dpi=dpi,
                          facecolor=bg_color, layout=layout)

  artists = []
  for row, col in np.ndindex(grid_rows, grid_cols):  # For each tile.
    tile_idx = row * grid_cols + col  # Index of current tile.
    ax = axs[row, col]  # Current axes.
    # Set axis properties. This is always done, regardless of whether there's
    # actually anything to display on this tile.
    ax.axis('off')
    if tile_idx >= num_tiles:
      # This tile is empty.
      continue
    # Set the title.
    if subplot_titles is not None:
      ax.set_title(subplot_titles[tile_idx])
    # Get image for this tile and frame.
    image = images[tile_idx, :, :]
    # Render image.
    artist = ax.imshow(image,
                       cmap=cmap,
                       norm=norm,
                       animated=True)
    artists.append(artist)
  artists.append(artists)

  if fig_title is not None:
    fig.suptitle(fig_title)

  if bbox_inches is not None:
    if bbox_inches == 'tight':
      bbox_inches = fig.get_tightbbox(fig.canvas.get_renderer())
      bbox_inches = bbox_inches.padded(pad_inches)
    tight_bbox.adjust_bbox(fig, bbox_inches)
    fig.set_size_inches(bbox_inches.width, bbox_inches.height)

  return artists


@api_util.export("plot.show")
def show(*args, **kwargs):
  """Displays all open figures.

  This function is an alias for `matplotlib.pyplot.show`_.

  For the parameters, see `matplotlib.pyplot.show`_.

  .. _matplotlib.pyplot.show: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
  """
  return plt.show(*args, **kwargs)


@api_util.export("plot.close")
def close(*args, **kwargs):
  """Closes a figure window.

  This function is an alias for `matplotlib.pyplot.close`_.

  For the parameters, see `matplotlib.pyplot.close`_.

  .. _matplotlib.pyplot.close: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html
  """
  return plt.close(*args, **kwargs)


def _preprocess_image(image, part=None, expected_ndim=None):  # pylint: disable=missing-param-doc
  """Preprocesses an image."""
  image = np.asarray(image)
  if expected_ndim is not None:
    if isinstance(expected_ndim, int):
      expected_ndim = (expected_ndim,)
    if image.ndim not in expected_ndim:
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


def _compute_grid_shape(num_tiles, tile_shape, aspect):
  """Computes the grid shape for an image tile.

  Args:
    num_tiles: An `int`. The number of tiles in the grid.
    tile_shape: A `tuple` of `int`s. The shape of each tile.
    aspect: A `float`. The desired aspect ratio of the overall figure.

  Returns:
    A `tuple` of `int`s. The number of rows and columns in the grid.
  """
  # The aspect ratio of a single tile.
  tile_aspect = tile_shape[1] / tile_shape[0]

  # The approximate aspect ratio of the grid.
  grid_aspect = aspect / tile_aspect

  # Now find rows and columns for this aspect ratio.
  grid_rows = int(np.sqrt(num_tiles / grid_aspect) + 0.5)  # Round.
  grid_cols = (num_tiles + grid_rows - 1) // grid_rows     # Ceil.

  return grid_rows, grid_cols


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
