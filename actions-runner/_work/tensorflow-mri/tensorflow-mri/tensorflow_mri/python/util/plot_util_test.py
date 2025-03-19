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
"""Tests for module `util.plot_util`."""

import matplotlib.animation as ani
import numpy as np

from tensorflow_mri.python.util import plot_util
from tensorflow_mri.python.util import test_util


class PlotImageSequenceTest(test_util.TestCase):
  """Tests for `plot_util.plot_image_sequence`."""
  # TODO(jmontalt): improve testing.
  def test_return_type(self):
    """Tests that the function returns the correct type."""
    # Create a dummy image sequence.
    rng = np.random.default_rng(432)
    images = rng.random((10, 16, 16))
    # Create a dummy figure.
    anim = plot_util.plot_image_sequence(images)
    self.assertIsInstance(anim, ani.ArtistAnimation)

  def test_rgb(self):
    """Test with RGB data."""
    # Create a dummy image sequence.
    rng = np.random.default_rng(432)
    images = rng.random((10, 16, 16, 3))
    # Create a dummy figure.
    plot_util.plot_image_sequence(images)

  def test_fig_title(self):
    """Test figure title."""
    # Create a dummy image sequence.
    rng = np.random.default_rng(432)
    images = rng.random((10, 16, 16, 3))
    # Create a dummy figure.
    plot_util.plot_image_sequence(
        images, fig_title='Title')


class PlotTiledImageSequenceTest(test_util.TestCase):
  """Tests for `plot_util.plot_tiled_image_sequence`."""
  # TODO(jmontalt): improve testing.
  def test_return_type(self):
    """Tests that the function returns the correct type."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16))
    anim = plot_util.plot_tiled_image_sequence(images)
    self.assertIsInstance(anim, ani.ArtistAnimation)

  def test_rgb(self):
    """Test with RGB data."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16, 3))
    plot_util.plot_tiled_image_sequence(images)

  def test_subplot_titles(self):
    """Test subplot titles."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16, 3))
    plot_util.plot_tiled_image_sequence(
        images, subplot_titles=['A', 'B', 'C', 'D'])

  def test_fig_title(self):
    """Test figure title."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16, 3))
    plot_util.plot_tiled_image_sequence(
        images, fig_title='Title')

  def test_tight_layout(self):
    """Test tight layout."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16, 3))
    plot_util.plot_tiled_image_sequence(
        images, grid_shape=(1, 4), layout='tight',
        subplot_titles=['A', 'B', 'C', 'D'])

  def test_tight_bbox(self):
    """Test tight bbox."""
    rng = np.random.default_rng(432)
    images = rng.random((4, 10, 16, 16, 3))
    plot_util.plot_tiled_image_sequence(
        images, grid_shape=(1, 4), layout='tight', bbox_inches='tight',
        subplot_titles=['A', 'B', 'C', 'D'])
