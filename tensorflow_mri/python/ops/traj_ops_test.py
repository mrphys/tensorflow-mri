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
"""Tests for module `traj_ops`."""

import itertools
import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util


class DensityGridTest():
  """Tests for `density_grid`."""
  @parameterized.product(transition_type=['linear', 'quadratic', 'hann'])
  def test_density(self, transition_type):  # pylint: disable=missing-function-docstring
    expected = {
        'linear': [
            0.3, 0.3, 0.34, 0.44, 0.54, 0.64, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.64, 0.54, 0.44, 0.34, 0.3],
        'quadratic': [
            0.3, 0.3, 0.304, 0.349, 0.444, 0.589, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.589, 0.444, 0.349, 0.304, 0.3],
        'hann': [
            0.3, 0.3, 0.30978873, 0.40920186, 0.56180346, 0.6782013, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.6782013 , 0.56180346, 0.40920186, 0.30978873, 0.3]
    }
    density = traj_ops.density_grid([16],
                                    inner_density=0.7, outer_density=0.3,
                                    inner_cutoff=0.3, outer_cutoff=0.8,
                                    transition_type=transition_type)
    self.assertAllClose(expected[transition_type], density)


class RadialTrajectoryTest(test_util.TestCase):
  """Radial trajectory tests."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/traj_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_waveform(self):
    """Test radial waveform."""
    waveform = traj_ops.radial_waveform(base_resolution=128)
    self.assertAllClose(waveform, self.data['radial/waveform'])

  @test_util.run_in_graph_and_eager_modes
  def test_waveform_3d(self):
    """Test radial waveform (3D)."""
    waveform = traj_ops.radial_waveform(base_resolution=128, rank=3)
    ref = self.data['radial/waveform']
    self.assertAllClose(waveform,
                        np.concatenate([np.zeros((256, 1)), ref], axis=-1))

  @test_util.run_in_graph_and_eager_modes
  def test_trajectory(self):
    """Test radial trajectory."""
    phi = 2.0 / (1.0 + tf.sqrt(5.0))
    phi_7 = 1.0 / (phi + 7)
    phi_6 = 1.0 / (phi + 6)

    trajectory = traj_ops.radial_trajectory(base_resolution=128,
                                            views=8,
                                            phases=4,
                                            ordering='golden')
    self.assertAllClose(trajectory,
                        _rotate_pi(self.data['radial/trajectory/golden']),
                        rtol=1e-4, atol=1e-4)

    trajectory = traj_ops.radial_trajectory(base_resolution=128,
                                            views=8,
                                            phases=4,
                                            ordering='tiny')

    # Verify rotation angles.
    spacing = _calculate_rotation_angles_from_radial_trajectory(trajectory)
    self.assertAllClose(
        spacing, phi_7 * 2.0 * np.pi * tf.ones([31]))
    # Verify central sample.
    self.assertAllClose(trajectory[..., 128, :], tf.zeros([4, 8, 2]))

    trajectory = traj_ops.radial_trajectory(base_resolution=128,
                                            views=8,
                                            phases=4,
                                            ordering='tiny',
                                            tiny_number=6)

    # Verify rotation angles.
    spacing = _calculate_rotation_angles_from_radial_trajectory(trajectory)
    self.assertAllClose(
        spacing, tf.broadcast_to(phi_6 * 2.0 * np.pi, spacing.shape))
    # Verify central sample.
    self.assertAllClose(trajectory[..., 128, :], tf.zeros([4, 8, 2]))

  @test_util.run_in_graph_and_eager_modes
  def test_trajectory_3d(self):
    """Test 3D radial trajectory."""
    traj1 = traj_ops.radial_trajectory(base_resolution=64,
                                       views=100,
                                       phases=None,
                                       ordering='sphere_archimedean',
                                       angle_range='half')

    ref = self.data['radial/trajectory/sphere_archimedean/half/1']
    self.assertAllClose(traj1, tf.concat([-ref[..., 0:1], ref[..., 1:3]], -1))

    traj2 = traj_ops.radial_trajectory(base_resolution=64,
                                       views=25,
                                       phases=4,
                                       ordering='sphere_archimedean',
                                       angle_range='half')

    ref = self.data['radial/trajectory/sphere_archimedean/half/2']
    self.assertAllClose(traj2, tf.concat([-ref[..., 0:1], ref[..., 1:3]], -1))

  @test_util.run_in_graph_and_eager_modes
  def test_density_2d(self):
    """Test 2D radial density."""
    dens = traj_ops.radial_density(base_resolution=4,
                                   views=5,
                                   phases=None,
                                   ordering='linear')
    ref = tf.convert_to_tensor(
        [[0.0625, 0.08333334, 0.125, 0.25, 1., 0.25, 0.125, 0.08333334]] * 5)
    self.assertAllClose(dens, ref * 20)

  def test_density_3d(self):
    """Test 3D radial density."""
    with self.assertRaisesRegex(
        ValueError, "`sphere_archimedean` is not implemented"):
      traj_ops.radial_density(base_resolution=64,
                              views=100,
                              phases=None,
                              ordering='sphere_archimedean',
                              angle_range='half')

  @parameterized.product(phases=[None, 2])
  def test_angles(self, phases): # pylint: disable=missing-param-doc
    """Test angles."""
    # pylint: disable=protected-access

    phi = 2.0 / (1.0 + tf.sqrt(5.0))
    phi_7 = 1.0 / (phi + 7)
    phi_6 = 1.0 / (phi + 6)

    def _calc(inc, max_, intl=False):
      res = tf.expand_dims(tf.math.floormod(
          tf.range(4 * (phases or 1), dtype=tf.float32) * inc, max_), -1)
      if phases is not None:
        if intl:
          res = tf.transpose(tf.reshape(res, [4, phases, 1]), [1, 0, 2])
        else:
          res = tf.reshape(res, [phases, 4, 1])
      return res

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='linear', angle_range='full')
    self.assertAllClose(angles, _calc(
        0.25 / (phases or 1) * math.pi * 2.0, math.pi * 2.0, intl=True))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='linear', angle_range='half')
    self.assertAllClose(angles, _calc(
        0.25 / (phases or 1) * math.pi, math.pi, intl=True))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='golden', angle_range='full')
    self.assertAllClose(angles, _calc(phi * math.pi * 2.0, math.pi * 2.0))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='tiny', angle_range='full')
    self.assertAllClose(angles, _calc(phi_7 * math.pi * 2.0, math.pi * 2.0))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='golden', angle_range='half')
    self.assertAllClose(angles, _calc(phi * math.pi, math.pi))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='tiny', angle_range='half')
    self.assertAllClose(angles, _calc(phi_7 * math.pi, math.pi))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='golden_half', angle_range='full')
    self.assertAllClose(angles, _calc(phi * math.pi, 2.0 * math.pi))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='tiny_half', angle_range='full')
    self.assertAllClose(angles, _calc(phi_7 * math.pi, 2.0 * math.pi))

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='tiny_half', angle_range='full',
        tiny_number=6)
    self.assertAllClose(angles, _calc(phi_6 * math.pi, 2.0 * math.pi))

    with self.assertRaisesRegex(
        ValueError, "`tiny_number` must be an integer >= 2"):
      angles = traj_ops._trajectory_angles(
          4, phases=phases, ordering='tiny', angle_range='full', tiny_number=1)

    angles = traj_ops._trajectory_angles(
        4, phases=phases, ordering='sphere_archimedean', angle_range='half')
    if phases is None:
      ref_angles = [[0.       , 0.       ],
                    [0.7227343, 1.9242810],
                    [1.0471976, 3.3939748],
                    [1.3181161, 4.7085090]]
    elif phases == 2:
      ref_angles = [[[0.       , 0.       ],
                     [0.7227343, 3.2197042],
                     [1.0471976, 5.4118570],
                     [1.3181161, 1.0290358]],
                    [[0.5053605, 1.8590320],
                     [0.8956648, 4.3726270],
                     [1.1863996, 0.0995198],
                     [1.4454685, 1.9361506]]]
    self.assertAllClose(angles, ref_angles)

  def test_number_of_views_warning(self):
    """Test warning about number of views for golden angle trajectories."""
    string = "optimal k-space filling is achieved when the number of views"

    with self.assertWarnsRegex(UserWarning, string):
      traj_ops.radial_trajectory(128, views=6, ordering='golden')
    with self.assertWarnsRegex(UserWarning, string):
      traj_ops.radial_trajectory(128, views=9, ordering='tiny')


class SpiralTrajectoryTest(test_util.TestCase):
  """Spiral trajectory tests."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/traj_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_waveform(self):
    """Test spiral waveform."""
    waveform = traj_ops.spiral_waveform(base_resolution=128,
                                        spiral_arms=64,
                                        field_of_view=300.0,
                                        max_grad_ampl=20.0,
                                        min_rise_time=10.0,
                                        dwell_time=2.6)
    self.assertAllClose(waveform, self.data['spiral/waveform'])

  @parameterized.product(vd_type=['linear', 'quadratic', 'hanning'])
  @test_util.run_in_graph_and_eager_modes
  def test_waveform_vd(self, vd_type): # pylint: disable=missing-param-doc
    """Test variable-density spiral waveform."""
    waveform = traj_ops.spiral_waveform(base_resolution=256,
                                        spiral_arms=32,
                                        field_of_view=400.0,
                                        max_grad_ampl=28.0,
                                        min_rise_time=6.0,
                                        dwell_time=1.4,
                                        vd_inner_cutoff=0.5,
                                        vd_outer_cutoff=0.8,
                                        vd_outer_density=0.5,
                                        vd_type=vd_type)
    self.assertAllClose(waveform, self.data['spiral/waveform_vd/' + vd_type])

  # @test_util.run_in_graph_and_eager_modes
  def test_trajectory(self):
    """Test spiral trajectory."""
    trajectory = traj_ops.spiral_trajectory(base_resolution=128,
                                            spiral_arms=64,
                                            field_of_view=300.0,
                                            max_grad_ampl=20.0,
                                            min_rise_time=10.0,
                                            dwell_time=2.6,
                                            views=8,
                                            phases=4,
                                            ordering='golden')
    self.assertAllClose(trajectory,
                        _rotate_pi(self.data['spiral/trajectory/golden']),
                        rtol=1e-4, atol=1e-4)


class TrajOpsTest(test_util.TestCase): # pylint: disable=missing-class-docstring

  # @test_util.run_in_graph_and_eager_modes
  def test_kspace_trajectory_shapes(self):
    """Test k-space trajectory."""

    radial_waveform_params = {'base_resolution': 2}

    spiral_waveform_params = {'base_resolution': 128,
                              'spiral_arms': 64,
                              'field_of_view': 300.0,
                              'max_grad_ampl': 20.0,
                              'min_rise_time': 10.0,
                              'dwell_time': 2.6}

    # Create a few parameters to test.
    params = {'traj_type': ('radial', 'spiral'),
              'views': (3, 5),
              'phases': (None, 2),
              'ordering': ('linear', 'golden', 'tiny', 'sorted')}

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        traj_type = p.pop('traj_type')
        views = p['views']
        phases = p['phases']
        ordering = p['ordering']

        if traj_type == 'radial':
          traj = traj_ops.radial_trajectory(**radial_waveform_params, **p)
          dens = traj_ops.radial_density(**radial_waveform_params, **p)
        elif traj_type == 'spiral':
          traj = traj_ops.spiral_trajectory(**spiral_waveform_params, **p)

        # Check shapes.
        if traj_type == 'spiral':
          samples = 494
        elif traj_type == 'radial':
          samples = radial_waveform_params['base_resolution'] * 2
        expected_shape = (samples, 2)
        expected_shape = (views,) + expected_shape
        if phases:
          expected_shape = (phases,) + expected_shape
        self.assertAllEqual(tf.shape(traj), expected_shape)

        if traj_type == 'radial':
          self.assertAllEqual(tf.shape(dens), tf.shape(traj)[:-1])

          if views == 3: # We'll check the exact results for this subset.

            expected_waveform = np.array([[-3.1415927, 0.0],
                                          [-1.5707964, 0.0],
                                          [0.0, 0.0],
                                          [ 1.5707964, 0.0]])

            expected_weights = None

            if phases is None and ordering == 'linear':
              expected_theta = np.array([0.0, 2.0943952, 4.1887903])
              expected_weights = 1.0 / 6.0 * np.array([[4.0, 2.0, 0.5, 2.0],
                                                 [4.0, 2.0, 0.5, 2.0],
                                                 [4.0, 2.0, 0.5, 2.0]])
            elif phases is None and ordering == 'golden':
              # phi = 2.0 / (1.0 + tf.sqrt(5.0))
              # expected_theta = (phi * tf.range(3.0) % 1.0) * 2.0 * math.pi
              expected_theta = np.array([0.0, 3.8832223, 1.4832591])
              expected_weights = 1.0 / 6.0 * np.array([
                [4.5835924, 2.2917962, 0.5, 2.2917962],
                [2.832816, 1.416408, 0.5, 1.416408],
                [4.583592, 2.291796, 0.5, 2.291796]])
            elif phases is None and ordering == 'tiny':
              # expected_theta = (
              #   1 / (phi + 7) * tf.range(3.0) % 1.0) * 2.0 * math.pi
              expected_theta = np.array([0.0, 0.8247778, 1.6495556])
              expected_weights = 1.0 / 6.0 * np.array([
                [4.424791, 2.2123954, 0.5, 2.2123954],
                [3.1504188, 1.5752094, 0.5, 1.5752094],
                [4.4247904, 2.2123952, 0.5, 2.2123952]])
            elif phases is None and ordering == 'sorted':
              expected_theta = np.array([0.0, 1.4832591, 3.8832223])
            elif phases == 2 and ordering == 'linear':
              expected_theta = np.array([[0.0, 2.0943952, 4.1887903],
                                         [1.0471975, 3.1415926, 5.2359877]])
            elif phases == 2 and ordering == 'golden':
              expected_theta = np.array([[0.0, 3.8832223, 1.4832591],
                                         [5.3664813, 2.9665182, 0.56655425]])
            elif phases == 2 and ordering == 'tiny':
              expected_theta = np.array([[0.0, 0.8247778, 1.6495556],
                                         [2.4743333, 3.2991111, 4.123889]])
            elif phases == 2 and ordering == 'sorted':
              expected_theta = np.array([[0.0, 1.4832591, 3.8832223],
                                         [0.56655425, 2.9665182, 5.3664813]])
              expected_weights = 1.0 / 6.0 * np.array([
                [[4.583592, 2.291796, 0.5, 2.291796],
                 [4.583592, 2.291796, 0.5, 2.291796],
                 [2.832816, 1.416408, 0.5, 1.416408]],
                [[4.583592, 2.291796, 0.5, 2.291796],
                 [2.832815, 1.416408, 0.5, 1.416408],
                 [4.583593, 2.291797, 0.5, 2.291797]]])
            else:
              raise ValueError("Unexpected parameter combination")

            expected_traj = np.zeros(
                expected_theta.shape + expected_waveform.shape)

            for idx in np.ndindex(expected_theta.shape):
              t = expected_theta[idx]
              rot_mat = np.array([[np.cos(t), -np.sin(t)],
                                  [np.sin(t), np.cos(t)]])

              expected_traj[idx] = expected_waveform @ np.transpose(rot_mat)

            self.assertAllClose(traj, expected_traj)
            if expected_weights is not None:
              self.assertAllClose(dens, 1 / expected_weights)

        elif traj_type == 'spiral':

          # Just a sanity check.
          self.assertAllInRange(traj, -math.pi, math.pi)


class DensityEstimationTest(test_util.TestCase):
  """Test density estimation."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/traj_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_estimate_density_jackson(self):
    """Test density estimation (Jackson's method)."""
    traj = self.data['estimate_density/trajectory']

    flat_traj = tf.reshape(traj, [-1, traj.shape[-1]])
    dens_flat = traj_ops.estimate_density(flat_traj, [128, 128])
    dens = tf.reshape(dens_flat, traj.shape[:-1])

    self.assertAllClose(dens, self.data['estimate_density/density'] * np.pi,
                        rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_estimate_density_pipe(self):
    """Test density estimation (Pipe's method)."""
    traj = self.data['estimate_density/trajectory']

    flat_traj = tf.reshape(traj, [-1, traj.shape[-1]])
    dens_flat = traj_ops.estimate_density(flat_traj,
                                          [128, 128],
                                          method='pipe',
                                          max_iter=20)
    dens = tf.reshape(dens_flat, traj.shape[:-1])

    self.assertAllClose(dens, self.data['estimate_density/pipe'] * np.pi,
                        rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_estimate_density_3d_many_points(self):
    """Test 3D density estimation with a large points array."""
    # Here we are just checking this runs without causing a seg fault.
    rng = tf.random.Generator.from_seed(0)
    flat_traj = rng.uniform([2560000, 3], minval=-np.pi, maxval=np.pi)
    traj_ops.estimate_density(flat_traj, [128, 128, 128])


def _calculate_rotation_angles_from_radial_trajectory(trajectory):
  """Calculates the angles between views for a given radial trajectory.

  Args:
    trajectory: A tensor of shape `[phases, views, samples, 2]` containing a
      radial trajectory.

  Returns:
    A tensor of shape `[phases * views - 1]` containing the angles between
    views.
  """
  # Flatten views and phases.
  trajectory = tf.reshape(trajectory, [-1] + trajectory.shape[-2:].as_list())
  # Calculate view angles using first sample of readout.
  angles = tf.atan2(trajectory[:, 0, 1], trajectory[:, 0, 0])
  # Calculate rotation angle between consecutive views.
  spacing = tf.experimental.numpy.diff(angles)
  # Normalize to range [0, 2 * pi].
  spacing = tf.where(spacing < 0.0, spacing + 2.0 * np.pi, spacing)
  return spacing


def _rotate_pi(traj):
  return -traj


if __name__ == '__main__':
  tf.test.main()
