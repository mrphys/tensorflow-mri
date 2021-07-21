# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for k-space trajectory ops."""

import itertools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import traj_ops


class TrajOpsTest(tf.test.TestCase): # pylint: disable=missing-class-docstring

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
    params = {'traj_type': ('radial',),
              'views': (3, 5),
              'phases': (None, 2),
              'spacing': ('linear', 'golden', 'tiny', 'sorted')}

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        traj_type = p.pop('traj_type')
        views = p['views']
        phases = p['phases']
        spacing = p['spacing']

        if traj_type == 'radial':
          traj = traj_ops.radial_trajectory(**radial_waveform_params, **p)
          dens = traj_ops.radial_density(**radial_waveform_params, **p)
        elif traj_type == 'spiral':
          traj = traj_ops.spiral_trajectory(
            **spiral_waveform_params, **p)

        # Check shapes.
        if traj_type == 'spiral':
          samples = 494
        elif traj_type == 'radial':
          samples = radial_waveform_params['base_resolution'] * 2
        expected_shape = (samples, 2)
        expected_shape = (views,) + expected_shape
        if phases:
          expected_shape = (phases,) + expected_shape
        self.assertEqual(traj.shape, expected_shape)

        if traj_type == 'radial':
          self.assertEqual(dens.shape, traj.shape[:-1])

          if views == 3: # We'll check the exact results for this subset.

            #  and phases is None and spacing == 'linear':
            expected_waveform = np.array([[3.1415927, 0.0],
                                          [1.5707964, 0.0],
                                          [0.0, 0.0],
                                          [-1.5707964, 0.0]])

            expected_dens = None

            if phases is None and spacing == 'linear':
              expected_theta = np.array([0.0, 2.0943952, 4.1887903])
              expected_dens = np.array([[4.0, 2.0, 0.25, 2.0],
                                        [4.0, 2.0, 0.25, 2.0],
                                        [4.0, 2.0, 0.25, 2.0]])
            elif phases is None and spacing == 'golden':
              # phi = 2.0 / (1.0 + tf.sqrt(5.0))
              # expected_theta = (phi * tf.range(3.0) % 1.0) * 2.0 * math.pi
              expected_theta = np.array([0.0, 3.8832223, 1.4832591])
              expected_dens = np.array([
                [4.5835924, 2.2917962, 0.25, 2.2917962],
                [2.832816, 1.416408, 0.25, 1.416408],
                [4.583592, 2.291796, 0.25, 2.291796]])
            elif phases is None and spacing == 'tiny':
              # expected_theta = (
              #   1 / (phi + 7) * tf.range(3.0) % 1.0) * 2.0 * math.pi
              expected_theta = np.array([0.0, 0.8247778, 1.6495556])
              expected_dens = np.array([
                [4.424791, 2.2123954, 0.25, 2.2123954],
                [3.1504188, 1.5752094, 0.25, 1.5752094],
                [4.4247904, 2.2123952, 0.25, 2.2123952]])
            elif phases is None and spacing == 'sorted':
              expected_theta = np.array([0.0, 1.4832591, 3.8832223])
            elif phases == 2 and spacing == 'linear':
              expected_theta = np.array([[0.0, 2.0943952, 4.1887903],
                                         [0.0, 2.0943952, 4.1887903]])
            elif phases == 2 and spacing == 'golden':
              expected_theta = np.array([[0.0, 3.8832223, 1.4832591],
                                         [5.3664813, 2.9665182, 0.56655425]])
            elif phases == 2 and spacing == 'tiny':
              expected_theta = np.array([[0.0, 0.8247778, 1.6495556],
                                         [2.4743333, 3.2991111, 4.123889]])
            elif phases == 2 and spacing == 'sorted':
              expected_theta = np.array([[0.0, 1.4832591, 3.8832223],
                                         [0.56655425, 2.9665182, 5.3664813]])
              expected_dens = np.array([
                [[4.583592, 2.291796, 0.25, 2.291796],
                 [4.583592, 2.291796, 0.25, 2.291796],
                 [2.832816, 1.416408, 0.25, 1.416408]],
                [[4.583592, 2.291796, 0.25, 2.291796],
                 [2.832815, 1.416408, 0.25, 1.416408],
                 [4.583593, 2.291797, 0.25, 2.291797]]])
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
            if expected_dens is not None:
              self.assertAllClose(dens, expected_dens)


if __name__ == '__main__':
  tf.test.main()
