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
"""Tests for coil ops."""

import tensorflow as tf

from tensorflow_mri.python.ops import coil_ops
from tensorflow_mri.python.utils import io_utils


class SensMapsTest(tf.test.TestCase):
  """Tests for ops related to estimation of coil sensitivity maps."""

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/coil_ops_data.h5')


  def test_walsh(self):
    """Test Walsh's method."""

    maps = coil_ops.estimate_coil_sens_maps(
      self.data['images'], method='walsh')

    self.assertAllClose(maps, self.data['maps/walsh'])


  def test_walsh_transposed(self):
    """Test Walsh's method with a transposed array."""

    maps = coil_ops.estimate_coil_sens_maps(
      tf.transpose(self.data['images'], [2, 0, 1]), coil_axis=0, method='walsh')

    self.assertAllClose(maps, tf.transpose(self.data['maps/walsh'], [2, 0, 1]))


  def test_inati(self):
    """Test Inati's method."""

    maps = coil_ops.estimate_coil_sens_maps(
      self.data['images'], method='inati')

    self.assertAllClose(maps, self.data['maps/inati'])


  def test_espirit(self):
    """Test ESPIRiT method."""

    maps = coil_ops.estimate_coil_sens_maps(
      self.data['kspace'], method='espirit')

    self.assertAllClose(maps, self.data['maps/espirit'])
