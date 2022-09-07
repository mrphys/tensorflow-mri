# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Tests for module `twix_io`."""

import tensorflow as tf
import tensorflow_mri as tfmri

from tensorflow_mri.python.util import test_util

TEST_FILE = 'tests/data/gre.dat'

class ParseTwixTest(test_util.TestCase):
  """Tests for `parse_twix` function."""

  def test_parse_twix(self):
    """Tests reading from a file."""
    contents = tf.io.read_file(TEST_FILE)

    # Parse TWIX file.
    twix = tfmri.io.parse_twix(contents)

    # Basic looping.
    kspace = []
    for scan in twix.measurements[0].scans:
      if not scan.header.eval_info_mask.ACQEND:
        kspace.append([])
        for channel in scan.channels:
          kspace[-1].append(channel.data)
    kspace = tf.convert_to_tensor(kspace)
    kspace = tf.transpose(kspace, [1, 0, 2])

    # Reconstruct an image.
    image = tfmri.recon.adj(kspace, [160, 320])

    # Test 40 pixels.
    test_pixels = [
        10274, 100414,  43817,  82566,  47161,  95817,  73454,  37660,
        17116,  37163,  36018,  94110,  61346,  30678,   3381,   2638,
        84365,  65707,   4120,  14389,  53742,  82352,  72599,  31465,
        34866,  92797,  40474,  58875,  80353,  86533,  61116,  39078,
        17864,  33563,  73549,  49704,  89011,  64354,  24272,  30771
    ]

    expected_values = [
        -1.2701083e-06-2.3950960e-07j,  1.5321851e-06-6.0459581e-07j,
        -2.3624499e-07-3.3690151e-07j, -1.8920543e-08+3.6719811e-07j,
         6.7760725e-06+6.7225542e-06j,  2.0235464e-05-2.6655940e-07j,
         1.2165106e-06-6.7168941e-07j,  1.8745079e-05-1.2247641e-05j,
         1.6538288e-06+2.0086329e-06j, -5.8484528e-07-1.4344830e-06j,
         9.6985887e-06-1.6889178e-05j,  7.2497940e-08+2.2277927e-06j,
        -7.6203497e-07+2.8434135e-06j, -1.8966676e-06+7.0185899e-07j,
         3.1993693e-06-4.0846868e-07j,  2.3145808e-06-4.0182917e-07j,
         1.4444863e-05-1.6325985e-05j,  1.2718554e-05+2.8837078e-06j,
        -2.5439724e-07-1.5061369e-07j,  9.1443229e-07+2.6340251e-07j,
        -1.6596479e-07-3.6424115e-07j,  2.0008658e-05+5.8169287e-09j,
        -4.9824121e-07+1.8819964e-07j,  1.6354710e-05+5.4614238e-06j,
         7.8803828e-07-5.8470232e-07j,  3.4816506e-07+1.2416560e-08j,
         1.3673757e-05-1.0114289e-05j,  2.3925327e-07+5.3502725e-07j,
        -4.6500159e-07-4.9572600e-07j, -2.6744103e-06+9.4111897e-07j,
        -8.6854664e-07-4.1762880e-07j, -4.3979887e-07-1.8401410e-07j,
        -4.7728474e-07+1.3821273e-06j, -1.4986534e-07+9.7267298e-07j,
        -1.1740372e-06-1.2283937e-06j, -1.2053533e-06+2.8251404e-06j,
         7.2834683e-07+1.4818811e-06j, -2.4099645e-06-5.7378662e-07j,
        -5.6130324e-07-1.1934012e-06j,  2.0544548e-09+1.2545783e-06j]

    flat_image = tf.reshape(image, [-1])
    self.assertAllEqual([2, 160, 320], tf.shape(image))
    self.assertAllClose(expected_values, tf.gather(flat_image, test_pixels))  # pylint: disable=no-value-for-parameter

    # Check protocol.
    meas_prot = twix.measurements[0].protocol['Meas']
    self.assertAllEqual(160, meas_prot.MEAS.sKSpace.lBaseResolution.value)
    self.assertAllEqual(
        200.0, meas_prot.MEAS.sSliceArray.asSlice[0].dReadoutFOV.value)


if __name__ == '__main__':
  tf.test.main()
