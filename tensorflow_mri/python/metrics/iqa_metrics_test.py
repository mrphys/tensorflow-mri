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
"""Tests for module `iqa_metrics`."""

import tensorflow as tf

    # import scipy.io as sio
    # data = sio.loadmat('tests/data/psnr.mat')
    
    # import h5py
    # with h5py.File('tests/data/image_ops_data.h5', 'r+') as f:
    #   f.create_dataset('psnr/2d/batch/img1', data=img1)
    #   f.create_dataset('psnr/2d/batch/img2', data=img2)


class StructuralSimilarityTest(tf.test.TestCase):
  """Tests for SSIM metric."""

  def test_ssim_2d(self):

    pass
    
    

if __name__ == '__main__':
  tf.test.main()
