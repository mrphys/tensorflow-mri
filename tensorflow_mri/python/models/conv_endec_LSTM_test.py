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
"""Tests for module `conv_endec_LSTM`."""
# pylint: disable=missing-param-doc
import sys
from tensorflow_mri.python.models import conv_endec_LSTM
sys.path.insert(0,'/workspaces/tensorflow-mri/tensorflow_mri')
config = dict(
        filters=[16, 32, 64],
        kernel_size=2,
        pool_size=2,
        block_depth=2,
        use_deconv=True,
        activation='tanh',
        use_bias=False,
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        bn_momentum=0.98,
        bn_epsilon=0.002,
        out_channels=1,
        out_kernel_size=1,
        out_activation='relu',
        use_global_residual=True,
        use_dropout=True,
        dropout_rate=0.5,
        dropout_type='spatial',
        use_tight_frame=True)

block = conv_endec_LSTM.LSTMUNet2D(**config)
self.assertEqual(block.get_config(), config)

block2 = conv_endec_LSTM.LSTMUNet2D.from_config(block.get_config())
self.assertAllEqual(block.get_config(), block2.get_config())
    