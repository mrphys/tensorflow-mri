# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Diagonal linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


LinearOperatorDiag = api_util.export(
    "linalg.LinearOperatorDiag")(
        doc_util.tf_linkcode(
            linear_operator_util.patch_operator(
                linear_operator.make_composite_tensor(
                    tf.linalg.LinearOperatorDiag))))


# Monkey-patch.
tf.linalg.LinearOperatorDiag = LinearOperatorDiag
