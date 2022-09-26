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
"""(Scaled) identity linear operators."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


LinearOperatorIdentity = api_util.export(
    "linalg.LinearOperatorIdentity")(
        doc_util.no_linkcode(
            linear_operator.make_linear_operator(
                tf.linalg.LinearOperatorIdentity)))


LinearOperatorScaledIdentity = api_util.export(
    "linalg.LinearOperatorScaledIdentity")(
        doc_util.no_linkcode(
            linear_operator.make_linear_operator(
                tf.linalg.LinearOperatorScaledIdentity)))


tf.linalg.LinearOperatorIdentity = LinearOperatorIdentity
tf.linalg.LinearOperatorScaledIdentity = LinearOperatorScaledIdentity
