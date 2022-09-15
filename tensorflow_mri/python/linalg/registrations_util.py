# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common utilities for registering LinearOperator methods.

Adapted from:
  tensorflow/python/ops/linalg/registrations_util.py
"""

from tensorflow.python.ops.linalg import registrations_util

combined_commuting_positive_definite_hint = (
    registrations_util.combined_commuting_positive_definite_hint)
combined_commuting_self_adjoint_hint = (
    registrations_util.combined_commuting_self_adjoint_hint)
combined_non_singular_hint = registrations_util.combined_non_singular_hint
