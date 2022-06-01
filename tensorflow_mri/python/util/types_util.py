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
"""Types utilities."""

import tensorflow as tf

SIGNED_INTEGER_TYPES = [tf.int8, tf.int16, tf.int32, tf.int64]
UNSIGNED_INTEGER_TYPES = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
INTEGER_TYPES = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES

FLOATING_TYPES = [tf.float16, tf.float32, tf.float64]
COMPLEX_TYPES = [tf.complex64, tf.complex128]
