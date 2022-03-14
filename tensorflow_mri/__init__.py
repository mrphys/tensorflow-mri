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
"TensorFlow MRI."

from tensorflow_mri.__about__ import *

# TODO(jmontalt): Remove these imports on release 1.0.0.
from tensorflow_mri.python.ops.array_ops import *
from tensorflow_mri.python.ops.coil_ops import *
from tensorflow_mri.python.ops.convex_ops import *
from tensorflow_mri.python.ops.fft_ops import *
from tensorflow_mri.python.ops.geom_ops import *
from tensorflow_mri.python.ops.image_ops import *
from tensorflow_mri.python.ops.linalg_ops import *
from tensorflow_mri.python.ops.math_ops import *
from tensorflow_mri.python.ops.optimizer_ops import *
from tensorflow_mri.python.ops.recon_ops import *
from tensorflow_mri.python.ops.signal_ops import *
from tensorflow_mri.python.ops.traj_ops import *

from tensorflow_mri import python

# Import public API.
from tensorflow_mri.python.util import api_util

for namespace in api_util.get_namespaces():
  globals()[namespace] = api_util.import_namespace(namespace)

del api_util
