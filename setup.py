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
"""TensorFlow MRI.

TensorFlow MRI is a package implementing functionality relevant to ML in
Magnetic Resonance Imaging (MRI).
"""

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install as _install

PROJECT_NAME = 'tensorflow-mri'

with open('tensorflow_mri/VERSION') as version_file:
  VERSION = version_file.read().strip()

with open("requirements.txt") as f:
  REQUIRED_PACKAGES = [line.strip() for line in f.readlines()]

DOCLINES = __doc__.split('\n')

class install(_install):

  def finalize_options(self):
    _install.finalize_options(self)
    self.install_lib = self.install_platlib

class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False

setup(
  name=PROJECT_NAME,
  version=VERSION,
  description=DOCLINES[0],
  long_description='\n'.join(DOCLINES[2:]),
  long_description_content_type="text/markdown",
  author='Javier Montalt-Tordera',
  author_email='javier.montalt@outlook.com',
  url='https://github.com/mrphys/tensorflow_nufft',
  packages=find_packages(),
  install_requires=REQUIRED_PACKAGES,
  include_package_data=True,
  zip_safe=False,
  distclass=BinaryDistribution,
  cmdclass={'install': install},
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Environment :: GPU',
    'Environment :: GPU :: NVIDIA CUDA',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: C++',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  
  license="Apache 2.0",
  keywords=[
    'tensorflow',
    'mri',
    'magnetic resonance imaging',
    'machine learning',
    'ml'
  ]
)
