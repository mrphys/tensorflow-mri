"""TensorFlow MRI.

TensorFlow MRI is a package implementing functionality relevant to ML in
Magnetic Resonance Imaging (MRI).
"""

from setuptools import find_packages
from setuptools import setup


PROJECT_NAME = 'tensorflow-mri'

with open('VERSION') as version_file:
    VERSION = version_file.read().strip()

with open("requirements.txt") as f:
    REQUIRED_PACKAGES = [line.strip() for line in f.readlines()]

DOCLINES = __doc__.split('\n')

setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    author='Javier Montalt-Tordera',
    author_email='javier.tordera.17@ucl.ac.uk',
    url='https://github.com/mrphys/tensorflow_nufft',
    packages=find_packages(),
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
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license="Apache 2.0",
    keywords=['tensorflow', 'mri'],
    include_package_data=True,
    zip_safe=True,
    install_requires=REQUIRED_PACKAGES
)
