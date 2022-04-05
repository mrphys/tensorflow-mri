.. image:: https://raw.githubusercontent.com/mrphys/tensorflow-mri/v0.6.0/tools/assets/tfmr_logo.svg?sanitize=true
  :align: center
  :scale: 100 %
  :alt: TFMRI logo

|

|pypi| |build| |docs| |doi|

.. |pypi| image:: https://badge.fury.io/py/tensorflow-mri.svg
    :target: https://badge.fury.io/py/tensorflow-mri
.. |build| image:: https://github.com/mrphys/tensorflow-mri/actions/workflows/build-package.yml/badge.svg
    :target: https://github.com/mrphys/tensorflow-mri/actions/workflows/build-package.yml
.. |docs| image:: https://img.shields.io/badge/api-reference-blue.svg
    :target: https://mrphys.github.io/tensorflow-mri/
.. |doi| image:: https://zenodo.org/badge/388094708.svg
    :target: https://zenodo.org/badge/latestdoi/388094708

.. start-intro

TensorFlow MRI is a library of TensorFlow operators for computational MRI which
includes: 

* A fast, native non-uniform fast Fourier transform (NUFFT) operator (see
  also `TensorFlow NUFFT <https://github.com/mrphys/tensorflow-nufft>`_).
* A unified gateway for MR image reconstruction, which supports parallel
  imaging, compressed sensing, machine learning and partial Fourier methods. 
* Common linear and nonlinear operators, such as Fourier operators and
  regularizers, to aid in the development of novel image reconstruction
  techniques. 
* Multicoil imaging operators, such as coil combination, coil compression and
  estimation of coil sensitivity maps. 
* Calculation of non-Cartesian k-space trajectories and sampling density
  estimation. 
* A collection of Keras objects including models, layers, metrics, loss
  functions and callbacks for rapid development of neural networks. 
* Many other differentiable operators for common tasks such as array
  manipulation and image/signal processing. 

The library has a Python interface and is mostly written in Python. However,
computations are efficiently performed by the TensorFlow backend (implemented in
C++/CUDA), which brings together the ease of use and fast prototyping of Python
with the speed and efficiency of optimized lower-level implementations. 

Being an extension of TensorFlow, TensorFlow MRI integrates seamlessly in ML
applications. No additional interfacing is needed to include a SENSE operator
within a neural network, or to use a trained prior as part of an iterative
reconstruction. Therefore, the gap between ML and non-ML components of image
processing pipelines is eliminated. 

Whether an application involves ML or not, TensorFlow MRI operators can take
full advantage of the TensorFlow framework, with capabilities including
automatic differentiation, multi-device support (CPUs and GPUs), automatic
device placement and copying of tensor data, and conversion to fast,
serializable graphs. 

.. end-intro

Installation
------------

.. start-install

You can install TensorFlow MRI with ``pip``:

.. code-block:: console

    $ pip install tensorflow-mri

Note that only Linux is currently supported.

TensorFlow Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^

Each TensorFlow MRI release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow MRI according to the table below.

======================  ========================  ============
TensorFlow MRI Version  TensorFlow Compatibility  Release Date
======================  ========================  ============
v0.16.0                 v2.8.x                    Apr 6, 2022
v0.15.0                 v2.8.x                    Apr 1, 2022
v0.14.0                 v2.8.x                    Mar 29, 2022
v0.13.0                 v2.8.x                    Mar 15, 2022
v0.12.0                 v2.8.x                    Mar 14, 2022
v0.11.0                 v2.8.x                    Mar 10, 2022
v0.10.0                 v2.8.x                    Mar 3, 2022
v0.9.0                  v2.7.x                    Dec 3, 2021
v0.8.0                  v2.7.x                    Nov 11, 2021
v0.7.0                  v2.6.x                    Nov 3, 2021
v0.6.2                  v2.6.x                    Oct 13, 2021
v0.6.1                  v2.6.x                    Sep 30, 2021
v0.6.0                  v2.6.x                    Sep 28, 2021
v0.5.0                  v2.6.x                    Aug 29, 2021
v0.4.0                  v2.6.x                    Aug 18, 2021
======================  ========================  ============

.. end-install

Documentation
-------------

Visit the `docs <https://mrphys.github.io/tensorflow-mri/>`_ for the API
reference and examples of usage. 

Contributions
-------------

If you use this package and something does not work as you expected, please
`file an issue <https://github.com/mrphys/tensorflow-mri/issues/new>`_
describing your problem. We will do our best to help.

Contributions are very welcome. Please create a pull request if you would like
to make a contribution.

Citation
--------

If you find this software useful in your work, please
`cite us <https://doi.org/10.5281/zenodo.5151590>`_.

FAQ
---

.. start-faq

**When trying to install TensorFlow MRI, I get an error about OpenEXR which
includes:
``OpenEXR.cpp:36:10: fatal error: ImathBox.h: No such file or directory``. What
do I do?**

OpenEXR is needed by TensorFlow Graphics, which is a dependency of TensorFlow
MRI. This issue can be fixed by installing the OpenEXR library. On
Debian/Ubuntu:

.. code-block:: console

    $ apt install libopenexr-dev

.. end-faq
