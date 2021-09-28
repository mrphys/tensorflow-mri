.. image:: https://raw.githubusercontent.com/mrphys/tensorflow-mri/develop/tools/assets/tfmr_logo.svg?sanitize=true
  :align: center
  :scale: 70 %
  :alt: TFMR logo

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

TensorFlow MRI is a collection of TensorFlow add-ons for computational MRI.

It contains functionality including:

* Estimation of coil sensitivity maps: Walsh's method, Inati's fast method and
  ESPIRiT.
* Coil compression using singular value decomposition (SVD).
* Image reconstruction operations: basic (FFT, NUFFT), parallel imaging
  (SENSE, GRAPPA, CG-SENSE) and partial Fourier (zero-filling, homodyne
  detection, projection onto convex sets). 
* Calculation of radial and spiral trajectories and sampling densities.
* Keras metrics for image quality assessment, classification and segmentation.
* Helper operations for array manipulation, image processing and linear algebra.

All operations are performed using a TensorFlow/Keras backend. This has several
key advantages:

* Seamless integration in machine learning applications.
* Runs on heterogeneous systems, with most operations supporting CPU and
  GPU-accelerated paths.
* Code is easy to understand, with most of this package written in Python.

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
TensorFlow. Please see the compatibility table below to see what versions of
each package you can expect to work together.

==============  ==========
TensorFlow MRI  TensorFlow
==============  ==========
v0.4            v2.6
v0.5            v2.6
v0.6            v2.6
==============  ==========

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
`OpenEXR.cpp:36:10: fatal error: ImathBox.h: No such file or directory`. What do
I do?**

OpenEXR is needed by TensorFlow Graphics, which is a dependency of TensorFlow
MRI. This issue can be fixed by installing the OpenEXR library. On
Debian/Ubuntu:

.. code-block:: console

    $ apt install libopenexr-dev

.. end-faq
