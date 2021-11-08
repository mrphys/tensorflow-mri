.. image:: https://raw.githubusercontent.com/mrphys/tensorflow-mri/v0.6.0/tools/assets/tfmr_logo.svg?sanitize=true
  :align: center
  :scale: 100 %
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

TensorFlow MRI (TFMR) is a Python library for MR image reconstruction and
processing. TFMR provides:

* A selection of differentiable operators for accelerated image reconstruction,
  Cartesian and non-Cartesian *k*-space sampling, and many other common MR image
  and signal processing tasks.
* Keras callbacks, layers, metrics and losses and other utilities for the
  creation, training and evaluation of machine learning models.

TFMR is aimed for scientists and researchers working with MRI data. Whether you
are planning to use machine learning or not, TFMR enables prototyping and
deployment of efficient computational MRI solutions easily and within Python.

Thanks to the use of a TensorFlow backend, TFMR integrates seamlessly in machine
learning projects. It also inherits other benefits of TensorFlow, including high
performance computation and GPU acceleration. 

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
v0.7            v2.6
v0.8            v2.7
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
``OpenEXR.cpp:36:10: fatal error: ImathBox.h: No such file or directory``. What
do I do?**

OpenEXR is needed by TensorFlow Graphics, which is a dependency of TensorFlow
MRI. This issue can be fixed by installing the OpenEXR library. On
Debian/Ubuntu:

.. code-block:: console

    $ apt install libopenexr-dev

.. end-faq
