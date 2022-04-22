Install TensorFlow MRI
======================

Requirements
------------

TensorFlow MRI should work in most Linux systems that meet the
`requirements for TensorFlow <https://www.tensorflow.org/install>`_.

.. warning::

    TensorFlow MRI is not yet available for Windows or macOS.
    `Help us support them! <https://github.com/mrphys/tensorflow-mri/issues/3>`_.


TensorFlow compatibility
~~~~~~~~~~~~~~~~~~~~~~~~

Each TensorFlow MRI release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow MRI according to the
:ref:`TensorFlow compatibility table`.


Set up your system
------------------

You will need a working TensorFlow installation. Follow the `TensorFlow
installation instructions <https://www.tensorflow.org/install>`_ if you do not
have one already.


Use a GPU
~~~~~~~~~

If you need GPU support, we suggest that you use one of the
`TensorFlow Docker images <https://www.tensorflow.org/install/docker>`_.
These come with a GPU-enabled TensorFlow installation and are the easiest way
to run TensorFlow and TensorFlow MRI on your system.

.. code-block:: console

    $ docker pull tensorflow/tensorflow:latest-gpu

Alternatively, make sure you follow
`these instructions <https://www.tensorflow.org/install/gpu>`_ when setting up
your system.


Download from PyPI
------------------

TensorFlow MRI is available on the Python package index (PyPI) and can be
installed using the ``pip`` package manager:

.. code-block:: console

    $ pip install tensorflow-mri
    

Run in Google Colab
-------------------

To get started without installing anything on your system, you can use
`Google Colab <https://colab.research.google.com/notebooks/welcome.ipynb>`_.
Simply create a new notebook and use ``pip`` to install TensorFlow MRI.

.. code:: python

  !pip install tensorflow-mri


The Colab environment is already configured to run TensorFlow and has GPU
support.


TensorFlow compatibility table
------------------------------

.. include:: ../../../README.rst
   :start-after: start-compatibility-table
   :end-before: end-compatibility-table
