# Install TensorFlow MRI

## Requirements

TensorFlow MRI should work in most Linux systems that meet the
[requirements for TensorFlow](https://www.tensorflow.org/install).

```{warning}
TensorFlow MRI is not yet available for Windows or macOS.
[`Help us support them!](https://github.com/mrphys/tensorflow-mri/issues/3).
```

### TensorFlow compatibility

Each TensorFlow MRI release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow MRI according to the table below.

```{include} ../../../README.md
---
start-after: <!-- start-compatibility-table -->
end-before: <!-- end-compatibility-table -->
---
```

```{warning}
Each TensorFlow MRI version aims to target and support the latest TensorFlow
version only. A new version of TensorFlow MRI will be released shortly after
each TensorFlow release. TensorFlow MRI versions that target older versions
of TensorFlow will not generally receive any updates.
```

## Set up your system

You will need a working TensorFlow installation. Follow the
[TensorFlow installation instructions](https://www.tensorflow.org/install) if
you do not have one already.


### Use a GPU

If you need GPU support, we suggest that you use one of the
[TensorFlow Docker images](https://www.tensorflow.org/install/docker).
These come with a GPU-enabled TensorFlow installation and are the easiest way
to run TensorFlow and TensorFlow MRI on your system.

.. code-block:: console

    $ docker pull tensorflow/tensorflow:latest-gpu

Alternatively, make sure you follow
[these instructions](https://www.tensorflow.org/install/gpu) when setting up
your system.


## Download from PyPI

TensorFlow MRI is available on the Python package index (PyPI) and can be
installed using the ``pip`` package manager:

```
pip install tensorflow-mri
```


## Run in Google Colab

To get started without installing anything on your system, you can use
[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb).
Simply create a new notebook and use ``pip`` to install TensorFlow MRI.

```
!pip install tensorflow-mri
```

The Colab environment is already configured to run TensorFlow and has GPU
support.
