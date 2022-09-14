<div align="center">
  <img src="https://raw.githubusercontent.com/mrphys/tensorflow-mri/v0.21.0/tools/assets/tfmri_logo.svg?sanitize=true">
</div>

[![PyPI](https://badge.fury.io/py/tensorflow-mri.svg)](https://badge.fury.io/py/tensorflow-mri)
[![Build](https://github.com/mrphys/tensorflow-mri/actions/workflows/build-package.yml/badge.svg)](https://github.com/mrphys/tensorflow-mri/actions/workflows/build-package.yml)
[![Docs](https://img.shields.io/badge/api-reference-blue.svg)](https://mrphys.github.io/tensorflow-mri/)
[![DOI](https://zenodo.org/badge/388094708.svg)](https://zenodo.org/badge/latestdoi/388094708)

<!-- start-intro -->

TensorFlow MRI is a library of TensorFlow operators for computational MRI.
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

TensorFlow MRI contains operators for:

- Multicoil arrays
  ([`tfmri.coils`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/coils)):
  coil combination, coil compression and estimation of coil sensitivity
  maps.
- Convex optimization
  ([`tfmri.convex`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/convex)):
  convex functions (quadratic, L1, L2, Tikhonov, total variation, etc.) and
  optimizers (ADMM).
- Keras initializers
  ([`tfmri.initializers`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/initializers)):
  neural network initializers, including support for complex-valued weights.
- I/O (`tfmri.io`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/io)):
  additional I/O functions potentially useful when working with MRI data.
- Keras layers
  ([`tfmri.layers`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/layers)):
  layers and building blocks for neural networks, including support for
  complex-valued weights, inputs and outputs.
- Linear algebra
  ([`tfmri.linalg`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/linalg)):
  linear operators specialized for image processing and MRI.
- Loss functions
  ([`tfmri.losses`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/losses)):
  for classification, segmentation and image restoration.
- Metrics
  ([`tfmri.metrics`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/metrics)):
  for classification, segmentation and image restoration.
- Image processing
  ([`tfmri.image`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/image)):
  filtering, gradients, phantoms, image quality assessment, etc.
- Image reconstruction
  ([`tfmri.recon`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/recon)):
  Cartesian/non-Cartesian, 2D/3D, parallel imaging, compressed sensing.
- *k*-space sampling
  ([`tfmri.sampling`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/sampling)):
  Cartesian masks, non-Cartesian trajectories, sampling density compensation,
  etc.
- Signal processing
  ([`tfmri.signal`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/signal)):
  N-dimensional fast Fourier transform (FFT), non-uniform FFT (NUFFT)
  ([see also `TensorFlow NUFFT`](https://github.com/mrphys/tensorflow-nufft)),
  discrete wavelet transform (DWT), *k*-space filtering, etc.
- Unconstrained optimization
  ([`tfmri.optimize`](https://mrphys.github.io/tensorflow-mri/api_docs/tfmri/optimize)):
  gradient descent, L-BFGS.
- And more, e.g., supporting array manipulation and math tasks.

<!-- end-intro -->

## Installation

<!-- start-install -->

You can install TensorFlow MRI with ``pip``:

```
pip install tensorflow-mri
```

Note that only Linux is currently supported.

### TensorFlow Compatibility

Each TensorFlow MRI release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow MRI according to the table below.

<!-- start-compatibility-table -->

| TensorFlow MRI Version | TensorFlow Compatibility | Release Date |
| ---------------------- | ------------------------ | ------------ |
| v0.22.0                | v2.9.x                   | Jul 24, 2022 |
| v0.21.0                | v2.9.x                   | Jul 24, 2022 |
| v0.20.0                | v2.9.x                   | Jun 18, 2022 |
| v0.19.0                | v2.9.x                   | Jun 1, 2022  |
| v0.18.0                | v2.8.x                   | May 6, 2022  |

<!-- end-compatibility-table -->

<!-- end-install -->

## Documentation

Visit the [docs](https://mrphys.github.io/tensorflow-mri/) for guides,
tutorials and the API reference.

## Issues

If you use this package and something does not work as you expected, please
[file an issue](https://github.com/mrphys/tensorflow-mri/issues/new)
describing your problem. We're here to help!

## Credits

If you like this software, star the repository! [![Stars](https://img.shields.io/github/stars/mrphys/tensorflow-mri?style=social)](https://github.com/mrphys/tensorflow-mri/stargazers)

If you find this software useful in your research, you can cite TensorFlow MRI
using its [Zenodo record](https://doi.org/10.5281/zenodo.5151590).

In the above link, scroll down to the "Export" section and select your favorite
export format to get an up-to-date citation.

## Contributions

Contributions of any kind are welcome! Open an issue or pull request to begin.
