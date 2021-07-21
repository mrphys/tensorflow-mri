# tensorflow_nufft

This is a TensorFlow op implementing the non-uniform fast Fourier transform
(NUFFT):

 - Native C++/CUDA kernels for CPU/GPU.
 - Python/TensorFlow interface.
 - Automatic differentiation.
 - Automatic shape inference.

The core NUFFT implementation is that of the Flatiron Institute. Please see the
original [FINUFFT]() and [cuFINUFFT]() repositories for details. The main
contribution of this package is the TensorFlow functionality.

## Installation



### Prerequisites

`tensorflow_nufft` uses the FFTW3 library. To install FFTW3 in Debian/Ubuntu
run:

```
apt-get install libfftw3-dev
```


### Install

The easiest way to install `tensorflow_nufft` is via pip.

```
pip install tensorflow_nufft
```


### Contributions
All contributions are welcome.


### Example



### Citations



