# Release 0.22.0



## Breaking Changes



## Major Features and Improvements

- `tf`:

  - Added custom FFT kernels for CPU. These can be used directly through the
    standard core TF APIs `tf.signal.fft`, `tf.signal.fft2d` and
    `tf.signal.fft3d`.

- `tfmri.activations`:

  - Added new functions `complex_relu` and `mod_relu`.

- `tfmri.callbacks`:

  - The `TensorBoardImages` callback can now create multiple summaries.

- `tfmri.coils`:

  - Added new function `estimate_sensitivities_from_kspace`.

- `tfmri.geometry`:

  - Added new extension type `Rotation2D`.

- `tfmri.layers`:

  - Added new wrapper layer `Normalized`.

- `tfmri.sampling`:

  - Added operator ``spiral_waveform`` to public API.


## Bug Fixes and Other Changes

- `tfmri`:

  - Removed the TensorFlow Graphics dependency, which should also eliminate
    the common OpenEXR error.

- `tfmri.recon`:

  - Improved error reporting for ``least_squares``.
