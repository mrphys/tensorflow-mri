# Release 0.22.0



## Breaking Changes

- `tfmri.models`

  - `ConvBlock1D`, `ConvBlock2D` and `ConvBlock3D`contain backwards
    incompatible changes.
  - `UNet1D`, `UNet2D` and `UNet3D` contain backwards incompatible changes.


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

  - Added new function `estimate_sensitivities_universal`.

- `tfmri.geometry`:

  - Added new extension type `Rotation2D`.

- `tfmri.layers`:

  - Added new wrapper layer `Normalized`.

- `tfmri.models`:

  - Added new models `ConvBlockLSTM1D`, `ConvBlockLSTM2D` and `ConvBlockLSTM3D`.
  - Added new models `UNetLSTM1D`, `UNetLSTM2D` and `UNetLSTM3D`.

- `tfmri.sampling`:

  - Added operator `spiral_waveform` to public API.
  - Added new functions `accel_mask` and `center_mask`.


## Bug Fixes and Other Changes

- `tfmri`:

  - Removed the TensorFlow Graphics dependency, which should also eliminate
    the common OpenEXR error.

- `tfmri.recon`:

  - Improved error reporting for ``least_squares``.
