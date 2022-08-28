# Release 0.22.0



## Breaking Changes



## Major Features and Improvements

- `tfmri.geometry`:

  - Added new extension types `Rotation2D` and `Rotation3D`.

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
