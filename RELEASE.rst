Release 0.7.0
=============

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

Breaking Changes
----------------

* `tfmr`:

  * The scaling of the densities returned by `estimate_density`,
    `radial_density` and `estimate_radial_density` has changed. This harmonizes
    the values returned by all three functions and enables correct NUFFT
    normalization. Any application relying on exact density values should be
    checked. Applications that only rely on the relative scaling between points
    should not be affected.
  * `estimate_radial_density` no longer accepts the input `base_resolution`.
    However, it will now accept the optional input `readout_os`.

Known Caveats
-------------

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>

Major Features and Improvements
-------------------------------

* `tfmr`:

  * Added new ops `flatten_trajectory` and `flatten_density`.
  * `central_crop` and `resize_with_crop_or_pad` now accept `shape` arguments
    whose length is less than the rank of `tensor`.
  * Added new argument `norm` to `LinearOperatorNUFFT` to control FFT
    normalization.

* `tfmr.callbacks`:

  * Added module.
  * Added new callback `TensorBoardImages`.

* `tfmr.io`:

  * Added module.
  * Added new function `encode_gif`.

* `tfmr.layers`:

  * Added module.
  * Added new convolutional layers `ConvBlock` and `UNet`.
  * Added new preprocessing layers `AddChannelDimension`, `Cast`,
    `KSpaceResampling`, `RepeatTensor`, `ResizeWithCropOrPad` and
    `ScaleByMinMax`.

* `tfmr.losses`:

  * Added module.
  * Added new losses `StructuralSimilarityLoss`,
    `MultiscaleStructuralSimilarityLoss`, `ssim_loss` and
    `ssim_multiscale_loss`.

* `tfmr.metrics`:

  * Added new metrics `DiceIndex` and `JaccardIndex` (aliases of `F1Score` and
    `IoU`, respectively).

Bug Fixes and Other Changes
---------------------------

* `tfmr`:

  * Fixed a bug in static shape inference for ops `central_crop` and
    `resize_with_crop_or_pad`.
  * Fixed a bug in `view_as_complex` that would result in incorrect results for
    multidimensional arrays.
  * Fixed a bug in `LinearOperatorNUFFT` that would result in incorrect batch
    shape processing when the rank of `domain_shape` was equal to the number of
    spatial dimensions.
