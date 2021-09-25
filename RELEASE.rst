Release 0.6.0
=============

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

Breaking Changes
----------------

* The keyword arguments `spacing` and `domain` of the ops
  `tfmr.radial_trajectory` and `tfmr.spiral_trajectory` have been renamed to
  `ordering` and `angle_range`, respectively.
* The range of the angles in 2D "half" radial trajectories will now be `[0, pi]`
  instead of `[-pi/2, pi/2]`.
* The range of the angles in 2D radial/spiral trajectories will now be
  `[0, 2 * pi]` instead of `[0, pi]`.
* Multi-phase linear trajectories will now be interleaved.

Known Caveats
-------------

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>

Major Features and Improvements
-------------------------------

* `tfmr`:

  * Added new image ops `total_variation` and `phantom`.
  * Addew new array ops `cartesian_product`, `meshgrid`, `ravel_multi_index` and
    `unravel_index`.
  * Added new geometry module with ops `rotate_2d` and `rotate_3d`.
  * Added new optimizers module with op `lbfgs_minimize`.
  * Added new convex operators module with ops `ConvexOperator`,
    `ConvexOperatorL1Norm`, `Regularizer` and `TotalVariationRegularizer`.
  * Added new signal processing module with ops `crop_kspace`, `filter_kspace`
    and `hamming_filter`.
  * Added new linear algebra ops `LinearOperatorFFT` and
    `LinearOperatorImaging`.
  * Added new math ops `make_val_and_grad_fn`, `view_as_complex` and
    `view_as_real`.
  * Added new ordering methods `"golden_half"`, `"tiny_half"` and
    `"sphere_archimedean"` to function `radial_trajectory`.
  * Added new method `"nufft"` to `reconstruct`.
  * Added new method `"pipe"` to `estimate_density`.
  * Added keyword argument `rank` to function `radial_waveform`.


Bug Fixes and Other Changes
---------------------------

* `tfmr`:

  * Fixed some bugs that would cause some ops to fail in graph mode.
  * Added graph mode tests.
  * Refactored testing modules.
  * Refactored linear algebra module.
  * Refactored utilities modules.
