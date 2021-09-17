Release 0.6.0
=============

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

Breaking Changes
----------------

* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API, ABI AND BEHAVIORAL BREAKING CHANGES>

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
  * Added new optimizer op `lbfgs_minimize`.
  * Added new linear algebra ops `LinearOperatorFFT` and
    `LinearOperatorImaging`.
  * Added new math ops `make_val_and_grad_fn`, `view_as_complex` and
    `view_as_real`.
  * Added new convex operators module with ops `ConvexOperator`,
    `ConvexOperatorL1Norm`, `Regularizer` and `TotalVariationRegularizer`.

Bug Fixes and Other Changes
---------------------------

* `tfmr`:

  * Fixed some bugs that would cause some ops to fail in graph mode.
  * Added graph mode tests.
  * Refactored testing modules.
  * Refactored linear algebra module.
  * Refactored utilities modules.
