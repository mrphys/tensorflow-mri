Release 0.9.0
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

  * Added new parameter `tiny_number` to `radial_trajectory` and
    `spiral_trajectory`. This parameter can be used to control which tiny golden
    angle is used for tiny golden angle trajectories. Previously it was only
    possible to use the 7th tiny golden angle (which is now the default).
  * `radial_trajectory` and `spiral_trajectory` will now warn the user when
    a non-optimal number of views is specified for golden angle ordering.

* `tfmr.layers`:

  * Added new functionality to `ConvBlock`:

    * A different activation for the last layer can now be specified with
      `out_activation`.
    * Bias vectors for convolutional layers can now be disabled with
      `use_bias=False`.
    * Block can now be made residual with `use_residual=True`.


Bug Fixes and Other Changes
---------------------------

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
