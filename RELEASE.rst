Release 0.7.0
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

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>
