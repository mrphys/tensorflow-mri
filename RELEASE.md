# Release 0.1.0-alpha.6

## Breaking Changes

  * `tfmr.radial_density` will now return the sampling density rather than the
    density compensation weights.

## Known Caveats


## Major Features and Improvements

  * Added new ops `tfmr.central_crop` and `tfmr.symmetric_pad_or_crop` for array
    manipulation.
  * Added new op `tfmr.estimate_coil_sens_maps` for coil sensitivity estimation.
    The op supports Walsh's method, Inati's iterative method and ESPIRiT. The op
    supports 2D and 3D images. 
  * Added new ops `tfmr.fftn` and `tfmr.ifftn` for N-D FFT calculation.

## Bug Fixes and Other Changes



# Release X.Y.Z

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

## Breaking Changes

*<DOCUMENT BREAKING CHANGES HERE>
*<THIS SECTION SHOULD CONTAIN API, ABI AND BEHAVIORAL BREAKING CHANGES>

## Known Caveats

*<CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
*<ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
*<KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>

## Major Features and Improvements

*<INSERT MAJOR FEATURE HERE, USING MARKDOWN SYNTAX>

## Bug Fixes and Other Changes

*<SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
*<IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
