Release 0.9.0
=============

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
    * Dropout layers may now be added with `use_dropout=True`. Additionally,
      the parameters `dropout_rate` and `dropout_type` can be used to specify
      the dropout rate and type (standard or spatial), respectively.
    
  * Added optional dropout layers to `UNet`. Dropout can be configured with the
    parameters `use_dropout`, `dropout_rate` and `dropout_type`.
