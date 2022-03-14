tfmri
=====

.. automodule:: tensorflow_mri

Modules
-------

.. autosummary::
    :nosignatures:

    callbacks
    io
    layers
    losses
    metrics
    plot
    recon
    summary


Classes
-------

.. autosummary::
    :toctree: tfmri/ops
    :template: ops/class.rst
    :nosignatures:

    CoilCompressorSVD
    ConvexFunction
    ConvexFunctionAffineMappingComposition
    ConvexFunctionLinearOperatorComposition
    ConvexFunctionL1Norm
    ConvexFunctionL2Norm
    ConvexFunctionL2NormSquared
    ConvexFunctionLeastSquares
    ConvexFunctionQuadratic
    ConvexFunctionTikhonov
    ConvexFunctionTotalVariation


Functions
---------

.. autosummary::
    :toctree: tfmri/ops
    :template: ops/function.rst
    :nosignatures:

    admm_minimize
    atanfilt
    block_soft_threshold
    broadcast_dynamic_shapes
    broadcast_static_shapes
    cartesian_product
    central_crop
    combine_coils
    compress_coils
    crop_kspace
    density_grid
    estimate_coil_sensitivities
    estimate_density
    estimate_radial_density
    euler_to_rotation_matrix_3d
    expand_density
    expand_trajectory
    extract_from_complex
    fftn
    filter_kspace
    flatten_density
    flatten_trajectory
    hann
    hamming
    ifftn
    lbfgs_minimize
    make_val_and_grad_fn
    meshgrid
    normalize_no_nan
    radial_density
    radial_trajectory
    radial_waveform
    random_sampling_mask
    ravel_multi_index
    resize_with_crop_or_pad
    rotate_2d
    rotate_3d
    scale_by_min_max
    shrinkage
    soft_threshold
    spiral_trajectory
    spiral_waveform
    unravel_index
    view_as_complex
    view_as_real
