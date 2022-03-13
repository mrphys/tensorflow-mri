tfmri
=====

.. automodule:: tensorflow_mri

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
    LinearOperatorFiniteDifference
    LinearOperatorMRI


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
    conjugate_gradient
    crop_kspace
    density_grid
    estimate_coil_sensitivities
    estimate_density
    estimate_radial_density
    euler_to_rotation_matrix_3d
    expand_density
    expand_trajectory
    extract_from_complex
    extract_glimpses
    fftn
    filter_kspace
    flatten_density
    flatten_trajectory
    gmsd
    gmsd2d
    gmsd3d
    gradient_magnitude
    hann
    hamming
    ifftn
    image_gradients
    lbfgs_minimize
    make_val_and_grad_fn
    meshgrid
    normalize_no_nan
    phantom
    psnr
    psnr2d
    psnr3d
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
    ssim
    ssim2d
    ssim3d
    ssim_multiscale
    ssim2d_multiscale
    ssim3d_multiscale
    total_variation
    unravel_index
    view_as_complex
    view_as_real
