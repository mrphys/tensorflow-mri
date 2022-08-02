/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/errors.h"
#define EIGEN_USE_THREADS

// See docs in ../ops/fft_ops.cc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow_mri/cc/third_party/fftw/fftw.h"

namespace tensorflow {
namespace mri {

class FFTBase : public OpKernel {
 public:
  explicit FFTBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const TensorShape& input_shape = in.shape();
    const int fft_rank = Rank();
    OP_REQUIRES(
        ctx, input_shape.dims() >= fft_rank,
        errors::InvalidArgument("Input must have rank of at least ", fft_rank,
                                " but got: ", input_shape.DebugString()));

    Tensor* out;
    TensorShape output_shape = input_shape;
    uint64 fft_shape[3] = {0, 0, 0};

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    if (IsReal()) {
      const Tensor& fft_length = ctx->input(1);
      OP_REQUIRES(ctx,
                  fft_length.shape().dims() == 1 &&
                      fft_length.shape().dim_size(0) == fft_rank,
                  errors::InvalidArgument("fft_length must have shape [",
                                          fft_rank, "]"));

      auto fft_length_as_vec = fft_length.vec<int32>();
      for (int i = 0; i < fft_rank; ++i) {
        OP_REQUIRES(ctx, fft_length_as_vec(i) >= 0,
                    errors::InvalidArgument(
                        "fft_length[", i,
                        "] must >= 0, but got: ", fft_length_as_vec(i)));
        fft_shape[i] = fft_length_as_vec(i);
        // Each input dimension must have length of at least fft_shape[i]. For
        // IRFFTs, the inner-most input dimension must have length of at least
        // fft_shape[i] / 2 + 1.
        bool inner_most = (i == fft_rank - 1);
        uint64 min_input_dim_length =
            !IsForward() && inner_most ? fft_shape[i] / 2 + 1 : fft_shape[i];
        auto input_index = input_shape.dims() - fft_rank + i;
        OP_REQUIRES(
            ctx,
            // We pass through empty tensors, so special case them here.
            input_shape.dim_size(input_index) == 0 ||
                input_shape.dim_size(input_index) >= min_input_dim_length,
            errors::InvalidArgument(
                "Input dimension ", input_index,
                " must have length of at least ", min_input_dim_length,
                " but got: ", input_shape.dim_size(input_index)));
        uint64 dim = IsForward() && inner_most && fft_shape[i] != 0
                         ? fft_shape[i] / 2 + 1
                         : fft_shape[i];
        output_shape.set_dim(output_shape.dims() - fft_rank + i, dim);
      }
    } else {
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] =
            output_shape.dim_size(output_shape.dims() - fft_rank + i);
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (IsReal()) {
      if (IsForward()) {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64) ||
                (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128),
            errors::InvalidArgument("Wrong types for forward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      } else {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT) ||
                (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE),
            errors::InvalidArgument("Wrong types for backward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      }
    } else {
      OP_REQUIRES(
          ctx,
          (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) ||
              (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128),
          errors::InvalidArgument("Wrong types for FFT: in=", in.dtype(),
                                  " out=", out->dtype()));
    }

    if (input_shape.num_elements() == 0) {
      DCHECK_EQ(0, output_shape.num_elements());
      return;
    }

    DoFFT(ctx, in, fft_shape, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsReal() const = 0;

  // The function that actually computes the FFT.
  virtual void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
                     Tensor* out) = 0;
};

typedef Eigen::ThreadPoolDevice CPUDevice;

template <bool Forward, bool _Real, int FFTRank>
class FFTCPU : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  int Rank() const override { return FFTRank; }
  bool IsForward() const override { return Forward; }
  bool IsReal() const override { return _Real; }

  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
    // Create the axes (which are always trailing).
    const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
    auto device = ctx->eigen_device<CPUDevice>();

    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    if (!IsReal()) {
      if (is_complex128) {
        DoComplexFFT<double>(ctx, fft_shape, in, out);
      } else {
        DoComplexFFT<float>(ctx, fft_shape, in, out);
      }
    } else {
      OP_REQUIRES(ctx, false,
                  errors::Unimplemented("Real FFT is not implemented"));
    }
  }

  template <typename FloatType>
  void DoComplexFFT(OpKernelContext* ctx, uint64* fft_shape,
                    const Tensor& in, Tensor* out) {
    auto device = ctx->eigen_device<CPUDevice>();
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    auto num_threads = worker_threads->num_threads;   

    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    if (is_complex128) {
      DCHECK_EQ(in.dtype(), DT_COMPLEX128);
      DCHECK_EQ(out->dtype(), DT_COMPLEX128);
    } else {
      DCHECK_EQ(in.dtype(), DT_COMPLEX64);
      DCHECK_EQ(out->dtype(), DT_COMPLEX64);
    }

    auto input = Tensor(in).flat_inner_dims<std::complex<FloatType>, FFTRank + 1>();
    auto output = out->flat_inner_dims<std::complex<FloatType>, FFTRank + 1>();

    int dim_sizes[FFTRank];
    int input_distance = 1;
    int output_distance = 1;
    int num_points = 1;
    for (int i = 0; i < FFTRank; ++i) {
      dim_sizes[i] = fft_shape[i];
      num_points *= fft_shape[i];
      input_distance *= input.dimension(i + 1);
      output_distance *= output.dimension(i + 1);
    }
    int batch_size = input.dimension(0);

    constexpr auto fft_sign = Forward ? FFTW_FORWARD : FFTW_BACKWARD;
    constexpr auto fft_flags = FFTW_ESTIMATE;

    #pragma omp critical
    {
      static bool is_fftw_initialized = false;
      if (!is_fftw_initialized) {
        // Set up threading for FFTW. Should be done only once.
        #ifdef _OPENMP
        fftw::init_threads<FloatType>();
        fftw::plan_with_nthreads<FloatType>(num_threads);
        #endif
        is_fftw_initialized = true;
      }
    }

    fftw::plan<FloatType> fft_plan;
    #pragma omp critical
    {
      fft_plan = fftw::plan_many_dft<FloatType>(
          FFTRank, dim_sizes, batch_size,
          reinterpret_cast<fftw::complex<FloatType>*>(input.data()),
          nullptr, 1, input_distance,
          reinterpret_cast<fftw::complex<FloatType>*>(output.data()),
          nullptr, 1, output_distance,
          fft_sign, fft_flags);
    }

    fftw::execute<FloatType>(fft_plan);

    #pragma omp critical
    {
      fftw::destroy_plan<FloatType>(fft_plan);
    }

    // Wait until all threads are done using FFTW, then clean up the FFTW state,
    // which only needs to be done once.
    #ifdef _OPENMP
    #pragma omp barrier
    #pragma omp critical
    {
      static bool is_fftw_finalized = false;
      if (!is_fftw_finalized) {
        fftw::cleanup_threads<FloatType>();
        is_fftw_finalized = true;
      }
    }
    #endif  // _OPENMP

    // FFT normalization.
    if (fft_sign == FFTW_BACKWARD) {
      output.device(device) = output / output.constant(num_points);
    }
  }

 private:
  // Used to control access to FFTW planner.
  mutex mu_;
};

// Environment variable `TFMRI_USE_CUSTOM_FFT` can be used to specify whether to
// use custom FFT kernels.
static bool InitModule() {
  const char* use_fftw_string = std::getenv("TFMRI_USE_CUSTOM_FFT");
  bool use_fftw;
  if (use_fftw_string == nullptr) {
    // Default to using FFTW if environment variable is not set.
    use_fftw = true;
  } else {
    // Parse the value of the environment variable.
    std::string str(use_fftw_string);
    // To lower-case.
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (str == "y" || str == "yes" || str == "t" || str == "true" ||
        str == "on" || str == "1") {
      use_fftw = true;
    } else if (str == "n" || str == "no" || str == "f" || str == "false" ||
               str == "off" || str == "0") {
      use_fftw = false;
    } else {
      LOG(FATAL) << "Invalid value for environment variable "
                 << "TFMRI_USE_CUSTOM_FFT: " << str;
    }
  }
  if (use_fftw) {
    // Register with priority 1 so that these kernels take precedence over the
    // default Eigen implementation. Note that core TF registers the FFT GPU
    // kernels with priority 1 too, so those still take precedence over these.
    REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<true, false, 1>);
    REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<false, false, 1>);
    REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<true, false, 2>);
    REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<false, false, 2>);
    REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<true, false, 3>);
    REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_CPU).Priority(1),
                            FFTCPU<false, false, 3>);
  }
  return true;
}

static bool module_initialized = InitModule();

}  // namespace mri
}  // namespace tensorflow
