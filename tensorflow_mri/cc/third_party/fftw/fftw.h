/* Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_MRI_CC_THIRD_PARTY_FFTW_H_
#define TENSORFLOW_MRI_CC_THIRD_PARTY_FFTW_H_

#include <fftw3.h>


namespace tensorflow {
namespace mri {
namespace fftw {

template<typename FloatType>
inline int init_threads();

template<>
inline int init_threads<float>() {
  return fftwf_init_threads();
}

template<>
inline int init_threads<double>() {
  return fftw_init_threads();
}

template<typename FloatType>
inline void cleanup_threads();

template<>
inline void cleanup_threads<float>() {
  return fftwf_cleanup_threads();
}

template<>
inline void cleanup_threads<double>() {
  return fftw_cleanup_threads();
}

template<typename FloatType>
inline void plan_with_nthreads(int nthreads);

template<>
inline void plan_with_nthreads<float>(int nthreads) {
  fftwf_plan_with_nthreads(nthreads);
}

template<>
inline void plan_with_nthreads<double>(int nthreads) {
  fftw_plan_with_nthreads(nthreads);
}

template<typename FloatType>
inline void make_planner_thread_safe();

template<>
inline void make_planner_thread_safe<float>() {
  fftwf_make_planner_thread_safe();
}

template<>
inline void make_planner_thread_safe<double>() {
  fftw_make_planner_thread_safe();
}

template<typename FloatType>
struct ComplexType;

template<>
struct ComplexType<float> {
  using Type = fftwf_complex;
};

template<>
struct ComplexType<double> {
  using Type = fftw_complex;
};

template<typename FloatType>
using complex = typename ComplexType<FloatType>::Type;

template<typename FloatType>
inline FloatType* alloc_real(size_t n);

template<>
inline float* alloc_real<float>(size_t n) {
  return fftwf_alloc_real(n);
}

template<>
inline double* alloc_real<double>(size_t n) {
  return fftw_alloc_real(n);
}

template<typename FloatType>
inline typename ComplexType<FloatType>::Type* alloc_complex(size_t n);

template<>
inline typename ComplexType<float>::Type* alloc_complex<float>(size_t n) {
  return fftwf_alloc_complex(n);
}

template<>
inline typename ComplexType<double>::Type* alloc_complex<double>(size_t n) {
  return fftw_alloc_complex(n);
}

template<typename FloatType>
inline void free(void* p);

template<>
inline void free<float>(void* p) {
  fftwf_free(p);
}

template<>
inline void free<double>(void* p) {
  fftw_free(p);
}

template<typename FloatType>
struct PlanType;

template<>
struct PlanType<float> {
  using Type = fftwf_plan;
};

template<>
struct PlanType<double> {
  using Type = fftw_plan;
};

template<typename FloatType>
using plan = typename PlanType<FloatType>::Type;

template<typename FloatType>
inline typename PlanType<FloatType>::Type plan_many_dft(
    int rank, const int *n, int howmany,
    typename ComplexType<FloatType>::Type *in, const int *inembed,
    int istride, int idist,
    typename ComplexType<FloatType>::Type *out, const int *onembed,
    int ostride, int odist,
    int sign, unsigned flags);

template<>
inline typename PlanType<float>::Type plan_many_dft<float>(
    int rank, const int *n, int howmany,
    ComplexType<float>::Type *in, const int *inembed,
    int istride, int idist,
    ComplexType<float>::Type *out, const int *onembed,
    int ostride, int odist,
    int sign, unsigned flags) {
  return fftwf_plan_many_dft(
      rank, n, howmany,
      in, inembed, istride, idist,
      out, onembed, ostride, odist,
      sign, flags);
}

template<>
inline typename PlanType<double>::Type plan_many_dft<double>(
    int rank, const int *n, int howmany,
    typename ComplexType<double>::Type *in, const int *inembed,
    int istride, int idist,
    typename ComplexType<double>::Type *out, const int *onembed,
    int ostride, int odist,
    int sign, unsigned flags) {
  return fftw_plan_many_dft(
      rank, n, howmany,
      in, inembed, istride, idist,
      out, onembed, ostride, odist,
      sign, flags);
}

template<typename FloatType>
inline void execute(typename PlanType<FloatType>::Type& plan);  // NOLINT

template<>
inline void execute<float>(typename PlanType<float>::Type& plan) {  // NOLINT
  fftwf_execute(plan);
}

template<>
inline void execute<double>(typename PlanType<double>::Type& plan) {  // NOLINT
  fftw_execute(plan);
}

template<typename FloatType>
inline void destroy_plan(typename PlanType<FloatType>::Type& plan);  // NOLINT

template<>
inline void destroy_plan<float>(typename PlanType<float>::Type& plan) {  // NOLINT
  fftwf_destroy_plan(plan);
}

template<>
inline void destroy_plan<double>(typename PlanType<double>::Type& plan) {  // NOLINT
  fftw_destroy_plan(plan);
}

}  // namespace fftw
}  // namespace mri
}  // namespace tensorflow

#endif  // TENSORFLOW_MRI_CC_THIRD_PARTY_FFTW_H_
