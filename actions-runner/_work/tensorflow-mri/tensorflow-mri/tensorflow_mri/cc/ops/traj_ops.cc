/*Copyright 2021 University College London. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "spiral_waveform.h"

using namespace tensorflow;


Status SpiralWaveformShapeFn(shape_inference::InferenceContext* c) {

  int base_resolution;
  int spiral_arms;
  float field_of_view;
  float max_grad_ampl;
  float min_rise_time;
  float dwell_time;
  float readout_os;
  float gradient_delay;
  float larmor_const;
  float vd_inner_cutoff;
  float vd_outer_cutoff;
  float vd_outer_density;
  string vd_type_str;

  TF_RETURN_IF_ERROR(c->GetAttr("base_resolution", &base_resolution));
  TF_RETURN_IF_ERROR(c->GetAttr("spiral_arms", &spiral_arms));
  TF_RETURN_IF_ERROR(c->GetAttr("field_of_view", &field_of_view));
  TF_RETURN_IF_ERROR(c->GetAttr("max_grad_ampl", &max_grad_ampl));
  TF_RETURN_IF_ERROR(c->GetAttr("min_rise_time", &min_rise_time));
  TF_RETURN_IF_ERROR(c->GetAttr("dwell_time", &dwell_time));
  TF_RETURN_IF_ERROR(c->GetAttr("readout_os", &readout_os));
  TF_RETURN_IF_ERROR(c->GetAttr("gradient_delay", &gradient_delay));
  TF_RETURN_IF_ERROR(c->GetAttr("larmor_const", &larmor_const));
  TF_RETURN_IF_ERROR(c->GetAttr("vd_inner_cutoff", &vd_inner_cutoff));
  TF_RETURN_IF_ERROR(c->GetAttr("vd_outer_cutoff", &vd_outer_cutoff));
  TF_RETURN_IF_ERROR(c->GetAttr("vd_outer_density", &vd_outer_density));
  TF_RETURN_IF_ERROR(c->GetAttr("vd_type", &vd_type_str));

  SpiralWaveform::VDType vd_type;
  if (vd_type_str == "linear") {
    vd_type = SpiralWaveform::VDType::Linear;
  } else if (vd_type_str == "quadratic") {
    vd_type = SpiralWaveform::VDType::Quadratic;
  } else if (vd_type_str == "hanning") {
    vd_type = SpiralWaveform::VDType::Hanning;
  }

  // At the moment we do not have a way to infer the output shape of the
  // waveform without actually computing it, so we compute the waveform.
  float waveform_ptr[SWF_MAX_WAVEFORM_SIZE * 2];
  long waveform_length = 0;
  int result = calculate_spiral_trajectory(&waveform_ptr[0],
                                           &waveform_length,
                                           (long) base_resolution,
                                           (long) spiral_arms,
                                           (double) field_of_view,
                                           (double) max_grad_ampl,
                                           (double) min_rise_time,
                                           (double) dwell_time,
                                           (double) readout_os,
                                           (double) gradient_delay,
                                           (double) larmor_const,
                                           (double) vd_inner_cutoff,
                                           (double) vd_outer_cutoff,
                                           (double) vd_outer_density,
                                           (int) vd_type);

  const PartialTensorShape output_shape({waveform_length, 2});

  shape_inference::ShapeHandle shape_handle;
  TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(output_shape, &shape_handle));
  c->set_output(0, shape_handle);

  shape_inference::ShapeAndType shape_and_type(shape_handle, DT_FLOAT);
  std::vector<shape_inference::ShapeAndType> shapes_and_types({shape_and_type});
  c->set_output_handle_shapes_and_types(0, shapes_and_types);

  return Status::OK();
}


REGISTER_OP("SpiralWaveform")
  .Output("waveform: float")
  .Attr("base_resolution: int")
  .Attr("spiral_arms: int")
  .Attr("field_of_view: float")
  .Attr("max_grad_ampl: float")
  .Attr("min_rise_time: float")
  .Attr("dwell_time: float")
  .Attr("readout_os: float = 2.0")
  .Attr("gradient_delay: float = 0.0")
  .Attr("larmor_const: float = 42.577478518")
  .Attr("vd_inner_cutoff: float = 1.0")
  .Attr("vd_outer_cutoff: float = 1.0")
  .Attr("vd_outer_density: float = 1.0")
  .Attr("vd_type: {'linear', 'quadratic', 'hanning'} = 'linear'")
  .SetShapeFn(SpiralWaveformShapeFn)
  .Doc(R"doc(
Calculate a spiral readout waveform.

When using variable-density spirals, *k*-space is divided into three portions.
If 0.0 is the center of *k*-space and 1.0 is the edge, the portions are defined
as follows:

* A Nyquist-density portion between 0.0 and `vd_inner_cutoff`, sampled at the
  Nyquist rate.
* A variable-density portion between `vd_inner_cutoff` and
  `vd_outer_cutoff`, sampled at a variable rate between the Nyquist rate
  and `vd_outer_density` times the Nyquist rate. The rate of variation
  is determined by `vd_type`.
* A fixed-density portion between `vd_outer_cutoff` and 1.0, sampled at
  `vd_outer_density` times the Nyquist rate.

.. [1] Pipe, J.G. and Zwart, N.R. (2014), Spiral trajectory design: A flexible
  numerical algorithm and base analytical equations. Magn. Reson. Med, 71:
  278-285. https://doi.org/10.1002/mrm.24675

base_resolution: The base resolution, or number of pixels in the readout
  dimension.
spiral_arms: The number of spiral arms that a fully sampled k-space should be
  divided into.
field_of_view: The field of view, in mm.
max_grad_ampl: The maximum allowed gradient amplitude, in mT/m.
min_rise_time: The minimum allowed rise time, in us/(mT/m).
dwell_time: The digitiser's real dwell time, in us. This does not include
  oversampling. The effective dwell time (with oversampling) is equal to
  `dwell_time * readout_os`.
readout_os: The readout oversampling factor.
gradient_delay: The system's gradient delay relative to the ADC, in us.
larmor_const: The Larmor constant of the imaging nucleus, in MHz/T.
vd_inner_cutoff: Defines the inner, high-density portion of *k*-space.
  Must be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0 is
  the edge. Between 0.0 and `vd_inner_cutoff`, *k*-space will be sampled
  at the Nyquist rate.
vd_outer_cutoff: Defines the outer, low-density portion of *k*-space. Must
  be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0 is the
  edge. Between `vd_outer_cutoff` and 1.0, *k*-space will be sampled at a
  rate `vd_outer_density` times the Nyquist rate.
vd_outer_density: Defines the sampling density in the outer portion of
  *k*-space. Must be > 0.0. Higher means more densely sampled. Multiplies the
  Nyquist rate: 1.0 means sampling at the Nyquist rate, < 1.0 means undersampled
  and > 1.0 means oversampled.
vd_type: Defines the rate of variation of the sampling density the
  variable-density portion of *k*-space, i.e., between `vd_inner_cutoff`
  and `vd_outer_cutoff`. Must be one of `'linear'`, `'quadratic'` or
  `'hanning'`.
)doc");
