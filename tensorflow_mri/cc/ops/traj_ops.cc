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

using namespace tensorflow;


Status SpiralWaveformShapeFn(shape_inference::InferenceContext* c) {
  
  const PartialTensorShape output_shape({-1, 2});

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
  .SetShapeFn(SpiralWaveformShapeFn)
  .Doc(R"doc(
Calculate a spiral readout waveform.

base_resolution: The base resolution, or number of pixels in the readout
  dimension.
spiral_arms: The number of spiral arms that a fully sampled k-space should be
  divided into.
field_of_view: The field of view, in mm.
max_grad_ampl: The maximum allowed gradient amplitude, in mT/m.
min_rise_time: The minimum allowed rise time, in us/(mT/m).
dwell_time: The digitiser's real dwell time, in us. This does not
  include oversampling. The effective dwell time (with oversampling) is
  equal to `dwell_time * readout_os`.
readout_os: The readout oversampling factor.
gradient_delay: The system's gradient delay relative to the ADC,
  in us.
larmor_const: The Larmor constant of the imaging nucleus, in
  MHz/T.
)doc");
