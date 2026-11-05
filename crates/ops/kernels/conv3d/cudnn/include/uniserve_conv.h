// SPDX-License-Identifier: Apache-2.0
//
// The C header of a convolution module: the call structure the operator
// `crates/ops/src/conv.rs` passes (`conv::Call`, mirrored field by field with
// its size and its error offset asserted), the element types it prints, and the
// entry the package declares.
//
// One module serves all three convolutions: the call names the spatial extents
// as three axes, so a 1-d convolution passes `[l, 1, 1]` and a 3-d one
// `[d, h, w]`, and `transposed` says which pass cuDNN runs.

#pragma once

#include <cstddef>
#include <cstdint>

struct uniserve_conv_call {
  void* out;
  const void* input;
  const void* weight;
  // Null when the operation has no bias.
  const void* bias;
  int64_t input_shape[5];
  int64_t weight_shape[5];
  int64_t out_shape[5];
  int32_t kernel[3];
  int32_t stride[3];
  int32_t padding[3];
  int32_t dilation[3];
  int32_t groups;
  // 0 for a convolution, 1 for a transposed one.
  int32_t transposed;
  // 1 when the library may use TF32 for an FP32 convolution. With 0 the module uses
  // only an engine that does not narrow its inputs; a convolution whose inputs are
  // not FP32 may not set it, and is refused if it does.
  int32_t tf32;
  // 0 for channels-first (NCDHW, the default), 1 for channels-last (NDHWC). The shapes in this
  // call are logical [n, c, d, h, w] either way: only the strides the entry derives change, and a
  // package with no channels-last path refuses 1 by name rather than reading it as 0.
  int32_t channels_last;
  char error[512];
};

// The operator crate's `DType::as_i64()` numbering.
enum uniserve_conv_dtype : int64_t {
  UNISERVE_CONV_DTYPE_U8 = 0,
  UNISERVE_CONV_DTYPE_I32 = 1,
  UNISERVE_CONV_DTYPE_I64 = 2,
  UNISERVE_CONV_DTYPE_F16 = 3,
  UNISERVE_CONV_DTYPE_BF16 = 4,
  UNISERVE_CONV_DTYPE_F32 = 5,
  UNISERVE_CONV_DTYPE_F8E4M3 = 6,
  UNISERVE_CONV_DTYPE_F4E2M1 = 7,
};

static_assert(offsetof(uniserve_conv_call, input_shape) == 32,
              "the convolution call's input shape must follow its four pointers");
// 216: four pointers, `[3][5]` shape triples, four `[3]` geometry triples, `groups`, `transposed`,
// `tf32` and `channels_last`. The message still ends the structure at 728 - the order filled the four
// bytes of tail padding the alignment had left.
static_assert(offsetof(uniserve_conv_call, error) == 216,
              "the convolution call's error message must sit after its geometry");
static_assert(sizeof(uniserve_conv_call) == 728,
              "the convolution call must be the size of the operator's layout");
