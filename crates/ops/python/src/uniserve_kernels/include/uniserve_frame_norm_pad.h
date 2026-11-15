// SPDX-License-Identifier: Apache-2.0
//
// The call structure of `frame_norm_pad`'s packages, and the normalizations it numbers, generated
// by cbindgen from the operator crate's `crates/ops/src/frame.rs` (`ops::abi`): do not edit.
//
// The assertions at the end state the size and every field offset of each structure as Rust lays it
// out.
//
// `cargo test -p ops` compares this file with the generator's output, and this command writes it:
//
//   UPDATE_EXPECT=1 cargo test -p ops abi::headers

#pragma once

#include <cstddef>
#include <cstdint>
#include "uniserve_abi.h"

/// The normalization a frame's values take in `frame_norm_pad`: the branch PyTorch's
/// `GroupNormKernelImplInternal` takes for the parameters and the frame's size. The call carries
/// its number (`uniserve_normalization`).
enum uniserve_normalization : int32_t {
  /// `ComputeFusedParamsCUDAKernel`'s `a = rstd · gamma` and `b = −a · mean + beta`, then the
  /// apply's `a · x + b`.
  UNISERVE_NORMALIZATION_FUSED = 0,
  /// No affine parameters: `(x − mean) · rstd`.
  UNISERVE_NORMALIZATION_CENTERED = 1,
  /// One-pixel frames (`GroupNorm1dForward`).
  UNISERVE_NORMALIZATION_SINGLEPIXEL = 2,
};

/// The call structure of `frame_norm_pad`, declared to the package as
/// `uniserve_frame_norm_pad_call` (`uniserve_frame_norm_pad.h`, generated with its layout
/// asserted: [`abi`]).
struct uniserve_frame_norm_pad_call {
  /// The values' address.
  uint64_t values;
  /// Elements between batches.
  int64_t values_batch;
  /// Elements between channels.
  int64_t values_channel;
  /// Elements between frames.
  int64_t values_frame;
  /// The per-frame group means' address.
  uint64_t mean;
  /// The per-frame reciprocal standard deviations' address.
  uint64_t rstd;
  /// The weight's address, or zero when the kernel normalizes without one.
  uint64_t weight;
  /// The bias's address, or zero when the kernel normalizes without one.
  uint64_t bias;
  /// The padded output's address.
  uint64_t out;
  /// Elements between output batches.
  int64_t out_batch;
  /// Elements between output channels.
  int64_t out_channel;
  /// Elements between output frames.
  int64_t out_frame;
  /// Elements between output rows.
  int64_t out_height;
  /// Elements between output columns.
  int64_t out_width;
  /// Channels of the source.
  int64_t channels;
  /// Frames of the source.
  int64_t frames;
  /// Rows of the source.
  int64_t height;
  /// Columns of the source.
  int64_t width;
  /// Frames of the output.
  int64_t out_frames;
  /// Rows of the output.
  int64_t out_rows;
  /// Columns of the output.
  int64_t out_columns;
  /// Channels in one group.
  int64_t channels_per_group;
  /// Groups per frame.
  int64_t groups;
  /// Output rows above the source.
  int64_t top;
  /// Output columns left of the source.
  int64_t left;
  /// Zero frames written before the source.
  int64_t front;
  /// Mirror the padded rows and columns instead of replicating them.
  int32_t reflect;
  /// The [`Normalization`] the kernel takes, by its number.
  int32_t normalization;
  /// The entry's NUL-terminated error message.
  char error[UNISERVE_MESSAGE];
};

static_assert(sizeof(uniserve_frame_norm_pad_call) == 728, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, values) == 0, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, values_batch) == 8, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, values_channel) == 16, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, values_frame) == 24, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, mean) == 32, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, rstd) == 40, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, weight) == 48, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, bias) == 56, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out) == 64, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_batch) == 72, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_channel) == 80, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_frame) == 88, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_height) == 96, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_width) == 104, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, channels) == 112, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, frames) == 120, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, height) == 128, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, width) == 136, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_frames) == 144, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_rows) == 152, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, out_columns) == 160, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, channels_per_group) == 168, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, groups) == 176, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, top) == 184, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, left) == 192, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, front) == 200, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, reflect) == 208, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, normalization) == 212, "the operator crate's layout of uniserve_frame_norm_pad_call");
static_assert(offsetof(uniserve_frame_norm_pad_call, error) == 216, "the operator crate's layout of uniserve_frame_norm_pad_call");
