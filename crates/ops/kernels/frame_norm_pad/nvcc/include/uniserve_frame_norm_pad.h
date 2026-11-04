// The call structure of `frame_norm_pad`: the operator's `FrameNormPadCall`,
// field for field and offset for offset.
//
// The Program Server reads the entry's answer and its error message from this
// structure, and `crates/ops/src/frame.rs` asserts the same offsets on its side.

#ifndef UNISERVE_FRAME_NORM_PAD_H
#define UNISERVE_FRAME_NORM_PAD_H

#include <cstddef>
#include <cstdint>

/// The largest message an entry writes into a call structure.
#define UNISERVE_FRAME_NORM_PAD_MESSAGE 512

typedef struct uniserve_frame_norm_pad_call {
  /// The values' address, `[batch, channels, frames, height, width]` float32.
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
  /// The normalization branch the kernel takes.
  int32_t normalization;
  /// The entry's NUL-terminated error message.
  char error[UNISERVE_FRAME_NORM_PAD_MESSAGE];
} uniserve_frame_norm_pad_call;

static_assert(offsetof(uniserve_frame_norm_pad_call, values) == 0, "call layout");
static_assert(offsetof(uniserve_frame_norm_pad_call, weight) == 48, "call layout");
static_assert(offsetof(uniserve_frame_norm_pad_call, out) == 64, "call layout");
static_assert(offsetof(uniserve_frame_norm_pad_call, channels) == 112, "call layout");
static_assert(offsetof(uniserve_frame_norm_pad_call, reflect) == 208, "call layout");
static_assert(offsetof(uniserve_frame_norm_pad_call, error) == 216, "call layout");
static_assert(sizeof(uniserve_frame_norm_pad_call) == 216 + UNISERVE_FRAME_NORM_PAD_MESSAGE,
              "call layout");

#endif  // UNISERVE_FRAME_NORM_PAD_H
