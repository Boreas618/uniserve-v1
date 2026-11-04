// The call structure of `frame_moments`: the operator's `FrameMomentsCall`,
// field for field and offset for offset.
//
// The Program Server reads the entry's answer and its error message from this
// structure, and `crates/ops/src/frame.rs` asserts the same offsets on its side.

#ifndef UNISERVE_FRAME_MOMENTS_H
#define UNISERVE_FRAME_MOMENTS_H

#include <cstddef>
#include <cstdint>

/// The largest message an entry writes into a call structure.
#define UNISERVE_FRAME_MOMENTS_MESSAGE 512

typedef struct uniserve_frame_moments_call {
  /// The values' address, `[batch, channels, frames, height, width]` float32.
  uint64_t values;
  /// Elements between batches.
  int64_t values_batch;
  /// Elements between channels.
  int64_t values_channel;
  /// Elements between frames.
  int64_t values_frame;
  /// Frames per batch.
  int64_t frames;
  /// Pixels per frame and channel, `height · width`.
  int64_t pixels;
  /// Channels in one group.
  int64_t channels_per_group;
  /// Groups per frame.
  int64_t groups;
  /// The epsilon added to the variance before its reciprocal square root.
  float epsilon;
  /// The mean buffer's address, `[batch · frames, groups]` float32.
  uint64_t mean;
  /// The reciprocal standard deviation buffer's address, the same shape.
  uint64_t rstd;
  /// The entry's NUL-terminated error message.
  char error[UNISERVE_FRAME_MOMENTS_MESSAGE];
} uniserve_frame_moments_call;

static_assert(offsetof(uniserve_frame_moments_call, values) == 0, "call layout");
static_assert(offsetof(uniserve_frame_moments_call, epsilon) == 64, "call layout");
static_assert(offsetof(uniserve_frame_moments_call, mean) == 72, "call layout");
static_assert(offsetof(uniserve_frame_moments_call, error) == 88, "call layout");
static_assert(sizeof(uniserve_frame_moments_call) == 88 + UNISERVE_FRAME_MOMENTS_MESSAGE,
              "call layout");

#endif  // UNISERVE_FRAME_MOMENTS_H
