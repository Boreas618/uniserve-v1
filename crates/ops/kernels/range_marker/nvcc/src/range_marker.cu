// SPDX-License-Identifier: Apache-2.0

// The device side of the engine's range markers (`crates/engine/src/profiling/markers.rs`): a
// stamp of the device's `%globaltimer` at the entry and the exit of a model module, and a read of
// the same clock for a calibration against the host's clocks.
//
// Both kernels are single-thread work launched with one block of one warp (the warp the module's
// metadata states, the narrowest launch a cubin has); lanes past the first return at once.
// Neither synchronizes anything, so a CUDA graph captures the stamp kernel and every replay runs
// it.
//
// ## The stamp buffer
//
// The stamps live in a double buffer of pinned host storage the engine allocated, `2 * N` slots of
// 16 bytes, `N = 1 << shift`. Sequence number `s` (taken from a device counter) writes slot
// `s mod N` of half `(s / N) mod 2`, and `s / N` is the half's epoch. A slot holds two words:
//
// - `time`: `%globaltimer`, nanoseconds;
// - `packed`: the marker id in bits 0-19, the low 12 bits of the epoch in bits 20-31 (the tag a
//   reader takes the stamp by) and the step in bits 32-63.
//
// ## The ordering a reader relies on
//
// A stamp writes `time` first and then `packed` with a release store at system scope, so a reader
// that acquires `packed` with the epoch's tag reads that stamp's `time`. Each stamp then counts
// itself into its half's `written` word with an acquire-release increment at system scope; the
// stamp that brings the count to `N` completes the half, takes the count back by `N` and
// publishes `epoch + 1` into the half's `filled` word with a release store at system scope. The
// increments form one release sequence, so a reader that acquires a half's `filled` word reads
// every stamp of the half: a complete half has no stamp the host cannot see yet.

#include <cuda/atomic>
#include <cuda/ptx>

#include <cstdint>

namespace {

// The bits of a stamp's marker id and of its epoch tag.
constexpr uint64_t kMarkerMask = (uint64_t{1} << 20) - 1;
constexpr uint64_t kTagMask = (uint64_t{1} << 12) - 1;

using HostWord = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
using HostCount = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
using DeviceWord = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>;
using DeviceCounter = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>;

}  // namespace

// A range marker, in the parameter order of the engine's launch:
//
// - `switches`: `u32[1 + classes]` of device storage, the master word then a word per class;
// - `klass`: the marker's word, 0 for a step marker, which is on with the master word;
// - `marker`: the marker's id (20 bits);
// - `cell`: the stream's step cell (`u32` of device storage);
// - `mark`, `step`: a step marker (`mark != 0`) stores `step` into the cell; a module's marker
//   reads the step from it;
// - `counter`: the `u64` sequence counter, `written`: `u32[2]`, the stamps written into each
//   half's current epoch, both device storage;
// - `stamps`: `16 * 2 * N` bytes of pinned host storage, `filled`: `u64[2]` of pinned host
//   storage;
// - `shift`: `log2 N`.
extern "C" __global__ void uniserve_range_marker(uint32_t* switches, uint32_t klass,
                                                 uint32_t marker, uint32_t* cell, uint32_t mark,
                                                 uint32_t step, unsigned long long* counter,
                                                 uint32_t* written, unsigned long long* stamps,
                                                 unsigned long long* filled, uint32_t shift) {
  if (threadIdx.x != 0) {
    return;
  }

  // The switch words are written by the host's fills on another stream while graphs replay, so
  // they are read at device scope every time rather than from a cache line of an earlier launch.
  if (DeviceWord(switches[0]).load(cuda::memory_order_relaxed) == 0) {
    return;
  }
  if (klass != 0 && DeviceWord(switches[klass]).load(cuda::memory_order_relaxed) == 0) {
    return;
  }

  // The step marker that precedes a step's forward on this stream names the step; the module
  // markers the forward runs read it.
  if (mark != 0) {
    DeviceWord(*cell).store(step, cuda::memory_order_relaxed);
  } else {
    step = DeviceWord(*cell).load(cuda::memory_order_relaxed);
  }

  const uint64_t time = cuda::ptx::get_sreg_globaltimer();
  const uint64_t sequence = DeviceCounter(*counter).fetch_add(1, cuda::memory_order_relaxed);
  const uint64_t half_stamps = uint64_t{1} << shift;
  const uint64_t epoch = sequence >> shift;
  const uint64_t half = epoch & 1;
  const uint64_t slot = half * half_stamps + (sequence & (half_stamps - 1));

  const uint64_t packed = (uint64_t{marker} & kMarkerMask) | ((epoch & kTagMask) << 20) |
                          (uint64_t{step} << 32);
  unsigned long long* words = stamps + 2 * slot;
  HostWord(words[0]).store(time, cuda::memory_order_relaxed);
  HostWord(words[1]).store(packed, cuda::memory_order_release);

  // The stamp that completes its half publishes the half's epoch plus one, after every stamp of
  // the half (module comment).
  const uint32_t count = HostCount(written[half]).fetch_add(1, cuda::memory_order_acq_rel);
  if (count + 1 != half_stamps) {
    return;
  }
  HostCount(written[half]).fetch_sub(static_cast<uint32_t>(half_stamps),
                                     cuda::memory_order_relaxed);
  HostWord(filled[half]).store(epoch + 1, cuda::memory_order_release);
}

// A clock read for a calibration: stores `%globaltimer` into `out`, a word of pinned host storage
// the engine reads once the kernel completed.
extern "C" __global__ void uniserve_clock(unsigned long long* out) {
  if (threadIdx.x != 0) {
    return;
  }
  HostWord(*out).store(cuda::ptx::get_sreg_globaltimer(), cuda::memory_order_release);
}
