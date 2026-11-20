//! Manifold-constrained hyper-connections (mHC): the operators of a residual held as several
//! streams that each sub-layer reads one input from and writes its output back into.
//!
//! The residual of a token is `h`, `[streams, hidden]`, `n` streams of the hidden width `d`. A
//! sub-layer's connection holds a projection `fn` of `m = n·(n + 2)` rows over the flattened
//! streams, three scales `s` and `m` biases `b`, all FP32, and the gain `w` of the RMS
//! normalization of the sub-layer's input. From the streams it reaches, the connection computes
//! its mixes:
//!
//! ```text
//! r       = rsqrt(mean(flat(h)²) + ε_rms)                    one statistic over all n·d
//! z       = (flat(h) · fnᵀ) · r                              [m]
//! pre_j   = sigmoid(z_j · s₀ + b_j) + ε                      j < n
//! post_j  = 2 · sigmoid(z_{n+j} · s₁ + b_{n+j})              j < n
//! comb    = sinkhorn(z_{2n+jn+k} · s₂ + b_{2n+jn+k})         [n, n], row j, column k
//! ```
//!
//! where `sinkhorn` is a row softmax plus `ε`, a division of every column by its sum plus `ε`,
//! and then `iterations − 1` more alternating divisions of the rows and the columns by their
//! sums plus `ε`, which makes the matrix doubly stochastic. The streams are read into the
//! sub-layer's input and its output `y` is written back with:
//!
//! ```text
//! pre    x  = rms_norm(round(Σ_j pre_j · h_j)) · w           the sum in FP32, rounded once
//! post   h'_k = round(post_k · y + Σ_j comb[j, k] · h_j)     every stream, FP32 until the store
//! ```
//!
//! In the **shifted** form the reference evaluates, a sub-layer's input is read with the pre-mix
//! the sub-layer *before* it computed, while the sub-layer's own mixes are computed from the
//! streams it reaches (after the previous output was written back) and its pre-mix is carried
//! to the next one. A boundary between two sub-layers is therefore one pass over the streams:
//!
//! ```text
//! boundary   h  ← post(h, y_prev, post_prev, comb_prev)       when a sub-layer output is pending
//!            (pre, post, comb) ← mixes(h)
//!            x  ← pre(h, pre_carried)
//! ```
//!
//! which is [`MhcBoundary`]; [`MhcPost`] writes an output back alone (before a layer that adds
//! into the streams between two sub-layers, and at the end of the stack) and [`MhcPre`] reads
//! the streams alone (the final collapse into the output norm). Every product of the mixes is
//! FP32 and every rounding is the reference's: `h'` and the collapsed sum `round(Σ pre_j h_j)`
//! into the streams' type, the normalized input once more.
//!
//! The operators' constants are the method's: the post-mix's factor of two and the order of the
//! Sinkhorn steps. The epsilons and the iteration count are the checkpoint's.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::{Key, row_tile};
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The most streams a kernel of these operators holds: its tiles are `[streams, …]` and
/// `[streams, streams]`, so a count is a power of two up to this.
pub const MAX_STREAMS: u32 = 8;

/// Tokens one program of [`MhcBoundary`] holds in a call of at most [`BOUNDARY_FEW`] tokens, and
/// in a longer one: the rows of its products, at least the sixteen a tensor-core product takes.
/// A block of a few tokens keeps a decode's last program short; a longer block reads the
/// projection, which every block reads whole, once for more tokens.
const BOUNDARY_TOKENS: (u32, u32) = (16, 64);

/// The most tokens a call holds that its programs hold in the smaller block.
const BOUNDARY_FEW: u64 = 512;

/// Columns of each stream one step of [`MhcBoundary`] reads, which divides the hidden width.
const BOUNDARY_COLUMNS: u32 = 64;

/// The programs a boundary's grid aims at per token of a program's block: a launch whose token
/// blocks are fewer splits the streams' columns across more programs
/// ([`BoundaryDescriptor::splits`]).
///
/// The count is a property of the kernel, not of the device, so the profile run on the meta
/// device sizes the same workspace a launch on any GPU takes, and two devices compute the same
/// partial sums in the same order.
const BOUNDARY_PROGRAMS_PER_TOKEN: u64 = 16;

/// Tokens one block of the fused form of [`MhcBoundary`] holds: the rows of its tensor-core
/// product of the mixes (a native backend's, DeepGEMM's `sm100_mega_mhc`).
pub const FUSED_TOKENS: u64 = 64;

/// Columns of each stream one step of the fused form reads.
const FUSED_COLUMNS: u32 = 64;

/// The programs the fused form's split rule plans for: the fewest multiprocessors an SM100 part
/// has. A property of the rule, not of the device, so the profile run on the meta device sizes
/// the workspace a launch on any GPU takes ([`BoundaryDescriptor::fused_splits`]).
const FUSED_PROGRAMS: u64 = 148;

/// The fewest and the most column splits of a fused block.
const FUSED_SPLITS: (u64, u64) = (16, 64);

/// The bytes of one arrival word of the fused form: a line of its own, so two blocks' arrivals
/// never share one.
const FUSED_LINE: u64 = 128;

/// The arrival lines of one fused block: its mixes' and its normalization's.
const FUSED_BARRIERS: u64 = 2;

/// Columns of each stream one program of [`MhcPost`] writes.
const POST_COLUMNS: u32 = 1024;

/// The mixes of a connection of `streams` streams: `n·(n + 2)`.
pub const fn mixes(streams: u32) -> u32 {
    streams * (streams + 2)
}

/// Check the streams and the width a descriptor states.
fn check_streams(operator: &'static str, dtype: DType, streams: u32, hidden: u32) -> Result<()> {
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Err(Error::argument(
            operator,
            format!("streams of {} elements", abi::name(dtype)),
        ));
    }
    if !streams.is_power_of_two() || streams > MAX_STREAMS {
        return Err(Error::argument(
            operator,
            format!("{streams} streams: a power of two up to {MAX_STREAMS} expected"),
        ));
    }
    if hidden == 0 {
        return Err(Error::argument(
            operator,
            "streams of zero elements".to_owned(),
        ));
    }
    Ok(())
}

/// Check that `view`, the operand `name`, is the contiguous `dims` of `dtype`.
fn contiguous(
    operator: &'static str,
    name: &str,
    view: &Operand,
    dtype: DType,
    dims: &[u64],
) -> Result<()> {
    let named = |error: Error| Error::argument(operator, format!("{name}: {error}"));

    view.require_shape(operator, dims).map_err(named)?;
    view.require_row(operator, dtype, dims[dims.len() - 1])
        .map_err(named)?;
    if !view.is_row_major() {
        return Err(Error::argument(
            operator,
            format!("{name}: the view {:?} is not contiguous", view.dims()),
        ));
    }
    Ok(())
}

/// One compilation of `mhc_boundary`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BoundaryDescriptor {
    /// Element type of the streams, the pending output, the norm's gain and the input.
    pub dtype: DType,
    /// Streams of the residual, `n`.
    pub streams: u32,
    /// Elements of one stream, `d`.
    pub hidden: u32,
    /// Sinkhorn iterations of the combination matrix.
    pub iterations: u32,
    /// Whether a pending sub-layer output is written back into the streams first.
    pub post: bool,
    /// Tokens one program holds.
    pub tokens: u32,
    /// Columns of each stream one step of a program reads.
    pub block: u32,
}

impl BoundaryDescriptor {
    /// The compilation of a boundary over `tokens` tokens of `streams` streams of `hidden`
    /// `dtype` elements whose combination matrix takes `iterations` Sinkhorn iterations, writing
    /// a pending output back first when `post`. The token count chooses the block of tokens one
    /// program holds, which is the key's only dependence on it.
    pub fn new(
        dtype: DType,
        streams: u32,
        hidden: u32,
        iterations: u32,
        post: bool,
        tokens: u64,
    ) -> Self {
        let (few, many) = BOUNDARY_TOKENS;

        Self {
            dtype,
            streams,
            hidden,
            iterations,
            post,
            tokens: if tokens <= BOUNDARY_FEW { few } else { many },
            block: BOUNDARY_COLUMNS,
        }
    }

    /// The mixes the connection computes, `n·(n + 2)`.
    pub const fn mixes(&self) -> u32 {
        mixes(self.streams)
    }

    /// The lanes of a program's mixes: the mixes rounded up to a power of two of at least the
    /// sixteen columns a tensor-core product takes.
    pub fn mix_tile(&self) -> u32 {
        self.mixes().next_power_of_two().max(16)
    }

    /// FP32 values a split publishes per token: its partial mixes, then its partial sums of the
    /// squares of the streams and of the collapsed input.
    pub fn partial_width(&self) -> u32 {
        self.mix_tile() + 2
    }

    /// The token blocks of a launch over `tokens` tokens: the first axis of its grid, and the
    /// arrival counters a split launch takes.
    pub fn token_blocks(&self, tokens: u64) -> u64 {
        tokens.div_ceil(u64::from(self.tokens))
    }

    /// The programs each token block's columns are split across, for `tokens` tokens: one when
    /// the token blocks alone fill the grid the kernel aims at, else as many as bring the grid to
    /// it, at most one per column block.
    ///
    /// A launch of more than one split publishes `[splits, tokens, partial_width]` FP32
    /// partials and counts each token block's arrivals in one zeroed I32 word
    /// ([`BoundaryArgs::partials`]); the last split of a block to arrive sums the partials in
    /// split order and finishes the block, so the result does not depend on which arrives last.
    pub fn splits(&self, tokens: u64) -> u32 {
        let blocks = self.token_blocks(tokens).max(1);
        let columns = u64::from(self.hidden / self.block).max(1);
        let programs = BOUNDARY_PROGRAMS_PER_TOKEN * u64::from(self.tokens);

        programs.div_ceil(blocks).clamp(1, columns) as u32
    }

    /// The blocks of [`FUSED_TOKENS`] tokens of the fused form over `tokens` tokens.
    pub fn fused_blocks(&self, tokens: u64) -> u64 {
        tokens.div_ceil(FUSED_TOKENS)
    }

    /// The parts each fused block's column steps are split into, for `tokens` tokens: the
    /// fewest equal shares of the steps that still bring the blocks' tasks to the programs the
    /// rule plans for ([`FUSED_PROGRAMS`]), between 16 and 64 parts (DeepGEMM's
    /// `get_num_splits`).
    ///
    /// Each split of a block publishes its partial mixes and its sums of the squares of the
    /// streams and of the collapsed input, `[blocks, splits, 64, mixes + 2]` FP32, and arrives
    /// on the block's two arrival lines; the block's mixes and its normalization sum the
    /// partials in split order.
    pub fn fused_splits(&self, tokens: u64) -> u32 {
        let (fewest, most) = FUSED_SPLITS;
        let steps = u64::from(self.hidden / FUSED_COLUMNS).max(1);
        let blocks = self.fused_blocks(tokens).max(1);
        let planned = (FUSED_PROGRAMS / blocks).clamp(fewest, most);

        steps.div_ceil(steps.div_ceil(planned)) as u32
    }

    /// FP32 values the partials of a launch over `tokens` tokens hold: the larger of what the
    /// split form ([`Self::splits`]) and the fused form ([`Self::fused_splits`]) publish.
    pub fn partial_floats(&self, tokens: u64) -> u64 {
        let split = u64::from(self.splits(tokens)) * tokens * u64::from(self.partial_width());
        let fused = self.fused_blocks(tokens)
            * u64::from(self.fused_splits(tokens))
            * FUSED_TOKENS
            * u64::from(self.mixes() + 2);

        split.max(fused)
    }

    /// The first I32 word of the fused form's arrival lines in the arrival words of a launch
    /// over `tokens` tokens: after the split form's one word per token block, at a whole line.
    pub fn fused_counters(&self, tokens: u64) -> u64 {
        let words = FUSED_LINE / 4;

        self.token_blocks(tokens).div_ceil(words) * words
    }

    /// I32 words of the arrival words of a launch over `tokens` tokens, zero on entry: the split
    /// form's one per token block, then the fused form's two lines per fused block.
    pub fn counter_words(&self, tokens: u64) -> u64 {
        self.fused_counters(tokens) + self.fused_blocks(tokens) * FUSED_BARRIERS * FUSED_LINE / 4
    }
}

impl fmt::Display for BoundaryDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("streams", self.streams)
            .field("hidden", self.hidden)
            .field("iterations", self.iterations)
            .field("post", self.post)
            .field("tokens", self.tokens)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The pending output a boundary writes back into the streams first, and where it writes them.
#[derive(Clone)]
pub struct Pending {
    /// `[tokens, hidden]`, contiguous: the previous sub-layer's output `y`.
    pub output: Operand,
    /// `[tokens, streams]` FP32, contiguous: the previous sub-layer's post-mix.
    pub post: Operand,
    /// `[tokens, streams, streams]` FP32, contiguous: its combination matrix, row `j` the stream
    /// read and column `k` the stream written.
    pub comb: Operand,
    /// `[tokens, streams, hidden]`, contiguous: receives the streams with the output written
    /// back, which the mixes and the input are computed from.
    pub streams: Operand,
}

/// The arguments of `mhc_boundary`.
#[derive(Clone)]
pub struct BoundaryArgs {
    /// `[tokens, streams, hidden]`, contiguous: the streams the boundary reaches, before the
    /// pending output is written back when there is one.
    pub streams: Operand,
    /// The pending output, present exactly when the descriptor writes one back.
    pub pending: Option<Pending>,
    /// `[mixes, streams · hidden]` FP32, contiguous: the connection's projection `fn`.
    pub weight: Operand,
    /// `[3]` FP32: the scales of the pre-mix, the post-mix and the combination logits.
    pub scale: Operand,
    /// `[mixes]` FP32: the biases of the mixes.
    pub base: Operand,
    /// `[tokens, streams]` FP32, contiguous: the pre-mix the input is read with, the one the
    /// sub-layer before computed.
    pub pre: Operand,
    /// `[hidden]`: the gain of the input's RMS normalization.
    pub norm: Operand,
    /// `[tokens, hidden]`, contiguous: receives the sub-layer's input.
    pub input: Operand,
    /// `[tokens, streams]` FP32, contiguous: receives the connection's pre-mix, which the next
    /// sub-layer's input is read with.
    pub next_pre: Operand,
    /// `[tokens, streams]` FP32, contiguous: receives the connection's post-mix.
    pub next_post: Operand,
    /// `[tokens, streams, streams]` FP32, contiguous: receives its combination matrix.
    pub next_comb: Operand,
    /// `[partial_floats]` FP32, contiguous ([`BoundaryDescriptor::partial_floats`]): the
    /// partials the kernel's splits publish, `[splits, tokens, partial_width]` of the split form
    /// or `[blocks, splits, 64, mixes + 2]` of the fused form, from the first value.
    pub partials: Operand,
    /// `[counter_words]` I32, contiguous, zero on entry ([`BoundaryDescriptor::counter_words`]):
    /// the split form's arrival count of each token block, which its last split sets back to
    /// zero, then from [`BoundaryDescriptor::fused_counters`] the fused form's two 128-byte
    /// arrival lines per block, which its launch tags with its grid's identity.
    pub counters: Operand,
    /// `ε_rms`, added to the mean square of the flattened streams.
    pub rms_eps: f32,
    /// `ε` of the pre-mix and of every Sinkhorn step.
    pub eps: f32,
    /// The epsilon of the input's RMS normalization.
    pub norm_eps: f32,
}

/// The call structure the fused form's entry takes, declared to its package as
/// `uniserve_mhc_boundary_call` (`uniserve_mhc_boundary.h`, generated with its layout asserted:
/// [`abi`]).
///
/// The operands' device addresses as [`BoundaryArgs`] states them (the pending output's three
/// and the written streams always present: the fused form writes an output back), the launch's
/// extents, the split count [`BoundaryDescriptor::fused_splits`] states, the multiprocessors
/// of the device its persistent grid covers, and the three epsilons.
#[repr(C)]
#[derive(abi::Layout)]
pub struct BoundaryCall {
    /// `[tokens, streams, hidden]` 16-bit: the streams the boundary reaches.
    pub streams: *const c_void,
    /// `[tokens, hidden]` 16-bit: the pending sub-layer output.
    pub output: *const c_void,
    /// `[tokens, streams]` FP32: its post-mix.
    pub post: *const c_void,
    /// `[tokens, streams, streams]` FP32: its combination matrix.
    pub comb: *const c_void,
    /// `[tokens, streams, hidden]` 16-bit: receives the written streams.
    pub written: *mut c_void,
    /// `[mixes, streams · hidden]` FP32: the connection's projection.
    pub weight: *const c_void,
    /// `[3]` FP32: the scales of the pre-mix, the post-mix and the combination logits.
    pub scale: *const c_void,
    /// `[mixes]` FP32: the mixes' biases.
    pub base: *const c_void,
    /// `[tokens, streams]` FP32: the carried pre-mix the input is read with.
    pub pre: *const c_void,
    /// `[hidden]` 16-bit: the input norm's gain.
    pub norm: *const c_void,
    /// `[tokens, hidden]` 16-bit: receives the normalized input.
    pub input: *mut c_void,
    /// `[tokens, streams]` FP32: receives the pre-mix.
    pub next_pre: *mut c_void,
    /// `[tokens, streams]` FP32: receives the post-mix.
    pub next_post: *mut c_void,
    /// `[tokens, streams, streams]` FP32: receives the combination matrix.
    pub next_comb: *mut c_void,
    /// `[blocks, splits, 64, mixes + 2]` FP32: the splits' partials.
    pub partials: *mut c_void,
    /// `[blocks, 2, 32]` I32, zero or tagged by an earlier grid: the blocks' arrival lines.
    pub lines: *mut c_void,
    pub tokens: i64,
    /// Streams of the residual.
    pub stream_count: i64,
    /// Elements of one stream.
    pub hidden: i64,
    /// Sinkhorn iterations of the combination matrix.
    pub iterations: i64,
    /// Column splits of each block ([`BoundaryDescriptor::fused_splits`]).
    pub splits: i64,
    /// The multiprocessors of the device the launch runs on: the persistent grid's extent.
    pub multiprocessors: i64,
    /// `ε_rms` of the flattened streams' mean square.
    pub rms_eps: f32,
    /// `ε` of the pre-mix and of every Sinkhorn step.
    pub eps: f32,
    /// The epsilon of the input's RMS normalization.
    pub norm_eps: f32,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for BoundaryCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// One boundary between two sub-layers of an mHC residual: the pending output written back, the
/// next sub-layer's mixes, and its input read with the carried pre-mix and normalized (module
/// documentation).
///
/// Two kinds of program serve it. A **split** program (a cubin, the Triton package) holds a block
/// of tokens and walks its columns of every stream once: it writes the pending output back, adds
/// the columns' products into the mixes (Triton's three-pass TF32 product of the FP32
/// projection), their squares into the mixes' statistic, and stores the columns of the collapsed
/// input with their squares summed. A block's columns are split across several programs when
/// there are few tokens; the last to arrive finishes the block: it scales the mixes, splits
/// them, runs the Sinkhorn iterations and normalizes the collapsed input. Its grid is the
/// operator's, `(token blocks, splits)`.
///
/// A **fused** program (a native module, DeepGEMM's persistent `sm100_mega_mhc`) does the same
/// over blocks of [`FUSED_TOKENS`] tokens whose column steps it splits into
/// [`BoundaryDescriptor::fused_splits`] parts: warp-specialized roles load the streams, the
/// pending output and the projection by TMA, write the output back, multiply the written
/// streams with the projection on the tensor cores (one TF32 pass, the projection read truncated
/// to TF32), and finish each block once its splits arrived. Its call is [`BoundaryCall`], which
/// carries the multiprocessors its persistent grid covers. It computes the shifted boundary of
/// four 16-bit streams that writes a pending output back.
pub struct MhcBoundary;

impl MhcBoundary {
    /// Calls the fused form's native entry with `args`, which [`MhcBoundary::check`] accepted.
    fn launch_fused(
        kernel: &Kernel,
        descriptor: &BoundaryDescriptor,
        args: &BoundaryArgs,
        stream: &Stream,
    ) -> Result<()> {
        let pending = args.pending.as_ref().ok_or_else(|| {
            Error::argument(
                Self::NAME,
                "the fused form writes a pending output back, and none was given".to_owned(),
            )
        })?;
        let tokens = args.streams.dim(0);
        let address = |operand: &Operand| operand.ptr.address() as usize as *const c_void;
        let target = |operand: &Operand| operand.ptr.address() as usize as *mut c_void;
        // The fused form's lines follow the split form's words, at a whole line.
        let lines = args.counters.ptr.address() + descriptor.fused_counters(tokens) * 4;

        let mut call = BoundaryCall {
            streams: address(&args.streams),
            output: address(&pending.output),
            post: address(&pending.post),
            comb: address(&pending.comb),
            written: target(&pending.streams),
            weight: address(&args.weight),
            scale: address(&args.scale),
            base: address(&args.base),
            pre: address(&args.pre),
            norm: address(&args.norm),
            input: target(&args.input),
            next_pre: target(&args.next_pre),
            next_post: target(&args.next_post),
            next_comb: target(&args.next_comb),
            partials: target(&args.partials),
            lines: lines as usize as *mut c_void,
            tokens: tokens as i64,
            stream_count: i64::from(descriptor.streams),
            hidden: i64::from(descriptor.hidden),
            iterations: i64::from(descriptor.iterations),
            splits: i64::from(descriptor.fused_splits(tokens)),
            multiprocessors: i64::from(stream.device().multiprocessors()?),
            rms_eps: args.rms_eps,
            eps: args.eps,
            norm_eps: args.norm_eps,
            error: [0; MESSAGE],
        };
        // SAFETY: `BoundaryCall` is the structure this package's entry declares, generated from
        // this definition with its layout asserted, and every address is an operand `check`
        // accepted at the extents the call states.
        unsafe { kernel.native(&mut call, stream) }
    }
}

impl Op for MhcBoundary {
    const NAME: &'static str = proto::module::operator::MHC_BOUNDARY;

    type Descriptor = BoundaryDescriptor;
    type Args<'a> = BoundaryArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let BoundaryDescriptor {
            dtype,
            streams,
            hidden,
            iterations,
            post,
            block,
            ..
        } = *descriptor;
        check_streams(Self::NAME, dtype, streams, hidden)?;
        if block == 0 || !hidden.is_multiple_of(block) {
            return Err(Error::argument(
                Self::NAME,
                format!("streams of {hidden} elements are not whole steps of {block} columns"),
            ));
        }
        if iterations == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a combination matrix of no Sinkhorn iteration".to_owned(),
            ));
        }

        args.streams.require_rank(Self::NAME, 3)?;
        let tokens = args.streams.dim(0);
        let (n, d, m) = (
            u64::from(streams),
            u64::from(hidden),
            u64::from(descriptor.mixes()),
        );
        contiguous(Self::NAME, "streams", &args.streams, dtype, &[tokens, n, d])?;

        match (&args.pending, post) {
            (Some(pending), true) => {
                contiguous(Self::NAME, "output", &pending.output, dtype, &[tokens, d])?;
                contiguous(Self::NAME, "post", &pending.post, DType::F32, &[tokens, n])?;
                contiguous(
                    Self::NAME,
                    "comb",
                    &pending.comb,
                    DType::F32,
                    &[tokens, n, n],
                )?;
                contiguous(
                    Self::NAME,
                    "written streams",
                    &pending.streams,
                    dtype,
                    &[tokens, n, d],
                )?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor writes a pending output back but none was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a pending output was given but the descriptor writes none back".to_owned(),
                ));
            }
        }

        contiguous(
            Self::NAME,
            "projection",
            &args.weight,
            DType::F32,
            &[m, n * d],
        )?;
        contiguous(Self::NAME, "scale", &args.scale, DType::F32, &[3])?;
        contiguous(Self::NAME, "base", &args.base, DType::F32, &[m])?;
        contiguous(Self::NAME, "pre-mix", &args.pre, DType::F32, &[tokens, n])?;
        contiguous(Self::NAME, "norm", &args.norm, dtype, &[d])?;
        contiguous(Self::NAME, "input", &args.input, dtype, &[tokens, d])?;
        contiguous(
            Self::NAME,
            "next pre-mix",
            &args.next_pre,
            DType::F32,
            &[tokens, n],
        )?;
        contiguous(
            Self::NAME,
            "next post-mix",
            &args.next_post,
            DType::F32,
            &[tokens, n],
        )?;
        contiguous(
            Self::NAME,
            "next combination",
            &args.next_comb,
            DType::F32,
            &[tokens, n, n],
        )?;

        contiguous(
            Self::NAME,
            "partials",
            &args.partials,
            DType::F32,
            &[descriptor.partial_floats(tokens)],
        )?;
        contiguous(
            Self::NAME,
            "arrival words",
            &args.counters,
            DType::I32,
            &[descriptor.counter_words(tokens)],
        )?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.streams.dim(0);
        // A native module is the fused form, with its call; a cubin the split form, with the
        // operator's grid.
        if !kernel.is_cubin() {
            return Self::launch_fused(kernel, descriptor, &args, stream);
        }

        let splits = descriptor.splits(tokens);
        // An operand the descriptor leaves unread is passed as the streams, which the kernel
        // never reads through it.
        let unread = &args.streams;
        let pending = args.pending.as_ref();

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("streams_ptr", &args.streams)
            .view(
                "output_ptr",
                pending.map_or(unread, |pending| &pending.output),
            )
            .view("post_ptr", pending.map_or(unread, |pending| &pending.post))
            .view("comb_ptr", pending.map_or(unread, |pending| &pending.comb))
            .view(
                "written_ptr",
                pending.map_or(unread, |pending| &pending.streams),
            )
            .view("weight_ptr", &args.weight)
            .view("scale_ptr", &args.scale)
            .view("base_ptr", &args.base)
            .view("pre_ptr", &args.pre)
            .view("norm_ptr", &args.norm)
            .view("input_ptr", &args.input)
            .view("next_pre_ptr", &args.next_pre)
            .view("next_post_ptr", &args.next_post)
            .view("next_comb_ptr", &args.next_comb)
            .view("partials_ptr", &args.partials)
            .view("counters_ptr", &args.counters)
            .u64("tokens", tokens)
            .u32("splits", splits)
            .f32("rms_eps", args.rms_eps)
            .f32("eps", args.eps)
            .f32("norm_eps", args.norm_eps);
        kernel.launch(
            LaunchGrid::new([descriptor.token_blocks(tokens), u64::from(splits), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        let pending = args.pending.map(|pending| Pending {
            streams: twins.shadow_of(pending.streams),
            ..pending
        });

        BoundaryArgs {
            pending,
            input: twins.shadow_of(args.input),
            next_pre: twins.shadow_of(args.next_pre),
            next_post: twins.shadow_of(args.next_post),
            next_comb: twins.shadow_of(args.next_comb),
            partials: twins.shadow_of(args.partials),
            // The split form's words are zero on entry and on exit, and the fused form tags its
            // lines with the identity of its grid, which no other launch's tags equal: a twin
            // reads the real words as the serving launch does.
            counters: twins.in_place(args.counters),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        boundary_tolerance(descriptor.dtype, descriptor.streams, descriptor.hidden)
    }
}

/// The bound of a boundary's outputs, relative to the largest magnitude of each.
///
/// The written streams are a sum of `n + 1` FP32 terms of mixed signs rounded once. The input is
/// the collapsed sum, rounded once (which two implementations may round to neighbouring values),
/// then normalized by a `d`-term FP32 mean square and rounded again. The mixes are an `n·d`-term
/// product of mixed signs in FP32 whose products cost the excess of the least exact product
/// either form computes: the split form's three-pass TF32 product ([`tolerance::tf32x3`]) or
/// the fused form's single pass, the 16-bit streams exact in TF32 and the projection read as
/// TF32 ([`tolerance::TF32_SINGLE`]), the larger. The scale, the sigmoids
/// (1/4-Lipschitz) and the Sinkhorn normalizations (each divides by a sum of positive terms,
/// which a relative error passes through, an approximate reciprocal adding a few units of FP32
/// to each) carry it to values below two; the bound is the largest of the three.
fn boundary_tolerance(dtype: DType, streams: u32, hidden: u32) -> Tolerance {
    let width = u64::from(streams) * u64::from(hidden);
    let written = Tolerance::rounding(u64::from(streams) + 1, DType::F32, dtype);
    let input = Tolerance::rounding(u64::from(hidden), DType::F32, dtype);
    let excess = tolerance::tf32x3(width).max(tolerance::TF32_SINGLE);
    let mixes = Tolerance::split_dot(width, excess, DType::F32, DType::F32);

    Tolerance {
        atol: tolerance::underflow(dtype) as f32,
        rtol: (written.rtol)
            .max(input.rtol + 2.0 * tolerance::unit_roundoff(dtype) as f32)
            .max(mixes.rtol),
        cancels: true,
    }
}

/// One compilation of `mhc_post`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PostDescriptor {
    /// Element type of the streams and the output.
    pub dtype: DType,
    /// Streams of the residual.
    pub streams: u32,
    /// Elements of one stream.
    pub hidden: u32,
    /// Columns of each stream one program writes.
    pub block: u32,
}

impl PostDescriptor {
    /// The compilation that writes outputs of `hidden` `dtype` elements back into `streams`
    /// streams.
    pub fn new(dtype: DType, streams: u32, hidden: u32) -> Self {
        Self {
            dtype,
            streams,
            hidden,
            block: row_tile(hidden).min(POST_COLUMNS),
        }
    }
}

impl fmt::Display for PostDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("streams", self.streams)
            .field("hidden", self.hidden)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `mhc_post`.
#[derive(Clone)]
pub struct PostArgs {
    /// `[tokens, streams, hidden]`, contiguous: the streams the output is written back into.
    pub streams: Operand,
    /// `[tokens, hidden]`, contiguous: the sub-layer's output.
    pub output: Operand,
    /// `[tokens, streams]` FP32, contiguous: its post-mix.
    pub post: Operand,
    /// `[tokens, streams, streams]` FP32, contiguous: its combination matrix.
    pub comb: Operand,
    /// `[tokens, streams, hidden]`, contiguous: receives the written streams.
    pub out: Operand,
}

/// A sub-layer's output written back into the streams:
/// `h'_k = round(post_k · y + Σ_j comb[j, k] · h_j)`, the sum over the streams first, in FP32.
pub struct MhcPost;

impl Op for MhcPost {
    const NAME: &'static str = proto::module::operator::MHC_POST;

    type Descriptor = PostDescriptor;
    type Args<'a> = PostArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let PostDescriptor {
            dtype,
            streams,
            hidden,
            block,
        } = *descriptor;
        check_streams(Self::NAME, dtype, streams, hidden)?;
        if !block.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!("a step of {block} columns is not a power of two"),
            ));
        }

        args.streams.require_rank(Self::NAME, 3)?;
        let tokens = args.streams.dim(0);
        let (n, d) = (u64::from(streams), u64::from(hidden));
        contiguous(Self::NAME, "streams", &args.streams, dtype, &[tokens, n, d])?;
        contiguous(Self::NAME, "output", &args.output, dtype, &[tokens, d])?;
        contiguous(Self::NAME, "post", &args.post, DType::F32, &[tokens, n])?;
        contiguous(Self::NAME, "comb", &args.comb, DType::F32, &[tokens, n, n])?;
        contiguous(
            Self::NAME,
            "written streams",
            &args.out,
            dtype,
            &[tokens, n, d],
        )
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.streams.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("streams_ptr", &args.streams)
            .view("output_ptr", &args.output)
            .view("post_ptr", &args.post)
            .view("comb_ptr", &args.comb)
            .view("out_ptr", &args.out);
        kernel.launch(
            LaunchGrid::new([
                tokens,
                u64::from(descriptor.hidden).div_ceil(u64::from(descriptor.block)),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        PostArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // `n + 1` FP32 terms of mixed signs, rounded once into the streams' type.
        let sum = Tolerance::rounding(
            u64::from(descriptor.streams) + 1,
            DType::F32,
            descriptor.dtype,
        );
        Tolerance {
            cancels: true,
            ..sum
        }
    }
}

/// One compilation of `mhc_pre`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PreDescriptor {
    /// Element type of the streams, the gain and the input.
    pub dtype: DType,
    /// Streams of the residual.
    pub streams: u32,
    /// Elements of one stream.
    pub hidden: u32,
    /// The lanes one program loads: `hidden` rounded up to a power of two.
    pub tile: u32,
}

impl PreDescriptor {
    /// The compilation that reads `streams` streams of `hidden` `dtype` elements into one
    /// normalized row.
    pub fn new(dtype: DType, streams: u32, hidden: u32) -> Self {
        Self {
            dtype,
            streams,
            hidden,
            tile: row_tile(hidden),
        }
    }
}

impl fmt::Display for PreDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("streams", self.streams)
            .field("hidden", self.hidden)
            .field("tile", self.tile)
            .finish(formatter)
    }
}

/// The arguments of `mhc_pre`.
#[derive(Clone)]
pub struct PreArgs {
    /// `[tokens, streams, hidden]`, contiguous.
    pub streams: Operand,
    /// `[tokens, streams]` FP32, contiguous: the pre-mix.
    pub pre: Operand,
    /// `[hidden]`: the normalization's gain.
    pub norm: Operand,
    /// `[tokens, hidden]`, contiguous: receives the normalized row.
    pub out: Operand,
    /// The epsilon added to the mean square.
    pub eps: f32,
}

/// The streams read into one row and normalized:
/// `x = rms_norm(round(Σ_j pre_j · h_j)) · w`, the sum in FP32 and rounded once before the
/// normalization reads it.
pub struct MhcPre;

impl Op for MhcPre {
    const NAME: &'static str = proto::module::operator::MHC_PRE;

    type Descriptor = PreDescriptor;
    type Args<'a> = PreArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let PreDescriptor {
            dtype,
            streams,
            hidden,
            tile,
        } = *descriptor;
        check_streams(Self::NAME, dtype, streams, hidden)?;
        if !tile.is_power_of_two() || tile < hidden {
            return Err(Error::argument(
                Self::NAME,
                format!("a tile of {tile} lanes does not cover a stream of {hidden} elements"),
            ));
        }

        args.streams.require_rank(Self::NAME, 3)?;
        let tokens = args.streams.dim(0);
        let (n, d) = (u64::from(streams), u64::from(hidden));
        contiguous(Self::NAME, "streams", &args.streams, dtype, &[tokens, n, d])?;
        contiguous(Self::NAME, "pre-mix", &args.pre, DType::F32, &[tokens, n])?;
        contiguous(Self::NAME, "norm", &args.norm, dtype, &[d])?;
        contiguous(Self::NAME, "input", &args.out, dtype, &[tokens, d])
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.streams.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("streams_ptr", &args.streams)
            .view("pre_ptr", &args.pre)
            .view("norm_ptr", &args.norm)
            .view("out_ptr", &args.out)
            .f32("eps", args.eps);
        kernel.launch(LaunchGrid::linear(tokens)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        PreArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The collapsed sum rounds once (to neighbouring values in two implementations), then the
        // `d`-term mean square normalizes it before the second rounding.
        let norm = Tolerance::rounding(u64::from(descriptor.hidden), DType::F32, descriptor.dtype);
        Tolerance {
            rtol: norm.rtol + 2.0 * tolerance::unit_roundoff(descriptor.dtype) as f32,
            cancels: true,
            ..norm
        }
    }
}

#[cfg(test)]
mod tests {
    use platform::DevicePtr;

    use super::*;

    fn view(dtype: DType, dims: &[u64]) -> Operand {
        Operand::new(DevicePtr::from_address(0x1000), dtype, dims)
    }

    /// The model's connection over `tokens` tokens: four streams of 5120 BF16 elements, twenty
    /// iterations.
    fn boundary(post: bool, tokens: u64) -> BoundaryDescriptor {
        BoundaryDescriptor::new(DType::BF16, 4, 5120, 20, post, tokens)
    }

    fn args(descriptor: &BoundaryDescriptor, tokens: u64) -> BoundaryArgs {
        let (n, d, m) = (4, 5120, u64::from(descriptor.mixes()));

        BoundaryArgs {
            streams: view(DType::BF16, &[tokens, n, d]),
            pending: descriptor.post.then(|| Pending {
                output: view(DType::BF16, &[tokens, d]),
                post: view(DType::F32, &[tokens, n]),
                comb: view(DType::F32, &[tokens, n, n]),
                streams: view(DType::BF16, &[tokens, n, d]),
            }),
            weight: view(DType::F32, &[m, n * d]),
            scale: view(DType::F32, &[3]),
            base: view(DType::F32, &[m]),
            pre: view(DType::F32, &[tokens, n]),
            norm: view(DType::BF16, &[d]),
            input: view(DType::BF16, &[tokens, d]),
            next_pre: view(DType::F32, &[tokens, n]),
            next_post: view(DType::F32, &[tokens, n]),
            next_comb: view(DType::F32, &[tokens, n, n]),
            partials: view(DType::F32, &[descriptor.partial_floats(tokens)]),
            counters: view(DType::I32, &[descriptor.counter_words(tokens)]),
            rms_eps: 1e-20,
            eps: 1e-6,
            norm_eps: 1e-20,
        }
    }

    #[test]
    fn the_key_names_every_compiled_value() {
        assert_eq!(
            boundary(true, 1).to_string(),
            "dtype_bf16_streams_4_hidden_5120_iterations_20_post_true_tokens_16_block_64"
        );
        assert_eq!(
            boundary(false, 4096).to_string(),
            "dtype_bf16_streams_4_hidden_5120_iterations_20_post_false_tokens_64_block_64"
        );
        assert_eq!(
            PostDescriptor::new(DType::BF16, 4, 5120).to_string(),
            "dtype_bf16_streams_4_hidden_5120_block_1024"
        );
        assert_eq!(
            PreDescriptor::new(DType::BF16, 4, 5120).to_string(),
            "dtype_bf16_streams_4_hidden_5120_tile_8192"
        );
    }

    #[test]
    fn few_tokens_split_their_columns_and_many_split_them_less() {
        let splits = |tokens| boundary(false, tokens).splits(tokens);

        // One block of a decode's tokens takes one program per step of 64 columns.
        assert_eq!(splits(1), 80);
        assert_eq!(splits(16), 80);
        // Four blocks share the 256 programs the small blocks aim at.
        assert_eq!(splits(64), 64);
        assert_eq!(splits(512), 8);
        // A prefill's blocks of 64 tokens aim at 1024 programs.
        assert_eq!(splits(4096), 16);
        assert_eq!(splits(16384), 4);
    }

    /// The fused form splits a block's 80 column steps into the fewest equal shares that bring
    /// its blocks' tasks to 148 programs, between 16 and 64 (DeepGEMM's `get_num_splits`).
    #[test]
    fn the_fused_form_splits_few_blocks_more() {
        let splits = |tokens| boundary(true, tokens).fused_splits(tokens);

        // Up to three blocks: two shares of 40 steps.
        assert_eq!(splits(1), 40);
        assert_eq!(splits(192), 40);
        // Four and five blocks: three shares; six and seven: four; then five shares of 16.
        assert_eq!(splits(193), 27);
        assert_eq!(splits(320), 27);
        assert_eq!(splits(384), 20);
        assert_eq!(splits(448), 20);
        assert_eq!(splits(512), 16);
        assert_eq!(splits(16384), 16);
    }

    /// The workspace holds either form's partials, and the fused form's arrival lines follow the
    /// split form's words at a whole 128-byte line.
    #[test]
    fn the_workspace_holds_either_forms_partials_and_arrivals() {
        let decode = boundary(true, 1);
        // The fused form's 40 splits of a block of 64 tokens' 26 values, more than the split
        // form's 80 splits of one token's 34.
        assert_eq!(decode.partial_floats(1), 40 * 64 * 26);
        assert_eq!(decode.fused_counters(1), 32);
        assert_eq!(decode.counter_words(1), 32 + 2 * 32);

        let prefill = boundary(true, 4096);
        // The split form's 16 splits of every token's 34 values, more than the fused form's 16
        // splits of 64 blocks of 64 tokens' 26.
        assert_eq!(prefill.partial_floats(4096), 16 * 4096 * 34);
        assert_eq!(prefill.fused_counters(4096), 64);
        assert_eq!(prefill.counter_words(4096), 64 + 64 * 2 * 32);

        for tokens in [1, 16, 64, 4096] {
            let descriptor = boundary(true, tokens);
            MhcBoundary::check(&descriptor, &args(&descriptor, tokens)).unwrap();
        }
        let mut short = args(&decode, 1);
        short.counters = view(DType::I32, &[decode.token_blocks(1)]);
        assert!(MhcBoundary::check(&decode, &short).is_err());
    }

    #[test]
    fn a_pending_output_is_given_exactly_when_the_descriptor_writes_one_back() {
        let written = boundary(true, 8);
        let read = boundary(false, 8);

        assert!(MhcBoundary::check(&written, &args(&read, 8)).is_err());
        assert!(MhcBoundary::check(&read, &args(&written, 8)).is_err());
    }

    #[test]
    fn a_stream_count_that_is_not_a_power_of_two_is_refused() {
        let descriptor = PreDescriptor::new(DType::BF16, 3, 64);
        let args = PreArgs {
            streams: view(DType::BF16, &[2, 3, 64]),
            pre: view(DType::F32, &[2, 3]),
            norm: view(DType::BF16, &[64]),
            out: view(DType::BF16, &[2, 64]),
            eps: 1e-6,
        };
        assert!(MhcPre::check(&descriptor, &args).is_err());
    }
}
