//! The operators of a linear-attention layer that keeps a recurrent state per request slot: the
//! short causal convolution and the gated delta rule.
//!
//! A gated delta network (Yang et al., "Gated Delta Networks", 2024; Qwen3-Next and Qwen3.5) mixes
//! tokens through a depthwise causal convolution of a few taps and a matrix-valued state updated by
//! the delta rule under a scalar decay per value head:
//!
//! ```text
//! causal_conv1d      x'[t, c] = silu(Σ_j w[c, j] · x[t − (K − 1) + j, c])        j < K
//! gated_delta_rule   g = −exp(A_log) · softplus(a + dt_bias),  β = sigmoid(b)
//!                    q̂ = l2norm(q) / √d_k,  k̂ = l2norm(k)
//!                    S ← exp(g) · S;  S ← S + k̂ ⊗ β (v − k̂ᵀS);  o = q̂ᵀS
//! ```
//!
//! ## Rows, slots and checkpoints
//!
//! Both operators read a batch the way paged attention does: `query_offsets [rows + 1]` packs each
//! row's new tokens and `seq_lens [rows]` counts every token the row has seen, so a row's new tokens
//! sit at absolute positions `seq_lens[r] − (offsets[r + 1] − offsets[r]) ..  seq_lens[r]`. A row
//! that starts at position zero begins from a zero state; any other row continues from the state its
//! request slot holds, `slots[r]` naming the slot of the `[slots, …]` banks. A row whose slot is
//! `u32::MAX` is padding: it reads and writes no state and writes zeros into its output rows, so a
//! captured bucket wider than its batch carries finite values.
//!
//! A row also names `checkpoints[r]`, an absolute token count at which its state is kept for the
//! prefix cache: when `start < checkpoints[r] ≤ end`, the state after the token at
//! `checkpoints[r] − 1` is written into the row's slot of the checkpoint banks, and zero names no
//! checkpoint. The live banks always end at the row's last token. The infrastructure that owns the
//! banks moves a checkpoint into the cache pool afterwards (`cache_block_snapshot`,
//! `cache_block_scatter`); neither operator knows the pool.
//!
//! ## Trails
//!
//! A batch whose rows' tokens may not all stand - a speculative verification, which learns how
//! many of a row's drafted tokens it accepts only after the forward - asks either operator for a
//! [`Trail`]: the state after **every** token, row `t` of a `[tokens, …]` bank for the batch's
//! token `t`. The infrastructure then installs the state after the last accepted token into the
//! slot. Only the recurrent walk of the delta rule keeps one; the chunked form solves a chunk at
//! once and holds no state between its tokens.
//!
//! ## The two forms of the delta rule
//!
//! [`Form::Recurrent`] walks a row's tokens one at a time, which is the decode step's shape.
//! [`Form::Chunked`] processes a row in chunks of [`GatedDeltaRuleDescriptor::chunk`] tokens aligned
//! to absolute positions (the first chunk ends at the next multiple of the chunk), solving each
//! chunk's delta rule in its WY representation; a checkpoint must therefore fall on a chunk boundary.
//! Both forms keep the state and every intermediate in FP32 and round once into the output type.

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{Operand, Shape, Strides};
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The batch columns a recurrent operator reads, as paged attention reads them.
#[derive(Clone, Copy)]
pub struct RowColumns {
    /// `[rows + 1]` u32: packed offsets of each row's new tokens.
    pub query_offsets: Operand,
    /// `[rows]` u32: tokens each row has seen, its new ones included.
    pub seq_lens: Operand,
    /// `[rows]` u32: each row's slot in the state banks; `u32::MAX` for a padding row.
    pub slots: Operand,
    /// `[rows]` u32: the absolute token count each row keeps a checkpoint at, zero for none.
    pub checkpoints: Operand,
}

impl RowColumns {
    /// The rows of the batch, checked against each other.
    fn check(&self, operator: &'static str) -> Result<u64> {
        self.seq_lens.require_rank(operator, 1)?;
        let rows = self.seq_lens.shape.dim(0);
        for (name, view, count) in [
            ("query_offsets", &self.query_offsets, rows + 1),
            ("seq_lens", &self.seq_lens, rows),
            ("slots", &self.slots, rows),
            ("checkpoints", &self.checkpoints, rows),
        ] {
            view.require_shape(operator, Shape::new(&[count]))
                .map_err(|error| Error::argument(operator, format!("{name}: {error}")))?;
            view.require_row(operator, DType::U32, count)
                .map_err(|error| Error::argument(operator, format!("{name}: {error}")))?;
        }
        if rows == 0 {
            return Err(Error::argument(
                operator,
                "a batch of no rows has no state to advance".to_owned(),
            ));
        }
        Ok(rows)
    }
}

/// The per-token state bank a batch's rows keep beside their slots' banks (the module's Trails):
/// `[rows, dims…]` of the slot banks' element type, contiguous, row `t` receiving the state after
/// the batch's token `t`.
#[derive(Clone, Copy)]
pub struct Trail {
    /// `[rows, dims…]`, contiguous, with a row for every token of the batch.
    pub bank: Operand,
}

/// The trail operand of a launch and the flag the kernel tests: the bank, or the state bank itself
/// and a cleared flag when the batch keeps none, which the kernel then never writes.
fn trail_operand(trail: Option<Trail>, state: Operand) -> (Operand, i32) {
    match trail {
        Some(trail) => (trail.bank, 1),
        None => (state, 0),
    }
}

/// Check that `trail`, when the batch keeps one, is a contiguous `[rows, dims…]` bank of `dtype`
/// with a row for every one of the batch's `tokens`.
fn check_trail(
    operator: &'static str,
    trail: Option<&Trail>,
    dtype: DType,
    dims: &[u64],
    tokens: u64,
) -> Result<()> {
    let Some(trail) = trail else {
        return Ok(());
    };
    let rows = check_bank(operator, "trail", &trail.bank, dtype, dims)?;
    if rows < tokens {
        return Err(Error::argument(
            operator,
            format!("a trail of {rows} rows for a batch of {tokens} tokens"),
        ));
    }
    Ok(())
}

/// Check that `bank` is `[slots, dims…]` of `dtype`, contiguous, and answer its slot count.
fn check_bank(
    operator: &'static str,
    name: &str,
    bank: &Operand,
    dtype: DType,
    dims: &[u64],
) -> Result<u64> {
    let refuse = |problem: String| Error::argument(operator, format!("the {name} bank {problem}"));
    if bank.dtype != dtype {
        return Err(refuse(format!(
            "holds {}, not {}",
            abi::name(bank.dtype),
            abi::name(dtype)
        )));
    }
    if bank.shape.rank() != dims.len() + 1 || bank.shape.dims()[1..] != *dims {
        return Err(refuse(format!(
            "is {:?}, [slots, {dims:?}] expected",
            bank.shape.dims()
        )));
    }
    if bank.strides != Strides::row_major(bank.shape) {
        return Err(refuse("is not contiguous".to_owned()));
    }
    Ok(bank.shape.dim(0))
}

/// One compilation of `causal_conv1d`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CausalConv1dDescriptor {
    /// Element type of the input, the taps, the output and the state banks.
    pub dtype: DType,
    /// Channels the convolution mixes, each through its own taps.
    pub channels: u32,
    /// Taps per channel, `K`; the state keeps the last `K − 1` inputs.
    pub width: u32,
    /// Channels one program covers.
    pub block: u32,
}

impl fmt::Display for CausalConv1dDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("channels", self.channels)
            .field("width", self.width)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `causal_conv1d`.
#[derive(Clone, Copy)]
pub struct CausalConv1dArgs {
    /// `[tokens, channels]`, unit-strided per channel, rows any stride apart: the projected inputs.
    pub x: Operand,
    /// `[channels, width]`, contiguous: each channel's taps, the oldest input's first.
    pub weight: Operand,
    /// `[tokens, channels]`, contiguous: receives the activated convolution.
    pub out: Operand,
    /// The batch's rows.
    pub rows: RowColumns,
    /// `[slots, width − 1, channels]`, contiguous: each slot's last inputs, oldest first; read for a
    /// row that continues and overwritten with the row's last inputs.
    pub state: Operand,
    /// `[slots, width − 1, channels]`, contiguous: receives a row's inputs before its checkpoint.
    pub checkpoint: Operand,
    /// `[tokens, width − 1, channels]`: receives, for each token, the inputs up to and including
    /// it, when the batch keeps a trail.
    pub trail: Option<Trail>,
}

/// The depthwise causal convolution of a linear-attention layer, with its SiLU, over a batch of
/// rows that continue from their slots' last inputs.
///
/// One program covers [`CausalConv1dDescriptor::block`] channels of one row and walks its tokens;
/// the taps and the sum are FP32 and the activated value rounds once. The state is written after
/// every read of it, so a row's own state view may be its bank's.
pub struct CausalConv1d;

impl Op for CausalConv1d {
    const NAME: &'static str = proto::module::operator::CAUSAL_CONV1D;

    type Descriptor = CausalConv1dDescriptor;
    type Args<'a> = CausalConv1dArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.width < 2 || descriptor.channels == 0 || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a convolution of {} taps over {} channels in blocks of {} keeps no state",
                    descriptor.width, descriptor.channels, descriptor.block
                ),
            ));
        }
        let channels = u64::from(descriptor.channels);
        let width = u64::from(descriptor.width);
        args.rows.check(Self::NAME)?;
        args.x.require_rank(Self::NAME, 2)?;
        args.x.require_row(Self::NAME, descriptor.dtype, channels)?;
        let tokens = args.x.shape.dim(0);
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, channels]))?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, channels)?;
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the output is not contiguous".to_owned(),
            ));
        }
        args.weight
            .require_shape(Self::NAME, Shape::new(&[channels, width]))?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, width)?;
        let dims = [width - 1, channels];
        let slots = check_bank(Self::NAME, "state", &args.state, descriptor.dtype, &dims)?;
        let kept = check_bank(
            Self::NAME,
            "checkpoint",
            &args.checkpoint,
            descriptor.dtype,
            &dims,
        )?;
        if kept != slots {
            return Err(Error::argument(
                Self::NAME,
                format!("{kept} checkpoint slots for {slots} state slots"),
            ));
        }
        check_trail(
            Self::NAME,
            args.trail.as_ref(),
            descriptor.dtype,
            &dims,
            tokens,
        )?;
        if args.x.strides.get(0) > i32::MAX as u64 || tokens * channels > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                "the inputs span more elements than the kernel's 32-bit rows address".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.rows.seq_lens.shape.dim(0);
        let (trail, trailing) = trail_operand(args.trail, args.state);

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.weight)
            .view(args.out)
            .view(args.rows.query_offsets)
            .view(args.rows.seq_lens)
            .view(args.rows.slots)
            .view(args.rows.checkpoints)
            .view(args.state)
            .view(args.checkpoint)
            .view(trail)
            .i32(args.x.strides.get(0) as i32)
            .i32(trailing);
        kernel.launch(
            LaunchGrid::new([
                u64::from(descriptor.channels).div_ceil(u64::from(descriptor.block)),
                rows,
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The state banks are read and rewritten in place, so the shadow runs on copies of them.
        CausalConv1dArgs {
            out: twins.shadow_of(args.out),
            state: twins.in_place(args.state, stream),
            checkpoint: twins.in_place(args.checkpoint, stream),
            trail: args.trail.map(|trail| Trail {
                bank: twins.shadow_of(trail.bank),
            }),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // A sum of `width` products of rounded inputs and taps in FP32, the SiLU's few ulps, and one
        // rounding into the output type.
        Tolerance::dot(u64::from(descriptor.width), DType::F32, descriptor.dtype)
            .with_atol(2.0 * tolerance::unit_roundoff(descriptor.dtype) as f32)
    }
}

/// How [`GatedDeltaRule`] walks a row's tokens.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Form {
    /// One token at a time: the decode step's form.
    Recurrent,
    /// Chunks of [`GatedDeltaRuleDescriptor::chunk`] tokens at absolute positions, each solved in
    /// its WY representation: the prompt's form.
    Chunked,
}

impl Form {
    /// The name a key prints.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recurrent => "recurrent",
            Self::Chunked => "chunked",
        }
    }
}

impl fmt::Display for Form {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One compilation of `gated_delta_rule`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatedDeltaRuleDescriptor {
    /// Element type of the queries, keys, values, gate inputs and the output.
    pub dtype: DType,
    /// Query and key heads; each serves `value_heads / key_heads` consecutive value heads.
    pub key_heads: u32,
    /// Value heads, each with its own state, decay and step size.
    pub value_heads: u32,
    /// Elements of a query or key head.
    pub key_dim: u32,
    /// Elements of a value head.
    pub value_dim: u32,
    /// The value columns one program owns of one head's state.
    pub block: u32,
    /// How the rows' tokens are walked.
    pub form: Form,
    /// Tokens of one chunk of [`Form::Chunked`]; one for [`Form::Recurrent`].
    pub chunk: u32,
}

impl fmt::Display for GatedDeltaRuleDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("key_heads", self.key_heads)
            .field("value_heads", self.value_heads)
            .field("key_dim", self.key_dim)
            .field("value_dim", self.value_dim)
            .field("block", self.block)
            .field("form", self.form)
            .field("chunk", self.chunk)
            .finish(formatter)
    }
}

/// The arguments of `gated_delta_rule`.
#[derive(Clone, Copy)]
pub struct GatedDeltaRuleArgs {
    /// `[tokens, key_heads, key_dim]`: the convolved queries, heads contiguous, rows `row_stride`
    /// elements apart.
    pub query: Operand,
    /// `[tokens, key_heads, key_dim]`: the convolved keys, rows as the queries'.
    pub key: Operand,
    /// `[tokens, value_heads, value_dim]`: the convolved values, rows as the queries'.
    pub value: Operand,
    /// `[tokens, value_heads]`, unit-strided per head: the decay's input projection.
    pub a: Operand,
    /// `[tokens, value_heads]`, unit-strided per head, rows as `a`'s: the step size's projection.
    pub b: Operand,
    /// `[value_heads]`: the log of each head's decay rate.
    pub a_log: Operand,
    /// `[value_heads]`: the bias of each head's time step.
    pub dt_bias: Operand,
    /// The batch's rows.
    pub rows: RowColumns,
    /// `[slots, value_heads, key_dim, value_dim]` FP32, contiguous: each slot's state, read for a
    /// row that continues and overwritten with the state after the row's last token.
    pub state: Operand,
    /// `[slots, value_heads, key_dim, value_dim]` FP32, contiguous: receives a row's checkpoint.
    pub checkpoint: Operand,
    /// `[tokens, value_heads, key_dim, value_dim]` FP32: receives the state after each token, when
    /// the batch keeps a trail; only [`Form::Recurrent`] keeps one.
    pub trail: Option<Trail>,
    /// `[tokens, value_heads, value_dim]`, contiguous: receives `q̂ᵀS` after each token.
    pub out: Operand,
}

/// The gated delta rule over a batch of rows that continue from their slots' states; see the module
/// documentation for the recurrence, the rows and the two forms.
///
/// One program owns [`GatedDeltaRuleDescriptor::block`] value columns of one value head of one row,
/// so the state it carries is `key_dim × block` FP32. The decay and the step size are computed from
/// `a`, `b`, `A_log` and `dt_bias` in FP32 (softplus with PyTorch's linear threshold of 20), and the
/// queries and keys are normalized in FP32 with an epsilon of `1e-6` under the square root.
pub struct GatedDeltaRule;

impl Op for GatedDeltaRule {
    const NAME: &'static str = proto::module::operator::GATED_DELTA_RULE;

    type Descriptor = GatedDeltaRuleDescriptor;
    type Args<'a> = GatedDeltaRuleArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.key_heads == 0
            || d.value_heads == 0
            || !d.value_heads.is_multiple_of(d.key_heads)
            || d.key_dim == 0
            || d.value_dim == 0
            || d.block == 0
            || !d.value_dim.is_multiple_of(d.block)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} value heads of {} over {} key heads of {}, in blocks of {}",
                    d.value_heads, d.value_dim, d.key_heads, d.key_dim, d.block
                ),
            ));
        }
        match (d.form, d.chunk) {
            (Form::Recurrent, 1) => {}
            (Form::Chunked, chunk) if chunk > 1 && chunk.is_power_of_two() => {}
            (form, chunk) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!("a {form} rule in chunks of {chunk} tokens"),
                ));
            }
        }
        args.rows.check(Self::NAME)?;
        let key_heads = u64::from(d.key_heads);
        let value_heads = u64::from(d.value_heads);
        let key_dim = u64::from(d.key_dim);
        let value_dim = u64::from(d.value_dim);

        args.query.require_rank(Self::NAME, 3)?;
        let tokens = args.query.shape.dim(0);
        let row_stride = args.query.strides.get(0);
        for (name, view, heads, dim) in [
            ("query", &args.query, key_heads, key_dim),
            ("key", &args.key, key_heads, key_dim),
            ("value", &args.value, value_heads, value_dim),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[tokens, heads, dim]))
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            view.require_row(Self::NAME, d.dtype, dim)
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            if view.strides.get(1) != dim || view.strides.get(0) != row_stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} heads are {:?} apart, [{row_stride}, {dim}, 1] expected",
                        view.strides
                    ),
                ));
            }
        }
        let gate_stride = args.a.strides.get(0);
        for (name, view) in [("a", &args.a), ("b", &args.b)] {
            view.require_shape(Self::NAME, Shape::new(&[tokens, value_heads]))
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            view.require_row(Self::NAME, d.dtype, value_heads)
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            if view.strides.get(0) != gate_stride {
                return Err(Error::argument(
                    Self::NAME,
                    "the a and b rows are not one stride apart".to_owned(),
                ));
            }
        }
        for (name, view) in [("A_log", &args.a_log), ("dt_bias", &args.dt_bias)] {
            view.require_shape(Self::NAME, Shape::new(&[value_heads]))
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            view.require_row(Self::NAME, d.dtype, value_heads)
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, value_heads, value_dim]))?;
        args.out.require_row(Self::NAME, d.dtype, value_dim)?;
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the output is not contiguous".to_owned(),
            ));
        }
        let dims = [value_heads, key_dim, value_dim];
        let slots = check_bank(Self::NAME, "state", &args.state, DType::F32, &dims)?;
        let kept = check_bank(
            Self::NAME,
            "checkpoint",
            &args.checkpoint,
            DType::F32,
            &dims,
        )?;
        if kept != slots {
            return Err(Error::argument(
                Self::NAME,
                format!("{kept} checkpoint slots for {slots} state slots"),
            ));
        }
        if args.trail.is_some() && d.form != Form::Recurrent {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a {} rule holds no state between its tokens, so it keeps no trail",
                    d.form
                ),
            ));
        }
        check_trail(Self::NAME, args.trail.as_ref(), DType::F32, &dims, tokens)?;
        if row_stride * tokens > i32::MAX as u64 || gate_stride * tokens > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                "the inputs span more elements than the kernel's 32-bit rows address".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.rows.seq_lens.shape.dim(0);
        // The query is scaled by the inverse square root of its width before it reads the state.
        let scale = (f64::from(descriptor.key_dim).sqrt().recip()) as f32;
        let (trail, trailing) = trail_operand(args.trail, args.state);

        let mut arguments = Arguments::new();
        arguments
            .view(args.query)
            .view(args.key)
            .view(args.value)
            .view(args.a)
            .view(args.b)
            .view(args.a_log)
            .view(args.dt_bias)
            .view(args.rows.query_offsets)
            .view(args.rows.seq_lens)
            .view(args.rows.slots)
            .view(args.rows.checkpoints)
            .view(args.state)
            .view(args.checkpoint)
            .view(trail)
            .view(args.out)
            .i32(args.query.strides.get(0) as i32)
            .i32(args.a.strides.get(0) as i32)
            .f32(scale)
            .i32(trailing);
        kernel.launch(
            LaunchGrid::new([
                u64::from(descriptor.value_dim / descriptor.block),
                u64::from(descriptor.value_heads),
                rows,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The state banks are read and rewritten in place, so the shadow runs on copies of them.
        GatedDeltaRuleArgs {
            out: twins.shadow_of(args.out),
            state: twins.in_place(args.state, stream),
            checkpoint: twins.in_place(args.checkpoint, stream),
            trail: args.trail.map(|trail| Trail {
                bank: twins.shadow_of(trail.bank),
            }),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Each output is a `key_dim`-term dot product of the normalized query with a state that is
        // itself a decayed sum of rank-one updates; two implementations differ by their FP32
        // association of those sums and by the one rounding into the output type. The update's
        // inner product `k̂ᵀS` has the same length, so the bound is that of a dot product over the
        // key width, applied twice.
        let dot = Tolerance::dot(u64::from(descriptor.key_dim), DType::F32, descriptor.dtype);
        Tolerance {
            rtol: 2.0 * dot.rtol,
            ..dot
        }
    }
}
