//! Per-channel bias addition, optionally onto a residual, channel-first.
//!
//! A convolution's bias is added in a separate pass in the reference, not folded into the
//! convolution, and H3's video encoder and decoder rely on that: a biased convolution must equal
//! `conv(x, weight) + bias` bit for bit, including the residual shortcut both passes receive.
//!
//! ```text
//! out = (residual + residual_bias) + (values + bias)
//! ```
//!
//! The operands are NCDHW — or `[batch, channels, spatial]` with the spatial axes merged, which is
//! the same storage read as three axes — and the result is contiguous and channel-first whichever
//! storage the inputs use (`ops::bias::BiasAdd`). Both biases are `[channels]`. A residual without
//! its own bias is the common case; a residual bias without a residual is refused, because there
//! would be nothing for it to bias.

/// Elements one bias program adds.
pub const BLOCK: u32 = 1024;

/// A per-channel bias, and the residual it is added to.
#[derive(Clone, Copy, Default)]
pub struct BiasAdd;

impl BiasAdd {}
