//! Rotation alone: the vision tower's queries and keys, from tables the host prepared.
//!
//! A text decoder fuses the per-head RMS norm, the rotation and the store into one program
//! ([`crate::layers::Rotary`]) because it has a weight per head and a position per token. A vision
//! tower's positions are an image's patch grid, cheap enough to evaluate on the host once per image,
//! and its heads carry no learned norm — so it needs only the rotation, applied to tables it has in
//! hand: `ops::rope::RopeApply`.
//!
//! The tables are `[tokens, rotated / 2]` FP32 (cosine and sine of each token's phase); the values
//! are `[tokens, heads, head_dim]` and the result is the same shape, rotated and stored in the
//! output's element type. No frequencies, no sections and no scale live here: a sectioned rotation
//! is [`crate::layers::Rotary`]'s, and this layer never guesses an axis cut - a model whose heads
//! turn only their leading channels says so with [`Rotate::rotated`].

use ops::TensorView;
use ops::rope::{RopeApply, RopeApplyArgs, RopeApplyDescriptor};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Rotation;
use crate::layers::launch::PDL;

/// Tokens one rotation program handles.
pub const ROWS: u32 = 8;

/// Head elements one program loads at a time.
pub const BLOCK: u32 = 128;

/// The rotation of a vision tower's queries or keys.
#[derive(Clone, Copy)]
pub struct Rotate {
    head_dim: u32,
    rotated: u32,
    heads: u32,
    rotation: Rotation,
}

impl Rotate {
    /// The rotation of `head_dim`-element heads, `heads` of them per token, paired as `rotation`.
    pub fn new(head_dim: u64, heads: u64, rotation: Rotation) -> Result<Self> {
        let narrow = |name: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| {
                Error::shape(
                    "Rotate",
                    format!("{name} {value} is wider than a descriptor holds"),
                )
            })
        };
        let (head_dim, heads) = (narrow("head_dim", head_dim)?, narrow("heads", heads)?);
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(Error::shape(
                "Rotate",
                format!("a head of {head_dim} elements has no rotation pair"),
            ));
        }
        if heads == 0 {
            return Err(Error::shape(
                "Rotate",
                "a token has at least one head".to_owned(),
            ));
        }
        Ok(Self {
            head_dim,
            rotated: head_dim,
            heads,
            rotation,
        })
    }

    /// The descriptor one rotation launch takes, for a walk that asks a model which programs it
    /// launches with no device: the geometry is the module's and the element types are the call's.
    pub fn descriptor(&self, input: ops::DType, output: ops::DType) -> RopeApplyDescriptor {
        RopeApplyDescriptor {
            dtype: input,
            output,
            head_dim: self.head_dim,
            heads: self.heads,
            rotation: self.rotation,
            rotated: self.rotated,
            rows: ROWS,
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// Turn only the leading `rotated` channels of each head, copying the rest unchanged.
    ///
    /// The default is the whole head. A model whose heads interleave a rotation of a few axes with
    /// channels that must keep their value states that cut here - the descriptor has carried the
    /// count all along, and this is the layer's way of saying it instead of pinning it to
    /// `head_dim`.
    ///
    /// The count must be positive, even, and no wider than the head; the tables are then
    /// `[tokens, rotated / 2]`. Each count is a separate compilation, so one the store does not
    /// carry yet is a load-time gap rather than a wrong result.
    pub fn rotated(mut self, rotated: u32) -> Result<Self> {
        if rotated == 0 || !rotated.is_multiple_of(2) || rotated > self.head_dim {
            return Err(Error::shape(
                "Rotate",
                format!(
                    "a rotation of {rotated} of {} channels is not a positive even count within the head",
                    self.head_dim
                ),
            ));
        }
        self.rotated = rotated;
        Ok(self)
    }

    /// The width of one head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// The heads one token carries.
    pub const fn heads(&self) -> u32 {
        self.heads
    }

    /// Rotate `x` `[tokens, heads, head_dim]` into `out` by the `cosine` and `sine` tables.
    ///
    /// Both tables are `[tokens, rotated / 2]` FP32 and were evaluated by the caller; the leading
    /// `rotated` channels turn and the rest are copied.
    pub fn apply(
        &self,
        f: &Forward,
        x: TensorView,
        cosine: TensorView,
        sine: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let descriptor = self.descriptor(x.dtype, out.dtype);
        f.operate::<RopeApply>(
            &descriptor,
            RopeApplyArgs {
                x,
                cosine,
                sine,
                out,
            },
        )
    }
}
