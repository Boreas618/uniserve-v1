//! Rotation alone: the vision tower's queries and keys, from tables the host prepared.
//!
//! A text decoder fuses the per-head RMS norm, the rotation and the store into one program
//! ([`crate::layers::Rotary`]) because it has a weight per head and a position per token. A vision
//! tower's positions are an image's patch grid, cheap enough to evaluate on the host once per image,
//! and its heads carry no learned norm — so it needs only the rotation, applied to tables it has in
//! hand: `ops::rope::RopeApply`.
//!
//! The tables are `[tokens, head_dim / 2]` FP32 (cosine and sine of each token's phase); the values
//! are `[tokens, heads, head_dim]` and the result is the same shape, rotated and stored in the
//! output's element type. No frequencies, no sections and no scale live here: a sectioned rotation
//! is [`crate::layers::Rotary`]'s, and this layer never guesses an axis cut.

use ops::TensorView;
use ops::rope::{RopeApply, RopeApplyArgs, RopeApplyDescriptor};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Rotation;
use crate::layers::launch::PDL;

/// Tokens one rotation program handles.
const ROWS: u32 = 8;

/// Head elements one program loads at a time.
const BLOCK: u32 = 128;

/// The rotation of a vision tower's queries or keys.
#[derive(Clone, Copy)]
pub struct Rotate {
    head_dim: u32,
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
            heads,
            rotation,
        })
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
    /// Both tables are `[tokens, head_dim / 2]` FP32 and were evaluated by the caller; the whole
    /// head turns, because a tower that wanted part of it would say which cut it means and only
    /// [`crate::layers::Rotary`] states that.
    pub fn apply(
        &self,
        f: &Forward,
        x: TensorView,
        cosine: TensorView,
        sine: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let descriptor = RopeApplyDescriptor {
            dtype: x.dtype,
            output: out.dtype,
            head_dim: self.head_dim,
            heads: self.heads,
            rotation: self.rotation,
            rotated: self.head_dim,
            rows: ROWS,
            block: BLOCK,
            pdl: PDL,
        };
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
