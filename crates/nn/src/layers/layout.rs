//! Layout moves: elements carried between views without arithmetic.
//!
//! ```text
//! Copy                 out[c, f, h, w] = source[c, f, h, w]        each side through its own strides
//! Patchify::forward    tokens[(f·h'+i)·w'+j, ((c·t+a)·s+b)·s+d] = latent[c, f·t+a, i·s+b, j·s+d]
//! Patchify::inverse    the same index map, from tokens back into a raster
//! ```
//!
//! Every move is exact: the same bits arrive on the other side. The extents and strides are launch
//! values, so one compiled program serves every geometry a request can have.

use ops::copy_block::{self, CopyBlock, RANK};
use ops::patch::{self, Patch};
use ops::unpatchify::{self, Unpatchify};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// How a [`Copy`] program moves its elements.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    /// One element per lane, any strides.
    Element,
    /// Whole vectors per lane: rows unit-stride and aligned on both sides, which the operator checks.
    Vector,
}

/// A four-axis block of one element type moved between two views of different layout.
///
/// The permutation, if any, lives in the strides: a permuted copy is the source's strides in the
/// destination's axis order. A model with a batch axis folds it into the first.
#[derive(Clone, Copy, Debug)]
pub struct Copy {
    dtype: DType,
    access: Access,
}

impl Copy {
    /// A copy of four-axis blocks of `dtype`, moved with `access`.
    ///
    /// # Errors
    ///
    /// A sub-byte element type, which a copy cannot address element by element.
    pub fn new(dtype: DType, access: Access) -> Result<Self> {
        if dtype.bytes().is_none() {
            return Err(Error::shape(
                "Copy",
                format!("{dtype} elements are sub-byte; a block copy moves whole elements"),
            ));
        }
        Ok(Self { dtype, access })
    }

    /// `destination[c, f, h, w] = source[c, f, h, w]` under each side's own strides.
    ///
    /// The destination's strides are passed rather than read off its view: a caller may address a
    /// region of a larger raster whose view it built row-major, and the strides say where each
    /// element lands. Both views are four-axis and hold the same extents.
    ///
    /// # Errors
    ///
    /// A view of another rank, or the operator's refusal (extents, strides, alignment).
    pub fn forward(
        &self,
        f: &Forward,
        source: TensorView,
        destination: TensorView,
        destination_strides: &[i64; RANK],
    ) -> Result<()> {
        for (name, view) in [("source", source), ("destination", destination)] {
            if view.shape.rank() != RANK {
                return Err(Error::shape(
                    "Copy",
                    format!(
                        "the {name} view has {} axes; a block copy moves four and a batch axis is \
                         folded into the first",
                        view.shape.rank()
                    ),
                ));
            }
        }
        let descriptor = copy_block::Descriptor {
            source: self.dtype,
            out: self.dtype,
            wide: self.access == Access::Vector,
        };
        let args = copy_block::Args {
            source,
            out: destination,
            out_strides: destination_strides,
        };
        f.operate::<CopyBlock>(&descriptor, args)
    }
}

/// The elements one program of the inverse walks.
const INVERSE_BLOCK: u32 = 1024;

/// The tokenization of a latent raster into patches, and its inverse.
///
/// One token holds a `temporal × spatial × spatial` patch of every channel, channel-major, and
/// tokens run in raster order over the patch grid. The forward is the `patch` operator, which packs
/// the `1 × 2 × 2` patch a diffusion transformer reads; the inverse is `unpatchify_video`, which
/// unpacks any patch shape into a planar raster.
#[derive(Clone, Copy, Debug)]
pub struct Patchify {
    channels: u32,
    temporal: u32,
    spatial: u32,
}

impl Patchify {
    /// Patches of `temporal × spatial × spatial` positions of `channels` channels.
    ///
    /// # Errors
    ///
    /// A zero extent.
    pub fn new(channels: u32, temporal: u32, spatial: u32) -> Result<Self> {
        if channels == 0 || temporal == 0 || spatial == 0 {
            return Err(Error::shape(
                "Patchify",
                "a patch has channels, frames and rows".to_owned(),
            ));
        }
        Ok(Self {
            channels,
            temporal,
            spatial,
        })
    }

    /// The elements of one token: `channels · temporal · spatial²`.
    pub fn token_width(&self) -> u64 {
        u64::from(self.channels)
            * u64::from(self.temporal)
            * u64::from(self.spatial)
            * u64::from(self.spatial)
    }

    /// The patch grid of a raster of `shape` `[channels, frames, rows, columns]`.
    fn grid(&self, shape: ops::Shape) -> Result<[u32; 3]> {
        if shape.rank() != 4 || shape.dim(0) != u64::from(self.channels) {
            return Err(Error::shape(
                "Patchify",
                format!(
                    "{:?} is not [{} channels, frames, rows, columns]",
                    shape.dims(),
                    self.channels
                ),
            ));
        }
        let patch = [self.temporal, self.spatial, self.spatial].map(u64::from);
        let mut grid = [0; 3];
        for axis in 0..3 {
            let extent = shape.dim(axis + 1);
            if !extent.is_multiple_of(patch[axis]) {
                return Err(Error::shape(
                    "Patchify",
                    format!(
                        "{:?} is not whole {}×{}×{} patches",
                        shape.dims(),
                        self.temporal,
                        self.spatial,
                        self.spatial
                    ),
                ));
            }
            grid[axis] = u32::try_from(extent / patch[axis]).map_err(|_| {
                Error::shape(
                    "Patchify",
                    format!("{extent} positions overflow a kernel word"),
                )
            })?;
        }
        Ok(grid)
    }

    /// `latent` `[channels, frames, rows, columns]` packed into fresh `[tokens, token_width]`
    /// tokens of its element type, in raster order over the patch grid.
    ///
    /// # Errors
    ///
    /// A latent that is not whole patches, a patch other than `1 × 2 × 2` (the one the `patch`
    /// operator packs), an arena that cannot hold the tokens, or the operator's refusal.
    pub fn forward(&self, f: &Forward, latent: TensorView) -> Result<TensorView> {
        if (self.temporal, self.spatial) != (1, 2) {
            return Err(Error::shape(
                "Patchify",
                format!(
                    "the packing operator takes 1×2×2 patches, not {}×{}×{}",
                    self.temporal, self.spatial, self.spatial
                ),
            ));
        }
        let [frames, height, width] = self.grid(latent.shape)?;
        let count = u64::from(frames) * u64::from(height) * u64::from(width);
        let tokens = f.empty(latent.dtype, ops::Shape::new(&[count, self.token_width()]))?;
        let narrow = |value: u64| {
            u32::try_from(value)
                .map_err(|_| Error::shape("Patchify", format!("{value} overflows a kernel word")))
        };
        let descriptor = patch::Descriptor {
            dtype: latent.dtype,
            block: patch::BLOCK,
            pdl: PDL,
        };
        let args = patch::Args {
            x: latent,
            out: tokens,
            frames,
            channels: self.channels,
            height: narrow(latent.shape.dim(2))?,
            width: narrow(latent.shape.dim(3))?,
        };
        f.operate::<Patch>(&descriptor, args)?;
        Ok(tokens)
    }

    /// `tokens` `[tokens, token_width]` unpacked into a fresh planar raster of `raster`
    /// `[channels, frames, rows, columns]` extents, which state the patch grid, in the tokens'
    /// element type; tokens past the grid are not read.
    ///
    /// # Errors
    ///
    /// A raster that is not whole patches, too few tokens or tokens of another width, an arena that
    /// cannot hold the raster, or the operator's refusal.
    pub fn inverse(
        &self,
        f: &Forward,
        tokens: TensorView,
        raster: ops::Shape,
    ) -> Result<TensorView> {
        let [frames, height, width] = self.grid(raster)?;
        let count = u64::from(frames) * u64::from(height) * u64::from(width);
        if tokens.shape.rank() != 2
            || tokens.shape.dim(1) != self.token_width()
            || tokens.shape.dim(0) < count
        {
            return Err(Error::shape(
                "Patchify",
                format!(
                    "{:?} does not cover {frames}×{height}×{width} tokens {} wide",
                    tokens.shape.dims(),
                    self.token_width()
                ),
            ));
        }
        let out = f.empty(tokens.dtype, raster)?;
        // The affine form adds a per-channel scale and bias in the same pass; no consumer folds
        // one here, so the plain unpacking is the program.
        let descriptor = unpatchify::Descriptor {
            dtype: out.dtype,
            affine: false,
            block: INVERSE_BLOCK,
            pdl: PDL,
        };
        let args = unpatchify::Args {
            tokens,
            scale: None,
            bias: None,
            out,
            frames,
            height,
            width,
            channels: self.channels,
            patch_frames: self.temporal,
            patch_height: self.spatial,
            patch_width: self.spatial,
        };
        f.operate::<Unpatchify>(&descriptor, args)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arena;
    use ops::Shape;

    #[test]
    fn a_copy_moves_whole_elements() {
        assert!(Copy::new(DType::BF16, Access::Element).is_ok());
        assert!(Copy::new(DType::F32, Access::Vector).is_ok());
        assert!(Copy::new(DType::F4E2M1, Access::Element).is_err());
    }

    /// A patch has extents; a raster is whole patches of the layer's channels; the forward packs
    /// the `1 × 2 × 2` patch alone; the inverse reads tokens of the patch's width covering the grid.
    #[test]
    fn a_patchify_reads_whole_patches_of_its_channels() {
        assert!(Patchify::new(0, 1, 2).is_err());
        assert!(Patchify::new(16, 0, 2).is_err());
        assert!(Patchify::new(16, 1, 0).is_err());

        let arena = Arena::counting();
        let f = Forward::recording(&arena);
        let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
        let patchify = Patchify::new(16, 1, 2).unwrap();
        assert_eq!(patchify.token_width(), 64);
        let raster = Shape::new(&[16, 3, 4, 6]);
        let tokens = patchify.forward(&f, view(raster.dims())).unwrap();
        assert_eq!(tokens.shape.dims(), [3 * 2 * 3, 64]);
        let restored = patchify.inverse(&f, tokens, raster).unwrap();
        assert_eq!(restored.shape, raster);
        assert!(patchify.forward(&f, view(&[16, 3, 5, 6])).is_err());
        assert!(patchify.forward(&f, view(&[8, 3, 4, 6])).is_err());
        assert!(patchify.inverse(&f, view(&[17, 64]), raster).is_err());
        assert!(patchify.inverse(&f, view(&[18, 32]), raster).is_err());

        let wide = Patchify::new(16, 2, 4).unwrap();
        assert!(wide.forward(&f, view(&[16, 2, 4, 4])).is_err());
        assert!(
            wide.inverse(&f, view(&[1, 512]), Shape::new(&[16, 2, 4, 4]))
                .is_ok()
        );
    }
}
