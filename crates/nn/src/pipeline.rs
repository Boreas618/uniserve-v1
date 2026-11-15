//! A media pipeline as its family declares it to execution.
//!
//! A video family's pipeline is a chain of phases over products: the media reader writes a
//! request's condition products, the text encoder its conditioning, the denoiser its latents,
//! the decoders its media, and the engine's codecs encode and assemble them. Execution runs the
//! phases and keeps the products, and the family states which of them its pipeline has
//! ([`Pipeline`]): the order a request offers its phases in, each product with the phase that
//! writes it and the phase that reads it, in the order of their session classes, and each block a
//! deployment may place with the phases its ranks serve and how its work divides over them
//! ([`proto::config::Partition`]). The engine iterates the declaration; it names no stage of a
//! family, and a block the declaration does not name is refused.
//!
//! The products are the data a capability reads and writes (an [`crate::Encoder`]'s
//! conditioning, a [`crate::Denoiser`]'s latents, a [`crate::Decoder`]'s media), so their kinds
//! are one vocabulary ([`Product`]); which of them a pipeline holds, who writes and reads each,
//! and in which order the session classes them is the family's.

use proto::config::Partition;
pub use proto::module::Phase;
use serde::{Deserialize, Serialize};

/// A product one phase's calls write for another's to read: one class of the session per product
/// a pipeline declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Product {
    /// The RGB24 pixels of every visual condition, `[pixels, 3]` bytes in request order, each
    /// condition's frames whole.
    ConditionPixels,
    /// The stereo PCM of every audio condition, `[samples, 2]` in request order.
    ConditionSamples,
    /// The vision processor patches of every vision block, `[patches, features]` in
    /// presentation order.
    VisionPixels,
    /// The text encoder's prompt features, one row per presentation token.
    Conditioning,
    /// The denoiser's final video latents.
    VideoLatents,
    /// The denoiser's final audio latents.
    AudioLatents,
    /// The video decoder's RGB media units.
    VideoUnits,
    /// The video decoder's splice tail of each media unit, `[3, overlap frames, rows, columns]`:
    /// the last frames of unit `n` the decode of unit `n + 1` splices onto.
    VideoTail,
    /// The video codec's encoded media units.
    EncodedUnits,
    /// The audio decoder's PCM track.
    AudioSamples,
}

impl Product {
    /// The product's name: what its encoder outputs are keyed under
    /// (`storage::key::encoder_output`) and what execution reports it by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::ConditionPixels => "condition_pixels",
            Self::ConditionSamples => "condition_samples",
            Self::VisionPixels => "vision_pixels",
            Self::Conditioning => "conditioning",
            Self::VideoLatents => "video_latents",
            Self::AudioLatents => "audio_latents",
            Self::VideoUnits => "video_units",
            Self::VideoTail => "video_tail",
            Self::EncodedUnits => "encoded_units",
            Self::AudioSamples => "audio_samples",
        }
    }
}

/// How many blocks of a product one request holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Extent {
    /// One block per request: the calls of the writing phase write it, its last call or each
    /// call its slice.
    Request,
    /// One block per media unit of the request's timeline: each call of the writing phase writes
    /// the blocks of the units of its round, each call of the reading phase reads those of the
    /// round of the same index, and a product its own phase reads is the carry the round before
    /// left, the block of the unit before the round's first.
    Unit,
}

/// One product of a pipeline: who writes it, who reads it and how many blocks a request holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductSpec {
    pub product: Product,
    /// The phase whose calls write the product.
    pub writer: Phase,
    /// The phase whose calls read the product.
    pub reader: Phase,
    pub extent: Extent,
}

/// A part of the pipeline a deployment places on a worker group's ranks, under its name in the
/// deployment file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    /// The block's name in a deployment file: the checkpoint's stage of a network
    /// (`model_index.json`'s name).
    pub name: String,
    /// The phases of a request the block's ranks serve; none for a block that computes no phase
    /// of its own.
    pub phases: Vec<Phase>,
    /// How the block's work divides over its ranks, which states the forms of a deployment's
    /// block the block admits.
    pub partition: Partition,
}

/// A media pipeline as its family declares it (module documentation).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pipeline {
    /// Every phase of a request, in the order a request offers the ones that are ready: the
    /// first is its next phase, which the scheduling policy reads.
    pub phases: Vec<Phase>,
    /// The products, in the order of their session classes.
    pub products: Vec<ProductSpec>,
    /// The blocks a deployment may place beside the whole model and the engine's host blocks.
    pub blocks: Vec<Block>,
}

impl Pipeline {
    /// The index of `product` among [`Self::products`], its kind: the place of its class among
    /// the session's product classes. `None` for a product the pipeline does not hold.
    pub fn kind(&self, product: Product) -> Option<usize> {
        self.products
            .iter()
            .position(|spec| spec.product == product)
    }

    /// The block a deployment names `name`; `None` for a name the pipeline declares no block of.
    pub fn block(&self, name: &str) -> Option<&Block> {
        self.blocks.iter().find(|block| block.name == name)
    }

    /// The blocks whose ranks serve `phase`, in declaration order.
    pub fn serving(&self, phase: Phase) -> impl Iterator<Item = &Block> {
        self.blocks
            .iter()
            .filter(move |block| block.phases.contains(&phase))
    }

    /// Checks that the declaration is one execution can run: each product written and read by
    /// phases the pipeline states and declared once, and each block's phases phases of the
    /// pipeline under a name declared once.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Unsupported`] naming the first product or block that breaks a rule.
    pub fn check(&self) -> crate::Result<()> {
        let refuse = |problem: String| Err(crate::Error::Unsupported(problem));
        for (kind, spec) in self.products.iter().enumerate() {
            for phase in [spec.writer, spec.reader] {
                if !self.phases.contains(&phase) {
                    return refuse(format!(
                        "product {} is written or read by {phase}, which the pipeline does not run",
                        spec.product.name()
                    ));
                }
            }
            if self.kind(spec.product) != Some(kind) {
                return refuse(format!(
                    "the pipeline declares product {} twice",
                    spec.product.name()
                ));
            }
        }
        for block in &self.blocks {
            if let Some(phase) = block
                .phases
                .iter()
                .find(|phase| !self.phases.contains(phase))
            {
                return refuse(format!(
                    "block {} serves {phase}, which the pipeline does not run",
                    block.name
                ));
            }
            if self
                .block(&block.name)
                .is_some_and(|first| !std::ptr::eq(first, block))
            {
                return refuse(format!("the pipeline declares block {} twice", block.name));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pipeline of a text encoder and a denoiser whose latents a decoder reads.
    fn pipeline() -> Pipeline {
        let spec = |product, writer, reader| ProductSpec {
            product,
            writer,
            reader,
            extent: Extent::Request,
        };
        Pipeline {
            phases: vec![Phase::TextEncode, Phase::Denoise, Phase::VideoDecode],
            products: vec![
                spec(Product::Conditioning, Phase::TextEncode, Phase::Denoise),
                spec(Product::VideoLatents, Phase::Denoise, Phase::VideoDecode),
            ],
            blocks: vec![
                Block {
                    name: "encoder".to_owned(),
                    phases: vec![Phase::TextEncode],
                    partition: Partition::Group,
                },
                Block {
                    name: "decoder".to_owned(),
                    phases: vec![Phase::VideoDecode],
                    partition: Partition::Units { carries: false },
                },
            ],
        }
    }

    /// A product's kind is its place among the declared products, and a block is found by its
    /// name and by the phases it serves.
    #[test]
    fn a_pipeline_is_read_by_its_declaration() {
        let pipeline = pipeline();
        pipeline.check().unwrap();
        assert_eq!(pipeline.kind(Product::VideoLatents), Some(1));
        assert_eq!(pipeline.kind(Product::AudioLatents), None);
        assert_eq!(
            pipeline.block("decoder").map(|block| block.partition),
            Some(Partition::Units { carries: false })
        );
        assert!(pipeline.block("video_decoder").is_none());
        let serving: Vec<&str> = pipeline
            .serving(Phase::TextEncode)
            .map(|block| block.name.as_str())
            .collect();
        assert_eq!(serving, ["encoder"]);
    }

    /// A declaration whose products or blocks name phases it does not run, or name one product
    /// or block twice, is refused.
    #[test]
    fn a_declaration_of_phases_it_does_not_run_is_refused() {
        let mut stray = pipeline();
        stray.products[0].reader = Phase::Mux;
        assert!(stray.check().is_err());

        let mut twice = pipeline();
        twice.products.push(twice.products[0]);
        assert!(twice.check().is_err());

        let mut block = pipeline();
        block.blocks[0].phases.push(Phase::AudioDecode);
        assert!(block.check().is_err());

        let mut named = pipeline();
        named.blocks.push(named.blocks[0].clone());
        assert!(named.check().is_err());
    }
}
