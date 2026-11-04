//! A loaded model as the executor sees it: named blocks, each implementing one capability.
//!
//! `uniserve engine` loads a checkpoint, hands the [`Model`] to the executor, and the executor
//! serves each block by the capability it implements and the phases the deployment gives it
//! (specs/worker.md 4.3). Nothing here names a concrete model: a block is a trait object, and the
//! one place a model class is named is the loader in `uniserve engine`.
//!
//! [`BlockName`] names a block of a deployment (`backbone`, `text_encoder`). The pair of a block
//! name and an [`ops::PhaseKind`] is the engine's addressing unit, carried there as its own opaque
//! `PhaseId`; this crate keeps only the name and re-exports the phase kind.

use std::fmt;
use std::sync::Arc;

use ops::PhaseKind;

use crate::capability::{CausalLm, Decoder, Denoiser, Encoder};

/// The name a deployment gives one block of a model, e.g. `backbone` or `text_encoder`.
///
/// The name comes from the deployment file's block assignment and travels with every phase the
/// executor schedules, so a log line or a scheduling decision says which block it is about.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockName(Arc<str>);

impl BlockName {
    /// The block named `name`.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
    }

    /// The name as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BlockName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl fmt::Debug for BlockName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BlockName({:?})", &*self.0)
    }
}

/// One phase of one block: what the scheduler's work items name.
///
/// The engine keeps its own opaque index for a phase (`PhaseId` in `crates/ipc`); this pair is the
/// model swarm's name for the same thing, so a log line or a runner's phase list is readable.
pub type PhaseId = (BlockName, PhaseKind);

/// One capability a model's block implements.
pub enum Block {
    /// A tower that turns inputs into conditioning.
    Encoder(Box<dyn Encoder>),
    /// A diffusion model that denoises latents.
    Denoiser(Box<dyn Denoiser>),
    /// A decoder that turns latents back into media.
    Decoder(Box<dyn Decoder>),
    /// An autoregressive token model.
    CausalLm(Box<dyn CausalLm>),
}

impl fmt::Debug for Block {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Encoder(_) => "Block::Encoder",
            Self::Denoiser(_) => "Block::Denoiser",
            Self::Decoder(_) => "Block::Decoder",
            Self::CausalLm(_) => "Block::CausalLm",
        })
    }
}

/// A loaded model: its named blocks.
///
/// A deployment places each block on a worker group; the executor builds one runner per resident
/// block from the capability it implements (specs/worker.md 4.3, 4.5).
#[derive(Debug)]
pub struct Model {
    blocks: Vec<(BlockName, Block)>,
}

impl Model {
    /// The model with `blocks`, in deployment order.
    pub fn new(blocks: Vec<(BlockName, Block)>) -> Self {
        Self { blocks }
    }

    /// The blocks, in deployment order.
    pub fn blocks(&self) -> impl Iterator<Item = (&BlockName, &Block)> {
        self.blocks.iter().map(|(name, block)| (name, block))
    }

    /// The block named `name`.
    pub fn block(&self, name: &BlockName) -> Option<&Block> {
        self.blocks
            .iter()
            .find_map(|(block, capability)| (block == name).then_some(capability))
    }

    /// Take the block named `name` out of the model, for the runner that serves it.
    pub fn take(&mut self, name: &BlockName) -> Option<Block> {
        self.blocks
            .iter()
            .position(|(block, _)| block == name)
            .map(|index| self.blocks.remove(index).1)
    }

    /// How many blocks the model has.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the model has no block.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
