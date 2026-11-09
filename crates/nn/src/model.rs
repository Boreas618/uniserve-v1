//! A loaded model as the worker is to see it: named blocks, each implementing one capability.
//!
//! This is the **producer half** of the seam `specs/worker.md` 4.3 states — a block per capability,
//! addressed by a name and a phase — and the consumer half is not wired yet. A model's loader
//! builds one of these and nothing in the tree hands it on: the engine's plan
//! (`src/model_runner.rs`) ends at `Arc<qwen3_moe::Qwen3Moe>`, and the worker's runners
//! (`crates/worker/src/runners/`) address work by `ipc::PhaseId` with no block registry. **The
//! consumer is `exec`'s to land**; until it does, this is a stated contract whose other half is owed
//! rather than a path a run exercises.
//!
//! Nothing here names a concrete model: a block is a trait object, and the one place a model class is
//! named is the loader in `uniserve engine`, which is also where the consumer half belongs.
//!
//! [`BlockName`] names a block of a deployment (`backbone`, `text_encoder`). The pair of a block name
//! and an [`ops::PhaseKind`] is the engine's addressing unit, carried there as its own opaque
//! `ipc::PhaseId`; this crate keeps only the name and re-exports the phase kind.

use std::fmt;
use std::sync::Arc;

use ops::PhaseKind;

use crate::capability::{CausalLm, Decoder, Denoiser, Encoder};

/// The name a deployment gives one block of a model, e.g. `backbone` or `text_encoder`.
///
/// The name comes from the deployment file's block assignment and travels with every phase the
/// worker schedules, so a log line or a scheduling decision says which block it is about.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockName(Arc<str>);

impl BlockName {
    /// The block named `name`.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
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
///
/// The payload is shared, not owned: a model that composes its components holds them as
/// `Arc<TextEncoder>`, `Arc<VideoDecoder>` and so on — it is the owner of its computation, and it is
/// also the party that must answer what programs those components launch. `Model` carries the same
/// objects as capability views, so there is one of each, one load, and no drift between the object
/// the model computes with and the trait object the worker dispatches on. `Arc<T>` coerces to
/// `Arc<dyn Capability>` at the construction site, as `Box<T>` did.
pub enum Block {
    /// A tower that turns inputs into conditioning.
    Encoder(Arc<dyn Encoder>),
    /// A diffusion model that denoises latents.
    Denoiser(Arc<dyn Denoiser>),
    /// A decoder that turns latents back into media.
    Decoder(Arc<dyn Decoder>),
    /// An autoregressive token model.
    CausalLm(Arc<dyn CausalLm>),
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
/// A deployment places each block on a worker group; the worker builds one runner per resident
/// block from the capability it implements (specs/worker.md 4.3, 4.5).
#[derive(Debug)]
pub struct Model {
    /// Held for the model's lifetime.
    _blocks: Vec<(BlockName, Block)>,
}

impl Model {
    /// The model with `blocks`, in deployment order.
    pub fn new(blocks: Vec<(BlockName, Block)>) -> Self {
        Self { _blocks: blocks }
    }
}
