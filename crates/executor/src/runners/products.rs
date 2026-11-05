//! A request's media products, held by the executor and moved between devices.
//!
//! A media request's conditioning and its latent trajectory are produced by one runner and read by
//! another: the text encoder's rows are read by every denoising step, the latent trajectory a denoise
//! advances is decoded by a third runner. **Nothing carries them today**, and the reason is not a
//! missing copy — it is a missing name: a runner's arena is its own, so a value another runner must
//! read has nowhere to be.
//!
//! This module states that name and nothing else. The executor owns the storage, decides where a
//! product lives, and moves it when the reader is on another device; **a runner only says which
//! product it wrote and which it needs**.
//!
//! # The two facts the contract fixes
//!
//! - **A product is named by `(request, kind)`**, never by a position: a batch numbers its calls by
//!   their order, so a position says where a request is in one batch and nothing about which state
//!   belongs to whom across a request's life.
//! - **A copy is ordered by the stream it is issued on.** The producer's write is on the producing
//!   device's stream; a reader on another device reads a copy issued on **its own** stream and
//!   ordered after the producer's. The executor owns that ordering because only it knows both
//!   streams; a runner that issued its own cross-device copy would be ordering against a stream it
//!   does not hold.
//!
//! # What is deliberately not a product here
//!
//! **The request's raw condition media.** `nn::capability::Conditioning::media` is a `MediaInput` per
//! condition - the input's pixels, its audio and its anchoring draw, all device views the media
//! reader produced - and a conditioner encodes them in its own forward. It is an **input** to the
//! conditioning rather than a row of it, it is per condition rather than per request, and the
//! scheduler already names each condition's product (`MediaFlowState::condition_media`, keyed by the
//! condition's own name). A region named `(request, kind)` cannot carry a list, so putting it here
//! would state a mechanism the other side cannot use. The kinds below are the ones a request has
//! exactly one of, which is what makes one name per kind enough.
//!
//! # What a runner may assume
//!
//! A region is written once per producing call and read by many later calls; a reader never writes.
//! `borrow` answers `None` for a product no call of the request produced, and that absence is a
//! refusal at the caller — *the step reads `text` and nothing produced it* — never an empty region,
//! because an empty region is a step that runs on zeros.

use ipc::RequestKey;
use ops::TensorView;
use platform::DeviceId;

use crate::Result;
use crate::arena::StreamId;

/// One kind of media product a request has.
///
/// The kinds are the model's own: `nn::capability::Layout` states the rows, elements and shapes of
/// each, so the executor sizes a region from the model's declaration and never from a guess. They
/// are one list rather than one mechanism per product because they differ only in their size and in
/// which call produces them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Product {
    /// `[text_rows, width]` encoded prompt: the text encoder's rows.
    Text,
    /// The text encoder's own features, which a conditioner folds into `Text` in place.
    Source,
    /// `[rows, …]` encoded reference image.
    Images,
    /// `[rows, …]` encoded reference audio.
    Audio,
    /// The request's latent trajectory: a denoise advances it step by step and a decode reads it.
    Latents,
}

impl Product {
    /// The product's name, as a refusal states it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Source => "source",
            Self::Images => "images",
            Self::Audio => "audio",
            Self::Latents => "latents",
        }
    }
}

/// A request's products, as a runner sees them.
///
/// One call per direction, and both are the executor's to implement: it owns the storage, the
/// devices and the streams, and a runner owns none of those.
pub trait Products: Send + Sync {
    /// Publishes `request`'s `kind` product from `view`.
    ///
    /// `view` is storage on `device`, and the bytes are copied out on `stream` — the producing
    /// runner's own stream, at the end of the call that computed them. The contract is that the copy
    /// is issued before this returns and that a later `borrow` on any device reads the bytes it
    /// covered.
    ///
    /// A product already published by this request is replaced: a trajectory is published step by
    /// step, and the reader wants the latest.
    fn publish(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
        view: &TensorView,
    ) -> Result<()>;

    /// Borrows `request`'s `kind` product for a call on `device`.
    ///
    /// When the product lives on another device of this process, the copy is issued on `stream` and
    /// ordered after the producer's write, so the returned view is readable by work this runner
    /// issues on that stream afterwards.
    ///
    /// `None` is the answer for a product no call of `request` produced: the caller refuses by name
    /// rather than reading a region nobody wrote.
    fn borrow(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Option<TensorView>>;
}

/// The answer to a `borrow` that names a product no call produced, as the caller reports it.
///
/// Stated here so the two runners that read products refuse in one voice: the alternative is two
/// spellings of the same absence, which is how a reader concludes the mechanism and the model
/// disagree.
pub fn no_store(request: RequestKey) -> crate::Error {
    crate::Error::Invalid(format!(
        "request {} reads a media product and this runner was given no product store: the epoch's \
         products are attached by the executor before the first batch",
        request.request_id.0
    ))
}

/// The absence of one kind, as a caller reports it.
pub fn absent(request: RequestKey, kind: Product) -> crate::Error {
    crate::Error::Invalid(format!(
        "request {} reads its {} region and no call of it produced one",
        request.request_id.0,
        kind.name()
    ))
}
