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

use std::collections::HashMap;
use std::sync::Mutex;

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
    /// `[channels, frames, rows, columns]` the video latent trajectory: a denoise advances it step
    /// by step and a decode then reads it.
    VideoLatents,
    /// The audio latent trajectory, in the same sense.
    AudioLatents,
    /// `[channels, frames, rows, columns]` the decoded clip: a decode writes it and the transport
    /// that turns it into an artifact reads it.
    ///
    /// It is a product for the same reason the latents are: the region is a runner's own column, and
    /// a value another call must read has nowhere to be unless it is named. The name is
    /// `(request, kind)` like every other, so the transport reads one clip by the identity the
    /// request was admitted under rather than by a position in a batch.
    Video,
    /// `[channels, samples]` the decoded audio track, in the same sense as [`Product::Video`].
    Track,
}

impl Product {
    /// The product's name, as a refusal states it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Source => "source",
            Self::Images => "images",
            Self::Audio => "audio",
            Self::VideoLatents => "video latents",
            Self::AudioLatents => "audio latents",
            Self::Video => "video",
            Self::Track => "track",
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

    /// Drops every product of `request`, answering how many there were.
    ///
    /// **The executor calls this when a request's last call has been served**, and it is the other
    /// half of `publish`: a product that outlives its request is a region a later request of the
    /// same identity could read, and an identity is reused across epochs. A store that kept every
    /// request's products would grow with the epoch while its regions belong to nobody.
    fn forget(&self, request: RequestKey) -> usize;
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

/// The products of the requests in flight, as the executor holds them.
///
/// A store of **views**, not of memory: [`Products::publish`] is handed the region a call wrote, so
/// this type allocates nothing and owns nothing. What it owns is the record of *where* each product
/// is - its device and the stream that wrote it - which is the half a reader must consult before it
/// reads: a view is only readable by work issued after the write that filled it.
///
/// # The two crossings it refuses, and why refusing is the safe answer
///
/// A product written on one stream and read on another needs an ordering the store cannot establish
/// by itself - an event recorded at the write and awaited at the read - and a product written on one
/// device and read on another needs that ordering **and** a copy. Both belong to the executor, which
/// owns the event pool and the pools a copy allocates from, so this store refuses by name and says
/// which of the two crossings it met. The refusal is what keeps a reader from aliasing a region
/// whose write it did not wait for: a read that silently proceeded would be correct in the common
/// case and wrong under load, which is the worst pair of outcomes to debug.
///
/// # Lifetime
///
/// A request's products live until the executor says the request is done
/// ([`Held::forget`]). Without that call the map grows with the epoch, so a runner that finishes a
/// request ends it here; the entries are views, so what leaks is bookkeeping and not memory.
pub struct Held {
    entries: Mutex<HashMap<(RequestKey, Product), Placed>>,
}

/// One product's region and the write that filled it.
#[derive(Clone, Copy, Debug)]
struct Placed {
    view: TensorView,
    device: DeviceId,
    stream: StreamId,
}

impl Default for Held {
    fn default() -> Self {
        Self::new()
    }
}

impl Held {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// How many products the store holds, for a reader checking that a request ended cleanly.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("the product store").len()
    }

    /// Whether the store holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Products for Held {
    fn publish(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
        view: &TensorView,
    ) -> Result<()> {
        // A trajectory is published again at every step and the last write is the one a reader
        // wants, so publishing replaces rather than refusing a second write.
        self.entries.lock().expect("the product store").insert(
            (request, kind),
            Placed {
                view: *view,
                device,
                stream,
            },
        );
        Ok(())
    }

    fn borrow(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Option<TensorView>> {
        let placed = self
            .entries
            .lock()
            .expect("the product store")
            .get(&(request, kind))
            .copied();
        let Some(placed) = placed else {
            return Ok(None);
        };
        if placed.device != device {
            return Err(crate::Error::Invalid(format!(
                "request {} reads its {} region on device {} and it was written on device {}: a \
                 product that moves between devices needs a copy ordered after its producer, which \
                 the executor issues and this store does not",
                request.request_id.0,
                kind.name(),
                device.0,
                placed.device.0
            )));
        }
        if placed.stream != stream {
            return Err(crate::Error::Invalid(format!(
                "request {} reads its {} region on stream {} and it was written on stream {}: the \
                 reader must wait for the writer's event, or read on the writer's stream",
                request.request_id.0,
                kind.name(),
                stream.0,
                placed.stream.0
            )));
        }
        Ok(Some(placed.view))
    }

    fn forget(&self, request: RequestKey) -> usize {
        let mut entries = self.entries.lock().expect("the product store");
        let before = entries.len();
        entries.retain(|(key, _), _| *key != request);
        before - entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ops::{DType, DevicePtr, Shape};
    use platform::DeviceId;

    fn request(id: u64) -> RequestKey {
        RequestKey::new(1, ipc::RequestId(id), 0)
    }

    /// A view whose element count names it: `TensorView` exposes no address, so the count is what a
    /// test compares - the store's contract is that the same view comes back, not that a number does.
    fn view(address: u64, elements: u64) -> TensorView {
        TensorView::new(
            DevicePtr::new(address),
            DType::BF16,
            Shape::new(&[elements]),
        )
    }

    /// The shape a request's own runner has: it writes the trajectory and reads it back on one
    /// stream, and that must be the plain answer rather than a refusal.
    #[test]
    fn a_product_written_and_read_on_one_stream_is_returned() {
        let store = Held::new();
        store
            .publish(
                request(1),
                Product::VideoLatents,
                DeviceId(0),
                StreamId(1),
                &view(4096, 2),
            )
            .expect("a store accepts a view");

        let read = store
            .borrow(request(1), Product::VideoLatents, DeviceId(0), StreamId(1))
            .expect("one stream needs no ordering")
            .expect("the product was published");
        assert_eq!(read.numel(), 2, "the same region comes back");
    }

    /// A reader on another device is refused, and the refusal names both devices: a silent answer
    /// would alias a region whose copy was never ordered.
    #[test]
    fn a_product_read_on_another_device_is_refused_naming_both() {
        let store = Held::new();
        store
            .publish(
                request(1),
                Product::Text,
                DeviceId(0),
                StreamId(1),
                &view(8192, 3),
            )
            .expect("a store accepts a view");

        let refusal = store
            .borrow(request(1), Product::Text, DeviceId(2), StreamId(1))
            .expect_err("another device needs a copy")
            .to_string();
        assert!(refusal.contains("device 2"), "names the reader: {refusal}");
        assert!(refusal.contains("device 0"), "names the writer: {refusal}");
    }

    /// A reader on another stream of the **same** device is refused too, and names both streams: the
    /// device is the same fact and the ordering is not.
    #[test]
    fn a_product_read_on_another_stream_is_refused_naming_both() {
        let store = Held::new();
        store
            .publish(
                request(1),
                Product::Images,
                DeviceId(0),
                StreamId(1),
                &view(16384, 4),
            )
            .expect("a store accepts a view");

        let refusal = store
            .borrow(request(1), Product::Images, DeviceId(0), StreamId(4))
            .expect_err("another stream needs the writer's event")
            .to_string();
        assert!(refusal.contains("stream 4"), "names the reader: {refusal}");
        assert!(refusal.contains("stream 1"), "names the writer: {refusal}");
    }

    /// A kind no call produced is `None`, which the runner turns into its own refusal by name: the
    /// store does not invent a region and does not decide what the absence means.
    #[test]
    fn an_unwritten_product_answers_none() {
        let store = Held::new();
        assert!(
            store
                .borrow(request(1), Product::Audio, DeviceId(0), StreamId(1))
                .expect("an absent product is not an error")
                .is_none(),
            "nothing wrote this kind"
        );
    }

    /// A trajectory is published at every step, so the second write replaces the first: a reader
    /// that got the first region would advance a step the solver already left.
    #[test]
    fn publishing_again_replaces_the_previous_region() {
        let store = Held::new();
        store
            .publish(
                request(1),
                Product::VideoLatents,
                DeviceId(0),
                StreamId(1),
                &view(4096, 2),
            )
            .expect("first step");
        store
            .publish(
                request(1),
                Product::VideoLatents,
                DeviceId(0),
                StreamId(1),
                &view(4096, 2),
            )
            .expect("second step");

        assert_eq!(store.len(), 1, "one product, published twice");
        assert_eq!(
            store
                .borrow(request(1), Product::VideoLatents, DeviceId(0), StreamId(1))
                .expect("readable")
                .expect("present")
                .numel(),
            2
        );
    }

    /// A request's products end with the request and take nothing else with them: a later request of
    /// the same identity must not read the earlier one's region.
    #[test]
    fn forget_drops_one_requests_products_and_no_other() {
        let store = Held::new();
        for kind in [Product::Text, Product::VideoLatents] {
            store
                .publish(request(1), kind, DeviceId(0), StreamId(1), &view(4096, 2))
                .expect("published");
        }
        store
            .publish(
                request(2),
                Product::Text,
                DeviceId(0),
                StreamId(1),
                &view(8192, 3),
            )
            .expect("published");

        assert_eq!(store.forget(request(1)), 2, "both of the first request's");
        assert_eq!(store.len(), 1, "the other request keeps its own");
        assert!(
            store
                .borrow(request(1), Product::Text, DeviceId(0), StreamId(1))
                .expect("absent, not an error")
                .is_none(),
            "the forgotten request keeps nothing"
        );
    }
}
