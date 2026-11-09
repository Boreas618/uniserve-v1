//! A request's media products, held by the worker and moved between devices.
//!
//! A media request's conditioning and its latent trajectory are produced by one runner and read by
//! another: the text encoder's rows are read by every denoising step, the latent trajectory a denoise
//! advances is decoded by a third runner. **Nothing carries them today**, and the reason is not a
//! missing copy — it is a missing name: a runner's arena is its own, so a value another runner must
//! read has nowhere to be.
//!
//! This module states that name and nothing else. The worker owns the storage, decides where a
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
//!   ordered after the producer's. The worker owns that ordering because only it knows both
//!   streams; a runner that issued its own cross-device copy would be ordering against a stream it
//!   does not hold.
//!
//! # A request's raw condition media
//!
//! `nn::capability::Conditioning::media` is a `MediaInput` per condition - the input's pixels, its
//! audio and its anchoring draw - and a conditioner encodes them in its own forward. The media reader
//! lays a request's conditions out one after another in three regions, the three the scheduler names
//! (`ipc::product`): [`Product::ConditionPixels`], [`Product::ConditionSamples`] and
//! [`Product::VisionPixels`]. A request has exactly one of each, which is what makes one name per
//! kind enough, and a reader finds one condition's rows from the extents the request's own
//! conditioning states.
//!
//! # What a runner may assume
//!
//! A region is written once per producing call and read by many later calls; a reader never writes.
//! `borrow` answers `None` for a product no call of the request produced, and that absence is a
//! refusal at the caller — *the step reads `text` and nothing produced it* — never an empty region,
//! because an empty region is a step that runs on zeros.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ipc::RequestKey;
use ops::TensorView;
use platform::{DeviceId, Event};

use crate::Result;
use crate::arena::StreamId;

/// One kind of media product a request has.
///
/// The kinds are the model's own: `nn::capability::Layout` states the rows, elements and shapes of
/// each, so the worker sizes a region from the model's declaration and never from a guess. They
/// are one list rather than one mechanism per product because they differ only in their size and in
/// which call produces them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// What one request hands to the next call.
///
/// **Which of these a runner publishes today, and which is only named:**
///
/// | product | published by | read by |
/// | --- | --- | --- |
/// | [`Product::Text`] | the encoder runner | the denoiser, through `copy_into` |
/// | [`Product::VideoLatents`] | the denoiser runner | the decoder, through `copy_into` |
/// | [`Product::AudioLatents`] | the denoiser runner | the audio decode |
/// | [`Product::Video`] | the decoder runner | the media encode, when it writes the artifact |
/// | [`Product::ConditionPixels`], [`Product::ConditionSamples`] | the media stage's read | the denoiser's first step |
/// | [`Product::VisionPixels`] | the media stage's read | the text encoding |
/// | [`Product::Track`] | the audio decode runner | the audio encode, when it writes the artifact |
///
/// A kind nobody published is not a hole a reader falls into: the store answers `None` and a refusal
/// names the product and the request.
pub enum Product {
    /// `[text_rows, width]` encoded prompt: the text encoder's rows.
    Text,
    /// Every visual condition's pixels, in request order, as its encoder reads them.
    ConditionPixels,
    /// Every audio track's samples, in request order, as the audio encoder reads them.
    ConditionSamples,
    /// Every vision block's patch rows, in presentation order.
    VisionPixels,
    /// `[channels, frames, rows, columns]` the video latent trajectory: a denoise advances it step
    /// by step and a decode then reads it.
    VideoLatents,
    /// The audio latent trajectory, in the same sense.
    AudioLatents,
    /// The request's video frames: a decode writes them and the encoder that turns them into an
    /// artifact's units reads them.
    ///
    /// **The extent is the postprocessor's**, not the clip's: `[frames, rows, columns, 3]` bytes,
    /// where a frame is one frame the timeline keeps. A deployment that serves no postprocessor
    /// publishes the decoded clip instead, `[channels, frames, rows, columns]` in the model's own
    /// element type. It is a product for the same reason the latents are: the region is a runner's
    /// own column, and a value another call must read has nowhere to be unless it is named. The name
    /// is `(request, kind)` like every other, so the encoder reads one request's frames by the
    /// identity the request was admitted under rather than by a position in a batch.
    ///
    /// **One product per unit of the timeline, which the number names.** A decode runs ahead of the
    /// encodes that read it, so a later clip must not replace one nobody has read.
    Video(u32),
    /// `[channels, samples]` the decoded audio track, in the same sense as [`Product::Video`].
    ///
    /// The audio decode runner publishes it and an audio encode reads it. It is a different product
    /// from [`Product::ConditionSamples`] on purpose: that one is a request's reference audio (an
    /// input the media reader produced), and this one is what this request's own denoising produced:
    /// a conditioner reads the first, the muxer's audio encode reads the second, and a store that
    /// gave them one name would let a `ref2va` request read its input as its output.
    Track,
    /// The last frames of decoded clip `n`, `[channels, frames, rows, columns]`, which clip `n + 1`
    /// fades against. It is a product because the two clips may be decoded on different ranks.
    Tail(u32),
}

impl Product {
    /// The product's name, as a refusal states it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::ConditionPixels => "condition pixels",
            Self::ConditionSamples => "condition samples",
            Self::VisionPixels => "vision pixels",
            Self::VideoLatents => "video latents",
            Self::AudioLatents => "audio latents",
            Self::Video(_) => "video",
            Self::Track => "track",
            Self::Tail(_) => "tail",
        }
    }
}

/// A request's products, as a runner sees them.
///
/// One call per direction, and both are the worker's to implement: it owns the storage, the
/// devices and the streams, and a runner owns none of those.
pub trait Products: Send + Sync {
    /// Publishes `request`'s `kind` product from `view`.
    ///
    /// `view` is storage on `device` and `stream` is the producing runner's own stream, named by the
    /// id a runner computes from it. `written` is that stream's write as an event
    /// ([`written_on`]) — **a producer that expects a reader on another stream or device must state
    /// it**, because that reader's copy is ordered by waiting on it; a producer publishing for its
    /// own stream alone may pass `None`, and a crossing read of such a product is refused by name
    /// rather than issued unordered.
    ///
    /// The ids and not the stream itself is what keeps this call free of a device: the store's rules
    /// — which device and stream may borrow, which must copy, what a mismatch says — are then testable
    /// where no GPU exists.
    ///
    /// A product already published by this request is replaced: a trajectory is published step by
    /// step, and the reader wants the latest.
    fn publish(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
        written: Option<Arc<Event>>,
        view: &TensorView,
    ) -> Result<()>;

    /// Borrows `request`'s `kind` product as a view a call on `stream` may read **directly**.
    ///
    /// This is the zero-copy answer, and it is only given where the producer's write is already
    /// ordered before the reader's work - the same device and the same stream. Any other device or stream
    /// is refused by name, and [`Products::copy_into`] is the call for it: a caller that wants the
    /// bytes somewhere else asks for a copy it can order, rather than a view whose ordering it
    /// cannot see.
    ///
    /// `None` is the answer for a product no call of `request` produced: the caller refuses by name
    /// rather than reading a region nobody wrote.
    ///
    /// **No production caller today, and it stays for two reasons.** Both crossings a runner makes go
    /// through [`Products::copy_into`] - the trajectory and the clip each land on their reader's
    /// device - so this answers the case a same-stream read would take; and the borrowing rules
    /// it states (same device *and* same stream, else refused by name) are the only ones asserted
    /// without a device. A caller arriving later gets the zero-copy read the moment its two calls are
    /// on one stream, and nothing else in the store decides that.
    fn borrow(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Option<TensorView>>;

    /// Orders `stream` after the write of `request`'s `kind` product and answers its view, to read
    /// in place.
    ///
    /// This is the read of a product too large to copy for one use: a request's condition media is
    /// read once, by the forward that encodes it, so a copy would double its bytes for nothing. The
    /// view is only for a reader on the writer's device, because a view cannot cross one, and only
    /// for work issued on `stream` after this call. `None` is a product no call of `request`
    /// produced.
    fn ordered(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        let _ = (request, kind, stream);
        Err(crate::Error::Invalid(
            "this product store cannot order a read in place: it holds no device".into(),
        ))
    }

    /// Whether `request` has a `kind` product, **whatever device and stream it lives on**.
    ///
    /// Presence is a fact about the store; the device and stream are facts about where a reader wants it. A
    /// runner's step check asks the first and only the copy answers the second, so a product on
    /// another device is *held* - a deployment that puts the text encoder on its own GPU would
    /// otherwise refuse every denoising step with "nothing produced it".
    fn holds(&self, request: RequestKey, kind: Product) -> bool;

    /// What `request`'s `kind` product states to a reader that must place its bytes, or `None` when
    /// no call produced it.
    ///
    /// A reader needs both facts before it copies. The **length** sizes the copy and the padding
    /// around it - the conditioning row takes the product's rows and pads the rest of its own. The
    /// **element type** decides whether that copy is a copy at all: a product of another type is a
    /// conversion, and the conversion a boundary performs is one explicit step rather than a byte
    /// copy (`copy_into` refuses it by name). The type has to come from the store because it is the
    /// writer that stated it, and a reader that had to borrow the region to learn it would have
    /// already crossed the boundary it is checking.
    fn product(&self, request: RequestKey, kind: Product) -> Result<Option<Published>>;

    /// Copies `request`'s `kind` product into `destination`, ordered on `stream`.
    ///
    /// **This is the call that crosses a device or a stream**, and it answers whether there was
    /// anything to copy: `false` is a product no call of `request` produced, which a caller refuses
    /// by name (`absent`).
    ///
    /// The ordering is the store's because the store is the only holder of both ends: `publish`
    /// records the producer's write as an event on the producer's stream, and this call makes
    /// `stream` wait on that event before issuing the copy on it. Same stream and same device is a
    /// plain copy; a second stream on the same device and a second device are both the ordered copy,
    /// because in both the reader's work is not otherwise ordered after the writer's.
    ///
    /// `destination` is the reader's buffer, and it must be exactly the product's size: a partial
    /// copy is a reader whose buffer disagrees with the product's shape, which is a refusal with
    /// both byte counts rather than a copy of the shorter one.
    fn copy_into(
        &self,
        request: RequestKey,
        kind: Product,
        destination: &TensorView,
        stream: &platform::Stream,
    ) -> Result<bool>;

    /// Orders `stream` after `kind`'s write and answers its region, for a reader that reads it in
    /// place: `None` when the request holds no such product.
    ///
    /// The region stays the producer's, and a reader on another device reads it where it lies. Only
    /// a region in pinned host storage may be read from another device.
    fn lent(
        &self,
        _request: RequestKey,
        _kind: Product,
        _stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        Err(crate::Error::Invalid(
            "this store lends no region to a reader".into(),
        ))
    }

    /// Copies `request`'s `kind` product into host memory and waits for the copy to land.
    ///
    /// This is the read the media work makes. `crates/media`'s encoders and muxer take host bytes,
    /// and both products they consume - the clip and the track - are device regions. `out` is pinned
    /// host memory of exactly the product's size, which [`Products::product`] states; `stream` is
    /// the reader's own stream, and the copy is ordered on it after the producer's write exactly as
    /// [`Products::copy_into`] orders a device copy. The call **waits** before it returns, because
    /// the reader runs on this thread and has no stream to wait on.
    ///
    /// `false` is a product no call of `request` produced - never a buffer of zeros, which would be
    /// a frame of black nobody made.
    ///
    /// The bytes carry the producer's own element type: this store copies, and a `f16` clip stays
    /// `f16`. A reader that wants `u8` pixels converts them, because the conversion is the model's
    /// arithmetic and its range is the model's statement.
    ///
    /// A store that holds no device cannot answer, and its default refusal says so rather than
    /// handing back a buffer nobody wrote.
    fn stage(
        &self,
        request: RequestKey,
        kind: Product,
        out: &platform::PinnedBuffer,
        stream: &platform::Stream,
    ) -> Result<bool> {
        let _ = (request, kind, out, stream);
        Err(crate::Error::Invalid(
            "this product store cannot copy a product to host memory: it holds no device, and the \
             media work reads its bytes on the host"
                .into(),
        ))
    }

    /// Drops every product of `request`, answering how many there were.
    ///
    /// **The worker calls this when a request's last call has been served**, and it is the other
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
         products are attached by the worker before the first batch",
        request.request_id.0
    ))
}

/// What a published product states to a reader that must place its bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Published {
    /// The product's length, in bytes.
    pub bytes: u64,
    /// The element type its producer wrote, which is the only type a copy of it can land in.
    pub dtype: ops::DType,
}

/// The write a producer is about to publish, as an event a crossing reader waits on.
///
/// One function for the three runners, so the event's flags and the fact that it is recorded **on
/// the producing stream** before the publish cannot differ between them.
pub fn written_on(stream: &platform::Stream) -> Result<Arc<Event>> {
    let written = Arc::new(Event::new(platform::EventFlags::default()));
    written.record(stream)?;
    Ok(written)
}

/// The absence of one kind, as a caller reports it.
pub fn absent(request: RequestKey, kind: Product) -> crate::Error {
    crate::Error::Invalid(format!(
        "request {} reads its {} region and no call of it produced one",
        request.request_id.0,
        kind.name()
    ))
}

/// The products of the requests in flight, as the worker holds them.
///
/// A store of **views**, not of memory: [`Products::publish`] is handed the region a call wrote, so
/// this type allocates nothing and owns nothing. What it owns is the record of *where* each product
/// is - its device and the stream that wrote it - which is the half a reader must consult before it
/// reads: a view is only readable by work issued after the write that filled it.
///
/// # The two crossings, and how they are ordered
///
/// A product written on one stream and read on another needs an ordering, and one written on one
/// device and read on another needs that ordering **and** a copy. The store is the only holder of
/// both ends, so it records the write as an event on the producer's stream at
/// [`Products::publish`] and makes the reader's stream wait on that event before it issues the copy
/// ([`Products::copy_into`]). **A deployment whose text phase owns its own GPU crosses a device on
/// every denoising step**, so this is a required path rather than a convenience.
///
/// [`Products::borrow`] stays the zero-copy answer and is only given where the writer's stream *is*
/// the reader's: a view handed across an ordering the caller cannot see is correct in the common
/// case and wrong under load, which is the worst pair of outcomes to debug.
///
/// # Lifetime
///
/// A request's products live until the worker says the request is done
/// ([`Held::forget`]). Without that call the map grows with the epoch, so a runner that finishes a
/// request ends it here; the entries are views, so what leaks is bookkeeping and not memory.
pub struct Held {
    entries: Mutex<HashMap<(RequestKey, Product), Placed>>,
}

/// The products of a group of ranks: each rank's own, and the ones a rank makes for another.
///
/// The ranks of a parallel group compute one block together and each holds the result, so a rank's
/// conditioning and trajectory are its own bytes on its own device under the name every rank
/// publishes: one store for the group would hold the last rank's region under it. What one rank
/// produces for another - a clip's frames, its tail, the track - is named once and read by whichever
/// rank reads it, so those are shared.
pub struct GroupProducts {
    ranks: Vec<Held>,
    shared: Held,
}

impl GroupProducts {
    /// The store of a group of `ranks` ranks.
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ranks: (0..ranks).map(|_| Held::new()).collect(),
            shared: Held::new(),
        })
    }
}

/// One rank's view of its group's products.
pub struct RankProducts {
    group: Arc<GroupProducts>,
    rank: usize,
}

impl RankProducts {
    /// The view of rank `rank` of `group`.
    pub fn new(group: Arc<GroupProducts>, rank: usize) -> Self {
        Self { group, rank }
    }

    /// The store that holds `kind` for this rank.
    fn store(&self, kind: Product) -> &Held {
        match kind {
            Product::Video(_)
            | Product::Track
            | Product::Tail(_)
            | Product::ConditionPixels
            | Product::ConditionSamples
            | Product::VisionPixels => &self.group.shared,
            _ => &self.group.ranks[self.rank],
        }
    }
}

impl Products for RankProducts {
    fn publish(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
        written: Option<Arc<Event>>,
        view: &TensorView,
    ) -> Result<()> {
        self.store(kind)
            .publish(request, kind, device, stream, written, view)
    }

    fn borrow(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<Option<TensorView>> {
        self.store(kind).borrow(request, kind, device, stream)
    }

    fn ordered(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        match kind {
            // The media reader's regions are pinned host storage, which every rank's device reads
            // in place: the first rank published them, and the rest order after its write.
            Product::ConditionPixels | Product::ConditionSamples | Product::VisionPixels => {
                self.group.shared.ordered_from_host(request, kind, stream)
            }
            _ => self.store(kind).ordered(request, kind, stream),
        }
    }

    fn holds(&self, request: RequestKey, kind: Product) -> bool {
        self.store(kind).holds(request, kind)
    }

    fn product(&self, request: RequestKey, kind: Product) -> Result<Option<Published>> {
        self.store(kind).product(request, kind)
    }

    fn copy_into(
        &self,
        request: RequestKey,
        kind: Product,
        destination: &TensorView,
        stream: &platform::Stream,
    ) -> Result<bool> {
        self.store(kind)
            .copy_into(request, kind, destination, stream)
    }

    fn lent(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        self.store(kind).lent(request, kind, stream)
    }

    fn stage(
        &self,
        request: RequestKey,
        kind: Product,
        out: &platform::PinnedBuffer,
        stream: &platform::Stream,
    ) -> Result<bool> {
        self.store(kind).stage(request, kind, out, stream)
    }

    fn forget(&self, request: RequestKey) -> usize {
        let own: usize = self
            .group
            .ranks
            .iter()
            .map(|rank| rank.forget(request))
            .sum();
        own + self.group.shared.forget(request)
    }
}

/// One product's region and the write that filled it.
///
/// `written` is that write, recorded on the producing stream by the producer ([`written_on`]): a
/// reader on any other stream or device waits on it before copying the bytes, which is what makes a
/// copy legal without the reader knowing anything about the producer. It is `None` for a product its
/// producer published for its own stream alone, and a crossing read of that is refused rather than
/// issued unordered.
#[derive(Clone)]
struct Placed {
    view: TensorView,
    device: DeviceId,
    stream: StreamId,
    written: Option<Arc<Event>>,
}

/// Where a copy of `placed` into `stream` starts: immediately, or after the writer's event.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// The writer's stream is the reader's, so its write is already ordered before the copy.
    Ordered,
    /// Another stream or another device: the reader waits on the writer's event first.
    AfterTheWriter,
}

/// Whether a product's elements may be written into a destination of another type.
///
/// Only the same type is a copy. A different type is a **conversion**, and a conversion is a
/// computation: the design replaced `torch.autocast` with each layer declaring its precision, which
/// leaves exactly the conversions a stated handover performs - and this store's handover copies. The
/// check is over types and not over byte counts because `f16` and `bf16` are the same width, so an
/// agreeing byte count is precisely the case that would copy one type's bits as another's.
fn copyable(from: ops::DType, to: ops::DType, request: RequestKey, kind: Product) -> Result<()> {
    if from == to {
        return Ok(());
    }
    Err(crate::Error::Invalid(format!(
        "request {} copies {} of {from:?} into a buffer of {to:?}: elements of another type are a \
         conversion and not a copy, and the conversion at a product's boundary is one explicit step \
         that the handover performs",
        request.request_id.0,
        kind.name()
    )))
}

/// Which route a read of a product takes from the writer's device and stream to the reader's.
///
/// One function for the decision, over ids rather than a stream, so the table is testable where no
/// device exists - and so the two facts cannot be read one at a time by different code: **a second
/// device always crosses, and a second stream on the same device crosses too.** Only both equal is
/// the plain copy.
fn route(written: (DeviceId, StreamId), read: (DeviceId, StreamId)) -> Route {
    if written == read {
        Route::Ordered
    } else {
        Route::AfterTheWriter
    }
}

impl Default for Held {
    fn default() -> Self {
        Self::new()
    }
}

impl Held {
    /// The view of a product held in pinned host storage, for a reader on any device, ordered after
    /// its write on `stream`.
    ///
    /// [`Products::ordered`] refuses a reader on another device because a device's storage does not
    /// cross; host storage does, which only a caller that knows the region's kind can say.
    fn ordered_from_host(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(None);
        };
        if let Some(written) = placed.written.as_ref() {
            stream.wait_event(written)?;
        }
        Ok(Some(placed.view))
    }

    /// An empty store.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// How many products the store holds, for a reader checking that a request ended cleanly.
    pub fn len(&self) -> usize {
        crate::lock(&self.entries).len()
    }

    /// Whether the store holds no product.
    pub fn is_empty(&self) -> bool {
        crate::lock(&self.entries).is_empty()
    }
}

impl Products for Held {
    fn publish(
        &self,
        request: RequestKey,
        kind: Product,
        device: DeviceId,
        stream: StreamId,
        written: Option<Arc<Event>>,
        view: &TensorView,
    ) -> Result<()> {
        // A trajectory is published again at every step and the last write is the one a reader
        // wants, so publishing replaces rather than refusing a second write.
        crate::lock(&self.entries).insert(
            (request, kind),
            Placed {
                view: *view,
                device,
                stream,
                written,
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
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(None);
        };
        if placed.device != device {
            return Err(crate::Error::Invalid(format!(
                "request {} reads its {} region on device {} and it was written on device {}: a \
                 borrowed view carries no ordering, so a reader on another device asks for a copy \
                 (`copy_into`) and this store orders it after the producer's write",
                request.request_id.0,
                kind.name(),
                device.0,
                placed.device.0
            )));
        }
        if placed.stream != stream {
            return Err(crate::Error::Invalid(format!(
                "request {} reads its {} region on stream {} and it was written on stream {}: a \
                 borrowed view carries no ordering, so a reader on another stream asks for a copy \
                 (`copy_into`) and this store orders it after the producer's write",
                request.request_id.0,
                kind.name(),
                stream.0,
                placed.stream.0
            )));
        }
        Ok(Some(placed.view))
    }

    fn ordered(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(None);
        };
        if placed.device != stream.device() {
            return Err(crate::Error::Invalid(format!(
                "request {} reads its {} region in place on device {} and it was written on device \
                 {}: a view does not cross a device, so such a reader asks for a copy (`copy_into`)",
                request.request_id.0,
                kind.name(),
                stream.device().0,
                placed.device.0
            )));
        }
        let read = (stream.device(), StreamId(stream.raw()));
        if route((placed.device, placed.stream), read) == Route::AfterTheWriter {
            let Some(written) = placed.written.as_ref() else {
                return Err(crate::Error::Invalid(format!(
                    "request {} reads {} on stream {} and it was written on stream {} with no \
                     record of its write: a producer that expects a crossing reader states the \
                     event (`written_on`) when it publishes",
                    request.request_id.0,
                    kind.name(),
                    stream.raw(),
                    placed.stream.0
                )));
            };
            stream.wait_event(written)?;
        }
        Ok(Some(placed.view))
    }

    fn holds(&self, request: RequestKey, kind: Product) -> bool {
        crate::lock(&self.entries).contains_key(&(request, kind))
    }

    fn product(&self, request: RequestKey, kind: Product) -> Result<Option<Published>> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        match placed {
            Some(placed) => Ok(Some(Published {
                bytes: placed.view.byte_len()?,
                dtype: placed.view.dtype,
            })),
            None => Ok(None),
        }
    }

    fn lent(
        &self,
        request: RequestKey,
        kind: Product,
        stream: &platform::Stream,
    ) -> Result<Option<TensorView>> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(None);
        };
        let read = (stream.device(), StreamId(stream.raw()));
        if route((placed.device, placed.stream), read) == Route::AfterTheWriter {
            let Some(written) = placed.written.as_ref() else {
                return Err(crate::Error::Invalid(format!(
                    "request {} lends {} written on device {} with no record of its write",
                    request.request_id.0,
                    kind.name(),
                    placed.device.0
                )));
            };
            stream.wait_event(written)?;
        }
        Ok(Some(placed.view))
    }

    fn copy_into(
        &self,
        request: RequestKey,
        kind: Product,
        destination: &TensorView,
        stream: &platform::Stream,
    ) -> Result<bool> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(false);
        };
        // **The element type first, and on types rather than on byte counts.** Two types can be the
        // same width - `f16` and `bf16` are both two bytes - so a byte count that agrees is exactly
        // the case that would copy a reinterpretation silently. A different type is a conversion, not
        // a copy, and the design names the boundary's conversion as one explicit step its handover
        // performs (specs/worker.md 9 replaced `torch.autocast` with declared per-layer precision);
        // this store copies bytes and does not pretend to convert them.
        copyable(placed.view.dtype, destination.dtype, request, kind)?;
        let product = placed.view.byte_len()?;
        let wanted = destination.byte_len()?;
        if product != wanted {
            return Err(crate::Error::Invalid(format!(
                "request {} copies {product} byte(s) of {} into a buffer of {wanted}: a partial \
                 copy is a reader whose buffer disagrees with the product's shape",
                request.request_id.0,
                kind.name()
            )));
        }
        let read = (stream.device(), StreamId(stream.raw()));
        if route((placed.device, placed.stream), read) == Route::AfterTheWriter {
            // The crossing needs the producer's write as an event. A product published without one
            // was published for the writer's own stream alone, so reading it here would be a copy
            // the producer never ordered: refused by name, never issued unordered.
            let Some(written) = placed.written.as_ref() else {
                return Err(crate::Error::Invalid(format!(
                    "request {} copies {} to device {} stream {} and it was written on device {} \
                     stream {} with no record of its write: a producer that expects a crossing \
                     reader states the event (`written_on`) when it publishes",
                    request.request_id.0,
                    kind.name(),
                    stream.device().0,
                    stream.raw(),
                    placed.device.0,
                    placed.stream.0
                )));
            };
            stream.wait_event(written)?;
        }
        // SAFETY: both ranges outlive the copy - the product's region belongs to the request, which
        // is in flight while it is read, and `destination` belongs to the caller - and the copy is
        // ordered on `stream` after the writer's event where they differ.
        unsafe {
            platform::memcpy_async(
                destination.ptr.address(),
                placed.view.ptr.address(),
                usize::try_from(product)
                    .map_err(|_| crate::Error::Invalid("a product is larger than memory".into()))?,
                stream,
            )?;
        }
        Ok(true)
    }

    fn stage(
        &self,
        request: RequestKey,
        kind: Product,
        out: &platform::PinnedBuffer,
        stream: &platform::Stream,
    ) -> Result<bool> {
        let placed = crate::lock(&self.entries).get(&(request, kind)).cloned();
        let Some(placed) = placed else {
            return Ok(false);
        };
        let product = placed.view.byte_len()?;
        if product != out.len() as u64 {
            return Err(crate::Error::Invalid(format!(
                "request {} stages {product} byte(s) of {} into {} byte(s) of host memory: a \
                 partial copy is a reader whose buffer disagrees with the product's shape",
                request.request_id.0,
                kind.name(),
                out.len()
            )));
        }
        let read = (stream.device(), StreamId(stream.raw()));
        if route((placed.device, placed.stream), read) == Route::AfterTheWriter {
            let Some(written) = placed.written.as_ref() else {
                return Err(crate::Error::Invalid(format!(
                    "request {} stages {} written on device {} stream {} with no record of its \
                     write: a producer that expects a host reader states the event (`written_on`) \
                     when it publishes",
                    request.request_id.0,
                    kind.name(),
                    placed.device.0,
                    placed.stream.0
                )));
            };
            stream.wait_event(written)?;
        }
        // SAFETY: both ranges outlive the copy - the product's region belongs to the request, which
        // is in flight while it is read, and `out` belongs to the caller - and the copy is ordered
        // on `stream` after the writer's event where they differ. `synchronize` is what makes the
        // host bytes readable on this thread, and it is why the caller needs no stream of its own.
        unsafe {
            platform::memcpy_async(
                out.address(),
                placed.view.ptr.address(),
                usize::try_from(product)
                    .map_err(|_| crate::Error::Invalid("a product is larger than memory".into()))?,
                stream,
            )?;
        }
        stream.synchronize()?;
        Ok(true)
    }

    fn forget(&self, request: RequestKey) -> usize {
        let mut entries = crate::lock(&self.entries);
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

    /// A copy is bytes of one type; another type is a conversion, and the store says so by name.
    ///
    /// The row that matters is `f16` into `bf16`: both are two bytes, so a store that compared byte
    /// counts would copy it and every reader would read a reinterpretation. On a device that is a
    /// wrong image rather than an error, and nothing downstream can tell.
    #[test]
    fn a_product_of_another_element_type_is_refused_by_both_types() {
        assert!(
            copyable(DType::F16, DType::F16, request(1), Product::VideoLatents).is_ok(),
            "one type is the copy this store performs"
        );
        for (from, to) in [
            (DType::BF16, DType::F16),
            (DType::F32, DType::F16),
            (DType::F16, DType::BF16),
        ] {
            let error = copyable(from, to, request(1), Product::VideoLatents)
                .expect_err("another type is a conversion");
            let message = format!("{error}");
            assert!(
                message.contains(&format!("{from:?}")) && message.contains(&format!("{to:?}")),
                "the refusal names both types: {message}"
            );
            assert!(
                message.contains("video latents"),
                "and the product it is about: {message}"
            );
        }
    }

    /// The crossing table: only the writer's own stream on the writer's own device is the plain copy.
    ///
    /// The two rows that matter are the ones a single-device pipeline never meets and a real
    /// deployment does: a second **device** (the text phase on its own GPU) and a second **stream** on
    /// one device (two runners). Both cross, so both wait on the producer's write before copying. A
    /// rule that compared only one of the two would pass on a single-GPU deployment and read
    /// unordered bytes on the one that ships.
    #[test]
    fn only_the_writers_own_stream_copies_without_waiting() {
        let written = (DeviceId(0), StreamId(7));
        assert_eq!(route(written, (DeviceId(0), StreamId(7))), Route::Ordered);
        assert_eq!(
            route(written, (DeviceId(0), StreamId(8))),
            Route::AfterTheWriter,
            "a second stream on one device is a crossing"
        );
        assert_eq!(
            route(written, (DeviceId(1), StreamId(7))),
            Route::AfterTheWriter,
            "a second device is a crossing even when the stream ids agree"
        );
    }

    /// Every kind's name is its own, so a refusal says which product is missing.
    ///
    /// The names are what a reader acts on: `absent` and `no_store` put `kind.name()` in the
    /// sentence a request fails with, and a name two kinds share would report the wrong one of them —
    /// a `video` missing when the trajectory is what never ran. The kinds are listed exhaustively so a
    /// new one cannot arrive without a name of its own.
    #[test]
    fn every_kind_of_product_has_a_name_of_its_own() {
        let kinds = [
            Product::Text,
            Product::VisionPixels,
            Product::ConditionPixels,
            Product::ConditionSamples,
            Product::VideoLatents,
            Product::AudioLatents,
            Product::Video(0),
            Product::Track,
        ];
        let mut names: Vec<&'static str> = kinds.iter().map(|kind| kind.name()).collect();
        assert!(names.iter().all(|name| !name.is_empty()), "{names:?}");
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two kinds share a name: {names:?}");
    }

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
    /// Each rank of a group holds its own conditioning under the one name every rank publishes,
    /// what a rank makes for another is one region for the whole group, and one reader's `forget`
    /// ends the request on every rank.
    #[test]
    fn a_groups_ranks_keep_their_own_rows_and_share_what_crosses_them() {
        let group = GroupProducts::new(2);
        let (first, second) = (
            RankProducts::new(Arc::clone(&group), 0),
            RankProducts::new(Arc::clone(&group), 1),
        );
        let publish = |store: &RankProducts, kind, address, device: i32| {
            store
                .publish(
                    request(1),
                    kind,
                    DeviceId(device),
                    StreamId(1),
                    None,
                    &view(address, 8),
                )
                .expect("a publish");
        };
        publish(&first, Product::Text, 0x1000, 0);
        publish(&second, Product::Text, 0x2000, 1);
        let read = |store: &RankProducts, device: i32| {
            store
                .borrow(request(1), Product::Text, DeviceId(device), StreamId(1))
                .expect("a read")
                .expect("the rank's own rows")
                .ptr
                .address()
        };
        assert_eq!(read(&first, 0), 0x1000);
        assert_eq!(read(&second, 1), 0x2000);

        publish(&second, Product::Video(3), 0x3000, 1);
        publish(&second, Product::Tail(3), 0x4000, 1);
        assert!(first.holds(request(1), Product::Video(3)));
        assert!(first.holds(request(1), Product::Tail(3)));

        assert_eq!(first.forget(request(1)), 4);
        assert!(!second.holds(request(1), Product::Text));
        assert!(!second.holds(request(1), Product::Video(3)));
    }

    #[test]
    fn a_product_written_and_read_on_one_stream_is_returned() {
        let store = Held::new();
        store
            .publish(
                request(1),
                Product::VideoLatents,
                DeviceId(0),
                StreamId(1),
                None,
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
                None,
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
                Product::ConditionPixels,
                DeviceId(0),
                StreamId(1),
                None,
                &view(16384, 4),
            )
            .expect("a store accepts a view");

        let refusal = store
            .borrow(
                request(1),
                Product::ConditionPixels,
                DeviceId(0),
                StreamId(4),
            )
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
                .borrow(
                    request(1),
                    Product::ConditionSamples,
                    DeviceId(0),
                    StreamId(1)
                )
                .expect("an absent product is not an error")
                .is_none(),
            "nothing wrote this kind"
        );
    }

    /// A reader that must place a product's bytes asks the store for both facts: the conditioning row
    /// sizes its copy from the length and pads the rest, and the length it sees is the **current**
    /// write's, because a trajectory is published step by step and the latest is the one a reader
    /// wants.
    #[test]
    fn the_length_is_the_latest_writes_and_the_request_that_wrote_it() {
        let store = Held::new();
        assert_eq!(
            store
                .product(request(1), Product::Text)
                .expect("an absent product is not an error"),
            None,
            "nothing wrote this request's kind"
        );
        for elements in [4096, 8192] {
            store
                .publish(
                    request(1),
                    Product::Text,
                    DeviceId(0),
                    StreamId(1),
                    None,
                    &view(64, elements),
                )
                .expect("a store accepts a view");
        }
        assert_eq!(
            store
                .product(request(1), Product::Text)
                .expect("a published product states its bytes and its type"),
            Some(Published {
                bytes: 8192 * 2,
                dtype: DType::BF16,
            }),
            "the second write is the one a reader places, in the type its producer wrote"
        );
        assert_eq!(
            store
                .product(request(2), Product::Text)
                .expect("another request's absence is not an error"),
            None,
            "a product belongs to the request that wrote it"
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
                None,
                &view(4096, 2),
            )
            .expect("first step");
        store
            .publish(
                request(1),
                Product::VideoLatents,
                DeviceId(0),
                StreamId(1),
                None,
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
                .publish(
                    request(1),
                    kind,
                    DeviceId(0),
                    StreamId(1),
                    None,
                    &view(4096, 2),
                )
                .expect("published");
        }
        store
            .publish(
                request(2),
                Product::Text,
                DeviceId(0),
                StreamId(1),
                None,
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
