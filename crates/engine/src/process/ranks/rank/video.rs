//! A rank of a media deployment (`specs/engine-worker-port.md` §2): its place among the
//! deployment's worker groups, its share of the video pipeline, the runners of the blocks its
//! group places on it over one media memory, and what it states of its group.
//!
//! **Ranks and blocks.** A media rank is one rank of one worker group, on the device the
//! deployment places it on; a device may hold ranks of several groups (a text encoder's and a
//! denoiser's, as `configs/minimax_h3/fasth3-dp8-text-tp8.toml` places them), each a rank of its
//! own with its own media memory, graph pool and runners. Its group's blocks name the phases it
//! serves: the text encoder's block its text encodings, the denoiser's its latent preparations
//! and steps, the video decoder's its decode rounds, the audio decoder's first rank the request's
//! track, and the host blocks ([`MEDIA_READER`], [`VIDEO_CODEC_BLOCK`], [`MUXER`]) its host work.
//! A group that places the whole model ([`DEFAULT_BLOCK`]) serves every phase on its one rank.
//! The rank loads the stages of its group's blocks that hold it, split as those blocks state
//! ([`ModelRunner::video_split`]).
//!
//! **Runners.** One per block the rank serves, each on a stream of its own: [`EncoderRunner`],
//! [`DenoiserRunner`], a [`DecoderRunner`] for the video and one for the track, and the
//! [`MediaHost`] of its host work, which computes nothing on the device. They read and write a
//! request's products in the blocks the rank's device block tables name, in the rank's one
//! [`BufferMemory`], which every media runner of the rank shares.
//!
//! **Sizes.** A rank of the group that denoises states its group's captured request shapes with the
//! bytes of a slot's trajectory and of a block of each product kind there (`Capacity::media_bytes`,
//! in `crate::batch::product::KINDS` order), from its denoiser's shapes and the decoders'
//! declarations, and a decoding rank buckets by the denoiser's layouts of the served requests.
//! Neither needs its share to hold the stage it reads: the meta load its profile run reads holds
//! the deployment's denoiser and decoders beside its share ([`ModelRunner::load_profiled`]), and
//! its device load its share alone. The session sizes every rank's product buffer and its host
//! regions from them (`crate::process::ranks::layout`).

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::batch::Capacity;
use crate::batch::capacity::variants;
use crate::batch::product;
use crate::batch::{Extent, Phase, Shape, Variant};
use crate::memory::buffers::{BufferMemory, UNIT_BYTES};
use crate::memory::client::StorageClient;
use crate::memory::division::{self, call_window};
use crate::memory::encoders;
use crate::memory::latent::host::BankPages;
use crate::parallel::HostGroup;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::decoder::{DecoderConfig, DecoderRunner, RoundConfig};
use crate::runner::denoiser::{DenoiserConfig, DenoiserRunner};
use crate::runner::encoder::{EncoderConfig, EncoderRunner};
use crate::runner::graphs::Graphs;
use crate::runner::media::{HostConfig, MediaHost};
use crate::runner::thread_pool::ThreadPool;
use crate::runner::{Runner, RunnerId};
use anyhow::{Context as _, Result, bail};
use nn::VideoRequest;
use nn::family::{Stage, VideoModel, VideoPlacement, VideoServing};
use ops::Generation;
use platform::{Communicator, DeviceId, Stream};
use proto::config::{
    DEFAULT_BLOCK, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK, WorkerGroupConfig, WorkerGroupRole,
};
use proto::video::Layout;
use tensor::DType;

use super::memory::{Peaks, granularity, unreserved};
use super::text::Keys;
use super::{ModelRunner, Weights};

/// The block a deployment places a video request's media read under: one rank of its group
/// reads the conditions into the request's products.
pub(crate) const MEDIA_READER: &str = "media_reader";

/// The block a deployment places a video request's track encode and its muxing under: one rank
/// of its group assembles the request's artifact.
pub(crate) const MUXER: &str = "muxer";

/// Every phase of a video request, which a group that places the whole model serves.
const MEDIA_PHASES: [Phase; 9] = [
    Phase::MediaRead,
    Phase::TextEncode,
    Phase::LatentPrepare,
    Phase::Denoise,
    Phase::VideoDecode,
    Phase::AudioDecode,
    Phase::VideoEncode,
    Phase::AudioEncode,
    Phase::Mux,
];

/// The host phases of a video request, which the media host serves.
const HOST_PHASES: [Phase; 4] = [
    Phase::MediaRead,
    Phase::VideoEncode,
    Phase::AudioEncode,
    Phase::Mux,
];

/// The values of one page of the host latent region: a mebibyte of float32 samples.
const LATENT_PAGE_VALUES: u32 = 1 << 18;

/// The runners of a media rank that compute on the device, by the region of the graph pool
/// each replays in. The media host computes nothing there.
pub(crate) const ENCODER: RunnerId = RunnerId(0);
pub(crate) const DENOISER: RunnerId = RunnerId(1);
pub(crate) const VIDEO_DECODER: RunnerId = RunnerId(2);
pub(crate) const AUDIO_DECODER: RunnerId = RunnerId(3);

/// The runners of a media rank's device blocks, each `None` on a rank that serves none of its
/// phases: the text encoder, the denoiser, the video decoder and the track's decoder.
pub(crate) type DeviceRunners = (
    Option<EncoderRunner>,
    Option<DenoiserRunner>,
    Option<DecoderRunner>,
    Option<DecoderRunner>,
);

/// `runners` as the worker takes them, each with the region of the graph pool it replays in.
pub(crate) fn boxed(
    (encoder, denoiser, video, audio): DeviceRunners,
) -> Vec<(RunnerId, Box<dyn Runner>)> {
    let mut runners: Vec<(RunnerId, Box<dyn Runner>)> = Vec::new();
    if let Some(runner) = encoder {
        runners.push((ENCODER, Box::new(runner)));
    }
    if let Some(runner) = denoiser {
        runners.push((DENOISER, Box::new(runner)));
    }
    if let Some(runner) = video {
        runners.push((VIDEO_DECODER, Box::new(runner)));
    }
    if let Some(runner) = audio {
        runners.push((AUDIO_DECODER, Box::new(runner)));
    }
    runners
}

/// A rank of a media deployment as its start holds it: its share of the pipeline, its parallel
/// groups' communicators, what the profile run of its runners states, and once it is opened its
/// media memory and its serving runners.
pub(crate) struct VideoRank {
    /// The stages the rank's group places, split as its blocks state.
    pub(crate) model: Arc<dyn VideoModel>,
    /// The text encoder's tensor group, and the denoiser's group along each axis of its
    /// partition ([`DENOISER_AXES`]); `None` for a group of one and on the meta device.
    pub(crate) tensor: Option<Arc<ops::Group>>,
    pub(crate) denoiser: DenoiserGroups,
    /// The communicator of the ranks a video decode round is dealt to, over which they pass the
    /// splice tails between consecutive shares, from the rank's load until its video decoder
    /// takes it; `None` for a round of one rank, on a rank that decodes no video and on the meta
    /// device.
    pub(crate) decoder: Option<Communicator>,
    /// What the profile run states of each runner that replays in the graph pool.
    pub(crate) peaks: Vec<(RunnerId, Peaks)>,
    /// The device bytes the runners hold beside their regions of the graph pool: the denoiser's
    /// state buffer and the decoders' buffers.
    pub(crate) buffers: u64,
    /// The bytes of one slot's row of the rank's trajectory rows; 0 on a rank that does not
    /// denoise.
    pub(crate) row: u64,
    /// The units of the longest timeline the deployment serves: the splice tails a request
    /// declares. 0 on a rank that does not denoise.
    pub(crate) units: u32,
    /// On a rank that denoises or decodes: the requests the deployment captures its denoiser
    /// for, each with the denoiser's layout of it, as the profile's meta load states them
    /// ([`ModelRunner::load_profiled`]); the rank's decoders bucket by them whether or not its
    /// share holds the denoiser. Empty on any other rank.
    pub(crate) laid: Vec<(VideoRequest, Layout)>,
    /// On a rank that denoises: the trajectory class's `tensors` and `encoding` digests, and the
    /// pages of its host latent region.
    pub(crate) trajectory: Option<([u8; 32], [u8; 32], BankPages)>,
    /// On a rank that denoises: the denoiser as its family describes it to request processing
    /// (`nn::family::Description::details`), which the engine states to the daemon; empty on any
    /// other rank.
    pub(crate) description: Vec<u8>,
    /// What the rank states of its worker group: its own phases and the shapes of its runners.
    pub(crate) capacity: Capacity,
    /// The allocation granularity of the rank's device, in bytes: what a slot's part of its
    /// product buffer and its trajectory row are whole multiples of
    /// (`BufferMemory::slot_units`). [`crate::memory::buffers::UNIT_BYTES`] on the meta device.
    pub(crate) granule: u64,
    /// The units of the rank's product buffer one request slot's part takes: the blocks one
    /// request holds of each product kind the rank backs or receives ([`held_kinds`]), in whole
    /// granules (`crate::process::ranks::fit_media_slots`); 0 for a rank that holds no kind,
    /// whose product buffer is one unit its slots take none of.
    pub(crate) slot_units: u64,
    /// The request slots of each of the deployment's worker groups, in the order the deployment
    /// states them: its running requests until the ranks of every host agree the fewest each
    /// group's ranks hold (`crate::process::ranks::fit_media_slots`). The session's layout states
    /// every group's slot rows and product buffers by them, those of other hosts' groups too.
    pub(crate) group_slots: Vec<u32>,
    /// The rank's member of the deployment's host group, from its load until the ranks of every
    /// host agreed `group_slots` over it.
    pub(crate) host: Option<HostGroup>,
    /// The rank's media memory, from its opening until the process's memories are connected.
    pub(crate) memory: Option<BufferMemory>,
    /// The rank's media memory as its runners share it once the process's memories are
    /// connected: what gives the storages of the slots the rank does not keep back
    /// (`crate::process::ranks::keep_warm_slots`).
    pub(crate) media: Option<Arc<BufferMemory>>,
    /// Once the rank is opened, by the numbers the session's answer gave them: the regions its
    /// device block tables map (its group's slot rows, its product buffer and its parts of host
    /// regions), and the product buffers of the process's other ranks, which its device reaches.
    pub(crate) regions: (Vec<u8>, Vec<u8>),
    /// The rank's serving runners, from its opening until its worker takes them.
    pub(crate) runners: Vec<Box<dyn Runner>>,
}

impl VideoRank {
    /// The units of the rank's product buffer for `slots` request slots: a part of
    /// [`Self::slot_units`] per slot, or one unit for a rank that backs no product kind.
    ///
    /// # Errors
    ///
    /// A buffer of more units than a region numbers.
    pub(crate) fn product_units(&self, slots: u32) -> Result<u32> {
        crate::process::ranks::buffer_units(self.slot_units, slots)
    }

    /// The parts the rank's product buffer of `slots` slots is mapped in: one per slot, whose
    /// storages a give-back releases with its slot, or one for a rank that backs no product kind.
    pub(crate) fn product_parts(&self, slots: u32) -> u32 {
        if self.slot_units > 0 { slots } else { 1 }
    }

    /// This rank holding `slots` request slots of its group, as the division of its GPU's share
    /// decides (`crate::process::ranks::fit_media_slots`) and its warm process then keeps
    /// (`crate::process::ranks::keep_warm_slots`): what it states of its group (its slots, and the
    /// prompts an encode holds) and the pages of its host latent region follow them. Its buckets
    /// and its pools are the deployment's, which the slots do not change.
    pub(crate) fn hold(&mut self, slots: u32) {
        self.capacity.slots = slots;
        if self.capacity.serves(Phase::TextEncode) {
            self.capacity.rows[Phase::TextEncode.index()] = phase_rows(Phase::TextEncode, slots, 0);
        }
        if let Some((_, _, pages)) = &mut self.trajectory {
            *pages = slot_pages(*pages, slots);
        }
    }
}

impl ModelRunner {
    /// One model runner per rank of a media deployment that this process's host holds, every
    /// such rank of its worker groups in the deployment's rank order, each numbered as the
    /// deployment numbers it and on the device the deployment places it on, which this process
    /// serves. A device that holds ranks of several groups serves each as a rank of its own. The
    /// ranks of other hosts are their own main processes' (`docs/architecture.md`, The channel);
    /// a product read in another process than the one that wrote it reaches the reader's process
    /// by a crossing into the receiving regions the session's layout states
    /// (`crate::process::ranks::layout`, [`product_ranks`]).
    ///
    /// # Errors
    ///
    /// A rank of this host on a device this process does not serve, or an expert group, which
    /// serves no video request.
    pub(super) fn media_ranks(&self) -> Result<Vec<Self>> {
        let groups = &self.engine_config().parallel.worker_groups;
        let mut ranks = Vec::new();
        for (rank, index, _, device) in super::placed_ranks(self.launch())? {
            let group = &groups[index];
            if group.role == WorkerGroupRole::Experts {
                bail!(
                    "worker group {} is an expert group, which serves no video request",
                    group.id
                );
            }

            let mut rank_runner = self.clone();
            rank_runner.devices = vec![device];
            rank_runner.rank = rank;
            rank_runner.group.clone_from(&group.id.0);
            rank_runner.world_size = group.ranks.len();
            // The text encoder's split is its block's (`video_split`).
            rank_runner.mapping.tensor_parallel = 1;
            rank_runner.mapping.tensor_rank = 0;
            ranks.push(rank_runner);
        }
        Ok(ranks)
    }

    /// The share of the video pipeline this rank's checkpoint holds that its group places on it,
    /// split as the group's blocks state, its weights reached through `weights`.
    ///
    /// # Errors
    ///
    /// The checkpoint's or the family's refusal, or a split the deployment's groups do not state.
    pub(crate) fn load_video(&self, weights: Weights<'_>) -> Result<Arc<dyn VideoModel>> {
        self.load_stages(weights, false)
    }

    /// The meta load a rank's profile run reads (steps 4-8): its share of the pipeline, and on a
    /// rank that denoises or decodes the stages whose declarations size what it computes,
    /// wherever the deployment places them ([`Self::profiled_stages`]). Nothing of a meta load
    /// computes or takes device memory, and the rank's device load holds its share alone.
    ///
    /// # Errors
    ///
    /// As [`Self::load_video`].
    pub(crate) fn load_profiled(&self) -> Result<Arc<dyn VideoModel>> {
        self.load_stages(Weights::Meta, true)
    }

    /// The stages of the video pipeline `sized` names ([`Self::profiled_stages`] when it holds,
    /// the rank's share otherwise), split as the group's blocks state, their weights reached
    /// through `weights`.
    ///
    /// # Errors
    ///
    /// As [`Self::load_video`].
    fn load_stages(&self, weights: Weights<'_>, sized: bool) -> Result<Arc<dyn VideoModel>> {
        let opened = self.open_video_checkpoint()?;
        let checkpoint = self.open_video_family(&opened)?;
        let (denoiser, tensor) = self.video_split(&*checkpoint)?;
        let stages = match sized {
            true => self.profiled_stages(checkpoint.denoisers())?,
            false => self.stages()?,
        };

        let mut loader = weights.loader(&opened);
        let placement = VideoPlacement {
            stages,
            denoiser,
            tensor,
            dtype: self.dtype(),
            serving: self.video_serving()?,
        };
        Ok(checkpoint.load(&opened, &mut loader, &placement)?)
    }

    /// The stages a rank's profile run loads: those its group places on it ([`Self::stages`]),
    /// and on a rank that denoises or decodes the stages whose declarations size what it
    /// computes, wherever the deployment places them: the deployment's denoiser, one of
    /// `denoisers`, whose layouts of the served requests a decoder's buckets follow, and the two
    /// decoders, whose rasters, splices and waveforms a denoising rank sizes the decoded products
    /// by (`Capacity::media_bytes`). `None` when the deployment places the whole pipeline.
    ///
    /// # Errors
    ///
    /// A rank its group does not place.
    fn profiled_stages(&self, denoisers: &[&str]) -> Result<Option<BTreeSet<String>>> {
        let Some(mut stages) = self.stages()? else {
            return Ok(None);
        };
        let sizes = self.media_phases()?.iter().any(|phase| {
            matches!(
                phase,
                Phase::Denoise | Phase::VideoDecode | Phase::AudioDecode
            )
        });
        if !sizes {
            return Ok(Some(stages));
        }

        let groups = &self.engine_config().parallel.worker_groups;
        let network = denoisers
            .iter()
            .find(|name| groups.iter().any(|group| group.blocks.contains_key(**name)));
        stages.extend(network.map(|name| (*name).to_owned()));
        stages.extend([Stage::VideoDecoder.name(), Stage::AudioDecoder.name()].map(str::to_owned));
        Ok(Some(stages))
    }

    /// The stages this rank's group places on it: those of the group's blocks that hold the
    /// rank's position (a block that states no ranks holds every one); `None` when the deployment
    /// places the whole model. A rank outside a block, such as a denoising rank beside a text
    /// encoder split over fewer ranks, loads none of that block's stage.
    ///
    /// A denoising rank also holds the condition encoders whenever the deployment places them
    /// (the latent encoder's block, in any group and on any ranks): its latent preparation
    /// encodes its requests' conditions (`specs/engine-worker-port.md` §5, D1), whichever ranks
    /// the block names.
    ///
    /// # Errors
    ///
    /// A rank its group does not place.
    fn stages(&self) -> Result<Option<BTreeSet<String>>> {
        let Some(group) = self.capability_groups().and(self.worker_group()) else {
            return Ok(None);
        };
        let position = self.position()?;
        let held = group
            .blocks
            .iter()
            .filter(|(_, block)| block.ranks.is_empty() || block.ranks.contains(&position));
        let mut stages: BTreeSet<String> = held.map(|(name, _)| name.clone()).collect();

        let denoises = [Stage::Transformer, Stage::TransformerRef]
            .iter()
            .any(|stage| stages.contains(stage.name()));
        let encoder = Stage::LatentEncoder.name();
        let groups = &self.engine_config().parallel.worker_groups;
        if denoises
            && groups
                .iter()
                .any(|group| group.blocks.contains_key(encoder))
        {
            stages.insert(encoder.to_owned());
        }
        Ok(Some(stages))
    }

    /// The deployment's configuration of this rank's worker group.
    fn worker_group(&self) -> Option<&WorkerGroupConfig> {
        self.engine_config()
            .parallel
            .worker_groups
            .iter()
            .find(|group| group.id.0 == self.group)
    }

    /// This rank's position in its worker group.
    ///
    /// # Errors
    ///
    /// A rank the deployment's groups do not place.
    pub(crate) fn position(&self) -> Result<usize> {
        let groups = &self.engine_config().parallel.worker_groups;
        WorkerGroupConfig::place_rank(groups, self.rank as usize)
            .map(|(_, position)| position)
            .with_context(|| format!("the deployment's worker groups place no rank {}", self.rank))
    }

    /// The element type the deployment computes in, which the daemon resolved against the
    /// checkpoint before it wrote this build config (a `dtype` of `auto` never reaches a main
    /// process as `auto`).
    fn dtype(&self) -> DType {
        ops::abi::from_wire(self.engine_config().model.dtype.into())
    }

    /// The phases this rank serves (module documentation).
    ///
    /// # Errors
    ///
    /// A rank its group does not place.
    pub(crate) fn media_phases(&self) -> Result<Vec<Phase>> {
        let group = self
            .worker_group()
            .with_context(|| format!("the deployment states no worker group {}", self.group))?;
        Ok(served_phases(group, self.position()?))
    }

    /// The parallel groups of the deployment's media groups over the process's host group, whose
    /// members are the deployment's ranks: the text encoders' tensor groups; the denoisers'
    /// groups along each axis of their partition ([`DENOISER_AXES`], `nn::family::ULYSSES_AXIS`
    /// and after), each a fiber of the denoiser block's places along that degree
    /// (`nn::family::DenoiserSplit` orders them); and the video decoders' ranks, which pass a
    /// round's splice tails between them (`crate::runner::decoder`). A block's ranks are any
    /// positions of its worker group, the block's rank `r` being its `r`-th, and a group lists
    /// its members in the order of their coordinate. Each kind holds this rank's place among
    /// its groups. Every rank forms all six kinds, in this order, so every member of the host
    /// group takes part in every group's id broadcast.
    ///
    /// # Errors
    ///
    /// A split block that states another number of ranks than its parallel degrees' product, or
    /// a rank twice.
    pub(crate) fn media_collectives(&self) -> Result<[super::Collective; 6]> {
        let groups = &self.engine_config().parallel.worker_groups;
        let firsts = WorkerGroupConfig::first_ranks(groups);
        let text = Stage::TextEncoder.name();
        let denoisers = [Stage::Transformer.name(), Stage::TransformerRef.name()];
        let own = (|| Some((self.worker_group()?, self.position().ok()?)))();

        let mut kinds: [super::Collective; 6] = std::array::from_fn(|_| super::Collective {
            groups: Vec::new(),
            member: None,
        });
        for (first, group) in firsts.iter().zip(groups) {
            for (name, block) in &group.blocks {
                // Each kind's members as places of the block: one list per group of the kind.
                let (kind, fibers) = if name == text {
                    let size = block.tensor_parallel_size;
                    if size > 1 && block.ranks.len() != size {
                        bail!(
                            "block {name} of worker group {} splits over its ranks {:?}, which \
                             are not {size} distinct ranks of the group",
                            group.id,
                            block.ranks
                        );
                    }
                    (0, vec![(0..size).collect::<Vec<_>>()])
                } else if name == VIDEO_DECODER_BLOCK {
                    // Every rank a decode round is dealt to (`ModelRunner::round`).
                    (5, vec![(0..block.ranks.len()).collect()])
                } else if denoisers.contains(&name.as_str()) {
                    for (axis, fibers) in denoiser_fibers(block)
                        .map_err(|refusal| {
                            anyhow::anyhow!("block {name} of worker group {}: {refusal}", group.id)
                        })?
                        .into_iter()
                        .enumerate()
                    {
                        let collective = &mut kinds[1 + axis];
                        add_fibers(collective, (group, *first, block), fibers, own)?;
                    }
                    continue;
                } else {
                    continue;
                };
                add_fibers(&mut kinds[kind], (group, *first, block), fibers, own)?;
            }
        }
        Ok(kinds)
    }

    /// How a video decode round is dealt to the ranks of the group that decodes this rank's
    /// requests: its video decoder block's ranks in order, by their rank index in the process's
    /// media memory (the deployment's rank), each decoding the block's units per rank when it
    /// deals units and one otherwise, the rounds alternately in that order and in reverse
    /// (`RoundConfig::rank_of`). Its width is the scheduler's (`crate::scheduler::video`). A group
    /// that places no video decoder block decodes a round of one unit on its own rank.
    ///
    /// # Errors
    ///
    /// No group that places the rank, or a rank number past the protocol's.
    pub(crate) fn round(&self) -> Result<RoundConfig> {
        let groups = &self.engine_config().parallel.worker_groups;
        let firsts = WorkerGroupConfig::first_ranks(groups);
        let own = groups.iter().position(|group| group.id.0 == self.group);
        let decoder = own
            .filter(|at| groups[*at].blocks.contains_key(VIDEO_DECODER_BLOCK))
            .or_else(|| {
                groups
                    .iter()
                    .position(|group| group.blocks.contains_key(VIDEO_DECODER_BLOCK))
            });

        let Some(at) = decoder else {
            return Ok(RoundConfig {
                ranks: vec![self.rank],
                units: 1,
            });
        };
        let block = &groups[at].blocks[VIDEO_DECODER_BLOCK];
        let ranks = block
            .ranks
            .iter()
            .map(|position| u32::try_from(firsts[at] + position))
            .collect::<Result<Vec<u32>, _>>()
            .context("a rank past what the protocol numbers")?;
        let units = block.rank_units();
        Ok(RoundConfig {
            ranks,
            units: u32::try_from(units).context("more units per rank than a round numbers")?,
        })
    }

    /// The config of the rank's text encoder of `model`, writing the conditioning and reading the
    /// vision pixels of their classes in `classes` (the session's class of each product kind, in
    /// `crate::batch::product::KINDS` order): one captured encode bucket per request count up to
    /// the deployment's running requests and per prompt capacity the deployment captures, and
    /// for an encoder that takes images the prompts with vision blocks the deployment admits,
    /// whose eager encodes it profiles. The request counts are the deployment's whatever slots
    /// the division decides, as a causal runner's decode rows are (`ModelRunner::buckets`): an
    /// encode holds at most the group's slots whatever its buckets hold.
    ///
    /// # Errors
    ///
    /// Capacities the deployment states that are not strictly increasing or above its prompt
    /// length, more running requests than the protocol addresses, or prompts with vision blocks
    /// the model cannot state.
    pub(crate) fn encoder_config(
        &self,
        model: &dyn VideoModel,
        classes: &[u8; 10],
    ) -> Result<EncoderConfig> {
        let serving = self.video_serving()?;
        let requests = u32::try_from(self.engine_config().max_running_requests.get())
            .context("more running requests than the protocol addresses")?;
        let mut buckets = Vec::new();
        for capacity in &serving.text_capacities {
            for rows in 1..=requests {
                buckets.push(Bucket::prefill(rows, *capacity, Variant::Encode));
            }
        }

        // A checkpoint whose text encoder has no vision tower states no patch format, and takes
        // no image.
        let patches = model.format().ok().map(|format| format.patches);
        let warm = match patches {
            Some(_) => model.warm_prompts(&serving)?,
            None => Vec::new(),
        };

        // The group's first text encoder rank, by the deployment's numbering of ranks, whose
        // product buffer holds the conditioning (`backed_kinds`): the group's first rank is
        // this rank less its position.
        let group = self.worker_group().with_context(|| {
            format!("the deployment's worker groups place no rank {}", self.rank)
        })?;
        let position = self.position()?;
        let first = (0..group.ranks.len())
            .find(|&at| served_phases(group, at).contains(&Phase::TextEncode))
            .unwrap_or(position);
        let holder = (self.rank as usize)
            .checked_sub(position)
            .and_then(|start| u32::try_from(start + first).ok())
            .context("a text encoder rank past what the protocol numbers")?;

        Ok(EncoderConfig {
            buckets: Buckets::new(buckets),
            width: model.text_width(),
            dtype: self.dtype(),
            patches,
            conditioning: classes[product_kind(product::CONDITIONING)?],
            holder,
            pixels: classes[product_kind(product::VISION_PIXELS)?],
            warm,
        })
    }

    /// The config of the rank's denoiser of `model`: the requests the deployment serves, the
    /// classes it reads and writes, the ranks of its group's denoiser block, and the host latent
    /// region's page bytes. The block's first rank in each main process shares the latents its
    /// last step writes with that process's other backing ranks and writes a saved trajectory's
    /// host copy into the part 0 its process copies with: a block whose ranks lie on two hosts
    /// saves on both, and its ranks on each host restore through their own host's part.
    ///
    /// # Errors
    ///
    /// A deployment the pipeline serves no request of.
    pub(crate) fn denoiser_config(
        &self,
        model: &dyn VideoModel,
        classes: &[u8; 10],
    ) -> Result<DenoiserConfig> {
        let candidates = model.candidates(&self.video_serving()?)?;
        if candidates.is_empty() {
            bail!("this deployment serves no request layout, so a denoiser would capture nothing");
        }

        let (ranks, first) = match (self.denoiser_block(), self.worker_group()) {
            (Some(block), Some(group)) => {
                let position = self.position()?;
                let ranks = u32::try_from(block.ranks.len()).context("a block wider than u32")?;
                // The block's first rank on this rank's host, which its main process serves.
                let host = |at: usize| group.ranks.get(at).map(|rank| rank.host.as_str());
                let here = host(position);
                let first = block.ranks.iter().find(|at| host(**at) == here);
                (ranks, first == Some(&position))
            }
            _ => (1, true),
        };
        let pick = |name: &str| -> Result<u8> { Ok(classes[product_kind(name)?]) };
        Ok(DenoiserConfig {
            phases: vec![Phase::LatentPrepare, Phase::Denoise],
            candidates,
            stack: self.dtype(),
            sample: model.sample_type(),
            width: model.text_width(),
            classes: [
                pick(product::CONDITIONING)?,
                pick(product::CONDITION_PIXELS)?,
                pick(product::CONDITION_SAMPLES)?,
                pick(product::VIDEO_LATENTS)?,
                pick(product::AUDIO_LATENTS)?,
            ],
            ranks: ranks.max(1),
            page_bytes: latent_pages(0, 0)?.page_bytes()?,
            first,
        })
    }

    /// The denoiser block of this rank's group: the block of a denoiser stage; `None` for a
    /// group that places the whole model.
    fn denoiser_block(&self) -> Option<&proto::config::BlockConfig> {
        let group = self.worker_group()?;
        [Stage::Transformer, Stage::TransformerRef]
            .into_iter()
            .find_map(|stage| group.blocks.get(stage.name()))
    }

    /// The configs of the rank's video decoder and its track's decoder of `model`, each `None`
    /// when the rank serves no such decode: one clip bucket of the decoder's window per canvas
    /// the deployment serves, with the timelines of `laid`, the served requests with the
    /// denoiser's layouts of them ([`VideoRank::laid`]), dealt as the deployment's rounds; one
    /// track bucket per timeline whose layout carries a track.
    ///
    /// # Errors
    ///
    /// A decoding rank given no layout, or a pipeline whose decoders read latents in another
    /// element type than the trajectory's.
    pub(crate) fn decoder_configs(
        &self,
        model: &dyn VideoModel,
        laid: &[(VideoRequest, Layout)],
        classes: &[u8; 10],
    ) -> Result<(Option<DecoderConfig>, Option<DecoderConfig>)> {
        let phases = self.media_phases()?;
        let (video, audio) = (
            phases.contains(&Phase::VideoDecode),
            phases.contains(&Phase::AudioDecode),
        );
        if !video && !audio {
            return Ok((None, None));
        }
        if laid.is_empty() {
            bail!("the rank decodes and holds no layout of the requests the deployment serves");
        }

        check_latents(model)?;
        let window = model.decode_window();
        let (mut clips, mut timelines, mut tracks) = (Vec::new(), Vec::new(), Vec::new());
        let mut served: Vec<u32> = Vec::new();
        for (request, layout) in laid {
            let clip = Bucket::clip(window, request.height, request.width);
            if !clips.contains(&clip) {
                clips.push(clip);
            }
            let timeline = (request.frames, layout.video.frames);
            if !timelines.contains(&timeline) {
                timelines.push(timeline);
            }
            // A track's bucket serves its timeline at every resolution.
            if layout.audio.is_some() && !served.contains(&request.frames) {
                served.push(request.frames);
                tracks.push(Bucket::audio(1, *layout).serving(
                    request.frames,
                    request.width,
                    request.height,
                ));
            }
        }

        let pick = |name: &str| -> Result<u8> { Ok(classes[product_kind(name)?]) };
        let tail = pick(product::VIDEO_TAIL)?;
        let video = video
            .then(|| -> Result<DecoderConfig> {
                Ok(DecoderConfig {
                    phases: vec![Phase::VideoDecode],
                    buckets: Buckets::new(clips),
                    classes: [
                        pick(product::VIDEO_LATENTS)?,
                        pick(product::VIDEO_UNITS)?,
                        tail,
                    ],
                    timelines,
                    round: self.round()?,
                })
            })
            .transpose()?;
        // A pipeline whose layouts carry no track decodes none.
        let audio = (audio && !tracks.is_empty())
            .then(|| -> Result<DecoderConfig> {
                Ok(DecoderConfig {
                    phases: vec![Phase::AudioDecode],
                    buckets: Buckets::new(tracks),
                    classes: [
                        pick(product::AUDIO_LATENTS)?,
                        pick(product::AUDIO_SAMPLES)?,
                        tail,
                    ],
                    timelines: Vec::new(),
                    round: RoundConfig {
                        ranks: vec![self.rank],
                        units: 1,
                    },
                })
            })
            .transpose()?;
        Ok((video, audio))
    }

    /// The config of the rank's media host of `model`, serving `phases`, its host phases: every
    /// product class, the timelines the deployment serves with their units, the decode rounds'
    /// deal, and the model's rates and media format.
    ///
    /// The timelines are every legal timeline from the shortest to the longest the deployment
    /// serves, which the model's clip schedule states whatever the rank places: a muxer or a
    /// media reader in a group of its own places no denoiser, and serves the timelines the
    /// denoising ranks serve.
    ///
    /// # Errors
    ///
    /// A duration the pipeline serves no timeline of, or a model that states no media format.
    pub(crate) fn host_config(
        &self,
        model: &dyn VideoModel,
        classes: &[u8; 10],
        phases: Vec<Phase>,
    ) -> Result<HostConfig> {
        // The shortest timeline is the least that lasts one frame. A frame count between the
        // two that is no legal timeline is one `video_units` refuses.
        let shortest = model.timeline(1.0 / f64::from(model.frame_rate().max(1)))?;
        let longest = model.timeline(self.video_serving()?.max_seconds)?;
        let timelines: Vec<(u32, Vec<u32>)> = (shortest..=longest)
            .filter_map(|frames| Some((frames, model.video_units(frames).ok()?)))
            .collect();

        Ok(HostConfig {
            phases,
            classes: product::KINDS
                .iter()
                .map(|(name, ..)| *name)
                .zip(classes.iter().copied())
                .collect(),
            timelines,
            round: self.round()?,
            frame_rate: model.frame_rate(),
            audio_rate: model.audio_rate(),
            format: model
                .format()
                .context("the media host lays condition media out in the model's format")?,
        })
    }

    /// The runners of this rank's device blocks over `memory`, the rank's media memory, each over
    /// the graphs and stream `graphs` gives it for its id and its input bytes: on the meta device
    /// graphs that hold none, on a device its region of the graph pool. The text encoder computes
    /// over the tensor group `tensor`, the denoiser over its groups `denoiser` along each axis of
    /// its partition, each at this rank's place (meta graphs take the splits alone), and the
    /// video decoder passes its rounds' tails over the communicator `decoder`; the decoders
    /// bucket by `laid` ([`VideoRank::laid`]). The media host is [`ModelRunner::media_host`]'s.
    ///
    /// # Errors
    ///
    /// A config the model refuses, or a runner that cannot be built over its graphs.
    pub(crate) fn device_runners(
        &self,
        model: &Arc<dyn VideoModel>,
        memory: &Arc<BufferMemory>,
        mut graphs: impl FnMut(RunnerId, u64) -> Result<(Graphs, Option<Stream>)>,
        (tensor, denoiser, decoder): (
            Option<Arc<ops::Group>>,
            DenoiserGroups,
            Option<Communicator>,
        ),
        (laid, classes): (&[(VideoRequest, Layout)], &[u8; 10]),
    ) -> Result<DeviceRunners> {
        let phases = self.media_phases()?;
        let (partition, text_split) = self.partition()?;

        let encoder = if phases.contains(&Phase::TextEncode) {
            let text = model.text_encoder().context(
                "the rank encodes prompts and its share of the checkpoint holds no text encoder",
            )?;
            let config = self.encoder_config(model.as_ref(), classes)?;
            let (graphs, stream) = graphs(ENCODER, EncoderRunner::input_bytes(&config))?;
            let (rank, ranks) = super::split(text_split.rank as usize, text_split.size as usize);
            graphs.set_group(tensor, rank, ranks)?;
            let runner = EncoderRunner::new(text, config, graphs, stream, Arc::clone(memory))
                .context("building the rank's text encoder")?;
            Some(runner)
        } else {
            None
        };

        let denoiser = if phases.contains(&Phase::Denoise) {
            let network = model
                .denoiser()
                .context("the rank denoises and its share of the checkpoint holds no denoiser")?;
            let config = self.denoiser_config(model.as_ref(), classes)?;
            let (graphs, stream) = graphs(DENOISER, 0)?;
            // The axes in the order the model addresses them (`nn::family::ULYSSES_AXIS` and
            // after).
            let places = [
                partition.ulysses.split,
                partition.gather,
                partition.tensor,
                partition.layers,
            ];
            let axes = denoiser
                .into_iter()
                .zip(places)
                .map(|(group, place)| {
                    let (rank, ranks) = super::split(place.rank as usize, place.size as usize);
                    (group, rank, ranks)
                })
                .collect();
            graphs.set_axes(axes)?;
            let runner = DenoiserRunner::new(network, config, graphs, stream, Arc::clone(memory))
                .context("building the rank's denoiser")?;
            Some(runner)
        } else {
            None
        };

        let (video, audio) = self.decoder_configs(model.as_ref(), laid, classes)?;
        let video = match video {
            Some(config) => {
                let network = model.video_decoder().context(
                    "the rank decodes video and its share of the checkpoint holds no video decoder",
                )?;
                let (graphs, stream) = graphs(VIDEO_DECODER, 0)?;
                let splicer = model.splicer();
                let runner = DecoderRunner::new(
                    network,
                    splicer,
                    config,
                    graphs,
                    (stream, decoder),
                    Arc::clone(memory),
                )
                .context("building the rank's video decoder")?;
                Some(runner)
            }
            None => None,
        };
        let audio = match audio {
            Some(config) => {
                let network = model.audio_decoder().context(
                    "the denoiser's layouts carry a track and the rank's share of the checkpoint \
                     holds no audio decoder",
                )?;
                let (graphs, stream) = graphs(AUDIO_DECODER, 0)?;
                let runner = DecoderRunner::new(
                    network,
                    None,
                    config,
                    graphs,
                    (stream, None),
                    Arc::clone(memory),
                )
                .context("building the rank's track decoder")?;
                Some(runner)
            }
            None => None,
        };
        Ok((encoder, denoiser, video, audio))
    }

    /// The media host of this rank's host phases over `memory`, issuing its copies on `stream`
    /// and running its work on `pool` with `media`, the process's one pool and media runner;
    /// `None` on a rank that serves no host phase.
    ///
    /// # Errors
    ///
    /// As [`ModelRunner::host_config`] and `MediaHost::new` refuse.
    pub(crate) fn media_host(
        &self,
        model: &dyn VideoModel,
        (memory, stream): (&Arc<BufferMemory>, Option<Stream>),
        (pool, media): (&Arc<ThreadPool>, &Arc<media::MediaRunner>),
        kernels: &Arc<Generation>,
        classes: &[u8; 10],
    ) -> Result<Option<MediaHost>> {
        let phases: Vec<Phase> = self
            .media_phases()?
            .into_iter()
            .filter(|phase| HOST_PHASES.contains(phase))
            .collect();
        if phases.is_empty() {
            return Ok(None);
        }

        let config = self.host_config(model, classes, phases)?;
        let host = MediaHost::new(
            config,
            Arc::clone(kernels),
            Arc::clone(memory),
            stream,
            Arc::clone(pool),
            Arc::clone(media),
        )
        .context("building the rank's media host")?;
        Ok(Some(host))
    }

    /// Profiles a media rank on the meta device (steps 7-8 of `specs/engine-worker-port.md`
    /// §1.1) over `model`, the meta load of its profile ([`ModelRunner::load_profiled`]): its
    /// runners over meta graphs and a media memory that holds no device, and each runner's
    /// profile run. Returns the rank as its opening and its statement read it, with the parallel
    /// groups `groups` it computes over: the runners' peaks and buffers, a slot's row, the
    /// layouts its decoders bucket by, on a denoising rank the trajectory class and its group's
    /// product sizes, and the rank's capacity with the media pool's `pool_tasks`; and every
    /// kernel the runners launch.
    ///
    /// # Errors
    ///
    /// What a runner or a forward of a profile run refuses, or a deployment that places no
    /// denoiser or decoder whose declarations the rank sizes by.
    pub(crate) fn profile_media(
        &self,
        model: Arc<dyn VideoModel>,
        groups: (Option<Arc<ops::Group>>, DenoiserGroups),
        pool_tasks: u32,
    ) -> Result<(VideoRank, Keys)> {
        let classes = product_classes(0);
        let slots = self.slots();
        let row = self.row_bytes(model.as_ref(), &classes)?;

        // The served requests' layouts, which a decoding rank buckets by and a denoising rank
        // sizes its products by.
        let phases = self.media_phases()?;
        let sizes = [Phase::Denoise, Phase::VideoDecode, Phase::AudioDecode]
            .iter()
            .any(|phase| phases.contains(phase));
        let laid = match sizes {
            true => laid_out(model.as_ref(), &self.video_serving()?)?,
            false => Vec::new(),
        };

        // The region numbers the profile's memory states are none of a forward's concern: the
        // session's answer numbers the rank's memory when it opens.
        let memory = Arc::new(BufferMemory::meta(self.rank, (1, 1), (0, slots), row, &[])?);
        let unserved = Arc::new(Generation::new(0, []));
        let (encoder, denoiser, video, audio) = self.device_runners(
            &model,
            &memory,
            |_, inputs| Ok((Graphs::meta(Arc::clone(&unserved), inputs), None)),
            (None, Default::default(), None),
            (&laid, &classes),
        )?;

        // What the runners hold beside their regions, and on a denoising rank the trajectory
        // class and its group's product sizes.
        let mut buffers: u64 = video
            .iter()
            .chain(&audio)
            .map(DecoderRunner::device_bytes)
            .sum();
        let (mut trajectory, mut media_bytes, mut units) = (None, Vec::new(), 0);
        let mut description = Vec::new();
        if let Some(denoiser) = &denoiser {
            buffers += denoiser.device_bytes();
            description = model.description(&self.video_serving()?)?.details;
            let (tensors, encoding) = denoiser.trajectory_digests();
            let shapes = denoiser.media_shapes()?;
            let largest = shapes.iter().map(|(_, bytes, _)| *bytes).max().unwrap_or(0);
            trajectory = Some((tensors, encoding, latent_pages(largest, slots)?));
            (media_bytes, units) = self.media_bytes(model.as_ref(), &shapes, &laid)?;
        }

        let runners = boxed((encoder, denoiser, video, audio));
        let mut keys = Keys::new();
        let mut peaks = Vec::with_capacity(runners.len());
        for (id, runner) in &runners {
            let graphs = runner.graphs();
            graphs
                .profile_run(runner.as_ref())
                .with_context(|| format!("profiling the rank's runner {}", id.0))?;
            keys.extend(graphs.keys());
            // A media runner prepares no inputs of a paired forward
            // (`Runner::prepare_shadow_inputs` refuses), so it is never shadowed and its rank's
            // scratch pool holds no shadow region for it.
            peaks.push((
                *id,
                Peaks {
                    graph: graphs.peak(),
                    shadow: 0,
                    inputs: graphs.input_peak(),
                },
            ));
        }

        let capacity = self.media_capacity(model.as_ref(), &runners, media_bytes, pool_tasks)?;
        let (tensor, denoiser) = groups;
        let group_slots = vec![slots; self.engine_config().parallel.worker_groups.len()];
        Ok((
            VideoRank {
                model,
                tensor,
                denoiser,
                decoder: None,
                peaks,
                buffers,
                row,
                units,
                laid,
                trajectory,
                description,
                capacity,
                granule: UNIT_BYTES,
                slot_units: 0,
                group_slots,
                host: None,
                memory: None,
                media: None,
                regions: (Vec::new(), Vec::new()),
                runners: Vec::new(),
            },
            keys,
        ))
    }

    /// The bytes of one slot's row of the rank's trajectory rows: the state of the largest
    /// request its denoiser serves (`DenoiserRunner::row_bytes`); 0 on a rank that does not
    /// denoise.
    ///
    /// # Errors
    ///
    /// A denoising rank whose share holds no denoiser, or a layout it refuses.
    pub(crate) fn row_bytes(&self, model: &dyn VideoModel, classes: &[u8; 10]) -> Result<u64> {
        if !self.media_phases()?.contains(&Phase::Denoise) {
            return Ok(0);
        }
        let denoiser = model
            .denoiser()
            .context("the rank denoises and its share of the checkpoint holds no denoiser")?;
        let config = self.denoiser_config(model, classes)?;
        Ok(DenoiserRunner::row_bytes(denoiser.as_ref(), &config)?)
    }

    /// Per captured request shape of the rank's denoiser (`DenoiserRunner::media_shapes`): the
    /// shape, the bytes of a slot's trajectory, and the bytes of a block of each product kind in
    /// `crate::batch::product::KINDS` order (`Capacity::media_bytes`); and the most units a
    /// timeline of the deployment has, the splice tails a request declares. `model` is the
    /// profile's meta load, which holds the decoders wherever the deployment places them
    /// ([`ModelRunner::load_profiled`]), and `laid` the served requests with their layouts.
    ///
    /// - The condition pixels, samples and vision pixels of a shape with condition rows: the
    ///   bounds the model states of one request's conditions (`ConditionBounds`), as the media
    ///   host lays them out in FP32 and the vision tower's element type; none for a shape
    ///   without.
    /// - The conditioning: the shape's prompt rows of the text encoder's width.
    /// - The latents: the denoiser's.
    /// - The video units: the timeline's frames as packed RGB bytes, from its last splice's end;
    ///   without a splicer, each unit's raster in the decoder's element type.
    /// - A splice tail: the most frames a unit of the timeline fades against, in the decoder's
    ///   element type; none without a splicer.
    /// - An encoded unit: the media host's row of the timeline's longest unit, one block per
    ///   unit.
    /// - The audio samples: the track's PCM window, stereo `i16`; none without a track.
    ///
    /// # Errors
    ///
    /// A pipeline with no video decoder, or with no audio decoder for a layout with a track; a
    /// pipeline whose decoders read latents in another element type than the trajectory's; a
    /// shape no candidate of the deployment lays out.
    #[expect(
        clippy::type_complexity,
        reason = "`Capacity::media_bytes` as the scheduler reads it, and the tails a request declares"
    )]
    fn media_bytes(
        &self,
        model: &dyn VideoModel,
        shapes: &[(Shape, u64, [u64; 2])],
        laid: &[(VideoRequest, Layout)],
    ) -> Result<(Vec<(Shape, u64, Vec<u64>)>, u32)> {
        let decoder = model.video_decoder().context(
            "the denoising rank sizes the decoded products, and the deployment places no video \
             decoder",
        )?;
        check_latents(model)?;
        let track = model.audio_decoder();

        let bounds = model.condition_bounds();
        let format = model.format().ok();
        let splicer = model.splicer();
        let stack = element(self.dtype())?;
        let raster = element(decoder.element_type())?;
        let window = u64::from(model.decode_window());

        let mut most_units = 0;
        let mut sized = Vec::with_capacity(shapes.len());
        for (shape, trajectory, [video, audio]) in shapes {
            let Extent::Request {
                frames,
                width,
                height,
                text_rows,
                condition_rows,
            } = shape.extent
            else {
                bail!("the denoiser states a captured shape of no request: {shape:?}");
            };
            let (_, layout) = laid
                .iter()
                .find(|(request, layout)| {
                    (request.frames, request.width, request.height) == (frames, width, height)
                        && (layout.text_rows, layout.condition_rows) == (text_rows, condition_rows)
                })
                .with_context(|| format!("no candidate of the deployment lays out {shape:?}"))?;

            let units = model.video_units(frames)?;
            most_units = most_units.max(u32::try_from(units.len()).unwrap_or(u32::MAX));
            let pixels = u64::from(width) * u64::from(height);

            // The condition products, bounded by what one request's conditions may reach.
            let conditioned = condition_rows > 0;
            let (condition_pixels, condition_samples, vision_pixels) = match (&format, conditioned)
            {
                (Some(format), true) => {
                    let hop = format.hop.max(1);
                    let samples = u64::from(bounds.samples).div_ceil(hop) * hop;
                    let patch = u64::from(format.patches.width()) * element(format.patches.dtype)?;
                    (
                        u64::from(bounds.pixels) * 3 * 4,
                        samples * 2 * 4,
                        u64::from(text_rows) * u64::from(bounds.patches_per_token) * patch,
                    )
                }
                _ => (0, 0, 0),
            };
            let conditioning = u64::from(text_rows) * u64::from(model.text_width()) * stack;

            // The decoded media of the timeline and the tail each unit leaves its successor.
            let (video_units, tail) = match &splicer {
                Some(splicer) => {
                    let splices = splicer.splices(frames)?;
                    let end = splices
                        .iter()
                        .map(|splice| u64::from(splice.first) + u64::from(splice.keep))
                        .max()
                        .unwrap_or(0)
                        .max(u64::from(frames));
                    let overlap = splices.iter().map(|splice| splice.overlap).max();
                    (
                        end * pixels * 3,
                        u64::from(overlap.unwrap_or(0)) * 3 * pixels * raster,
                    )
                }
                None => (units.len() as u64 * 3 * window * pixels * raster, 0),
            };
            let encoded = MediaHost::encoded_bytes(&units, height, width);
            let samples = match (&layout.audio, &track) {
                (Some(audio), Some(track)) => {
                    let (samples, channels) = track.pcm_window(audio)?;
                    samples * u64::from(channels) * 2
                }
                (Some(_), None) => bail!(
                    "the denoiser's layouts carry a track and the deployment places no audio \
                     decoder"
                ),
                (None, _) => 0,
            };

            sized.push((
                *shape,
                *trajectory,
                vec![
                    condition_pixels,
                    condition_samples,
                    vision_pixels,
                    conditioning,
                    *video,
                    *audio,
                    video_units,
                    tail,
                    encoded,
                    samples,
                ],
            ));
        }
        Ok((sized, most_units))
    }

    /// What this rank states of its worker group (`specs/scheduler-design.md` §2.1): the phases
    /// it serves; per phase the requests its stage holds at once (the slots of an encode, one
    /// request of a denoiser or a decoder, the tasks of the process's media pool of
    /// `pool_tasks` for a host phase); the shapes its runners `runners` compute, the host
    /// buckets of its host phases; and on a denoising rank the denoising steps and its group's
    /// product sizes `media_bytes`. A media rank keeps no KV cache and encodes no chat image.
    /// The ranks of a group state their own phases, which the group's statement joins
    /// (`crate::process::ranks::capacities`).
    ///
    /// # Errors
    ///
    /// A queue depth of none, or a pipeline that describes no denoiser it places.
    pub(crate) fn media_capacity(
        &self,
        model: &dyn VideoModel,
        runners: &[(RunnerId, Box<dyn Runner>)],
        media_bytes: Vec<(Shape, u64, Vec<u64>)>,
        pool_tasks: u32,
    ) -> Result<Capacity> {
        let phases = self.media_phases()?;
        let slots = self.slots();

        let mut served = 0;
        let mut rows = [0; Phase::COUNT];
        for phase in &phases {
            served |= 1 << phase.index();
            rows[phase.index()] = phase_rows(*phase, slots, pool_tasks);
        }

        // The shapes a batch of a phase it serves is formed against.
        let mut buckets: Vec<Shape> = Vec::new();
        let shapes = runners
            .iter()
            .flat_map(|(_, runner)| runner.buckets().into_iter().chain(runner.eager_buckets()))
            .map(Bucket::shape)
            .chain(
                phases
                    .iter()
                    .filter(|phase| HOST_PHASES.contains(phase))
                    .filter_map(|phase| {
                        variants(*phase)
                            .first()
                            .map(|variant| Bucket::host(*variant).shape())
                    }),
            );
        for shape in shapes {
            let stated = phases
                .iter()
                .any(|phase| variants(*phase).contains(&shape.variant));
            if stated && !buckets.contains(&shape) {
                buckets.push(shape);
            }
        }

        let denoise_steps = if phases.contains(&Phase::Denoise) {
            model.description(&self.video_serving()?)?.denoise_steps
        } else {
            0
        };
        let calls = buckets.iter().map(|shape| shape.rows).max().unwrap_or(1);
        let window = call_window(self.queue_depth(), calls.max(1) as usize)
            .context("bounding the calls of a request the group holds unresolved")?;
        let config = &self.engine_config().model;

        Ok(Capacity {
            phases: served,
            slots,
            kv_cache: None,
            cache_groups: Vec::new(),
            seeds: Vec::new(),
            buckets,
            rows,
            max_model_len: config.max_model_len,
            denoise_steps,
            bos_token_id: config.bos,
            eos_token_ids: config.eos.clone(),
            feature_bytes: 0,
            // The group's encoder identity and latent seed are stated from its ranks' weights by
            // the rank start (`crate::process::ranks::capacities`).
            encoder_identity: String::new(),
            latent_seed: 0,
            media_bytes,
            call_window: u32::try_from(window).unwrap_or(u32::MAX),
        })
    }

    /// Checks that a media rank's regions on `device` fit what its GPU's quota leaves it once its
    /// weights are mapped, within its own share of the GPU (steps 11-14): the graph and scratch
    /// pools `pools` of its runners' regions, its media memory `resident` from the resident
    /// pool, and the `buffers` its runners hold beside them. A media rank keeps no cache, so
    /// nothing takes the remainder, and nothing of it waits for another epoch to drain.
    ///
    /// # Errors
    ///
    /// The Storage Server's or the driver's refusal, or the region the quota cannot hold.
    pub(crate) fn media_division(
        &self,
        client: &StorageClient,
        device: DeviceId,
        (graph, scratch): (u64, u64),
        resident: u64,
        buffers: u64,
    ) -> Result<()> {
        let (share, _) = self.fractions()?;
        let (unreserved, _) = unreserved(client, device, share)?;
        division::remainder(
            unreserved,
            &[
                ("graph pool", graph),
                ("scratch pool", scratch),
                ("media memory", resident),
                ("runner buffers", buffers),
            ],
            0,
            &[],
        )?;
        Ok(())
    }

    /// The bytes a media rank's share of its GPU leaves for its request slots on `device` once its
    /// weights are mapped (steps 11-14): what the GPU's quota holds now within the rank's own
    /// share, less the graph and scratch pools of its runners' regions `peaks` and the `buffers`
    /// its runners hold beside them, none of which grows with the slots.
    ///
    /// # Errors
    ///
    /// The Storage Server's or the driver's refusal, or a pool the quota cannot hold.
    pub(crate) fn media_left(
        &self,
        client: &StorageClient,
        device: DeviceId,
        peaks: &[(RunnerId, Peaks)],
        buffers: u64,
    ) -> Result<u64> {
        let granularity = granularity(device)?;
        let graphs: Vec<u64> = peaks.iter().map(|(_, peaks)| peaks.graph).collect();
        let shadows: Vec<u64> = peaks.iter().map(|(_, peaks)| peaks.shadow).collect();
        let (graph, scratch) = division::pool_bytes(granularity, &graphs, &shadows, 0);

        let (share, _) = self.fractions()?;
        let (unreserved, _) = unreserved(client, device, share)?;
        Ok(division::remainder(
            unreserved,
            &[
                ("graph pool", graph),
                ("scratch pool", scratch),
                ("runner buffers", buffers),
            ],
            0,
            &[],
        )?)
    }

    /// The digests the class of a video request's conditioning states (`storage::runtime::Class`)
    /// for `model`: its tensors, rows of the text encoder's width in the deployment's element
    /// type, and its encoding, an output's bytes in row order over the host encoder region's
    /// units, which every parallel configuration lays alike (`crate::memory::encoders`).
    pub(crate) fn conditioning_digests(&self, model: &dyn VideoModel) -> ([u8; 32], [u8; 32]) {
        let mut tensors = storage::key::Digest::new(b"uniserve.conditioning");
        tensors.text(product::CONDITIONING);
        tensors.text(ops::abi::name(self.dtype()));
        tensors.number(u64::from(model.text_width()));
        (tensors.finish(), encoders::layout())
    }

    /// What this video deployment serves: its sizes, its longest output, the prompt capacities it
    /// captures and the condition rows a request may take.
    pub(super) fn video_serving(&self) -> Result<VideoServing> {
        let video = &self.engine_config().video;
        let longest = self.engine_config().model.max_model_len;
        let sizes = video
            .sizes
            .as_deref()
            .context("the leader states a video deployment's sizes")?
            .iter()
            .map(|size| {
                size.split_once('x')
                    .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
                    .with_context(|| format!("the deployment's size {size:?} is not WxH"))
            })
            .collect::<Result<Vec<(u32, u32)>>>()?;
        Ok(VideoServing {
            sizes,
            max_seconds: video
                .max_seconds
                .context("the leader states a video deployment's longest output")?,
            text_capacities: prompt_capacities(video, longest)?,
            condition_rows: video.max_condition_rows,
        })
    }
}

/// The phases position `position` of `group` serves (module documentation): each block whose
/// ranks hold the position names its phases; the audio decoder's track is one unit, dealt to
/// the block's first rank; a block of no phase (the latent encoder, whose work the latent
/// preparation computes) names none.
pub(crate) fn served_phases(group: &WorkerGroupConfig, position: usize) -> Vec<Phase> {
    let mut phases: Vec<Phase> = Vec::new();
    for (name, block) in &group.blocks {
        let holds = block.ranks.contains(&position);
        let named: &[Phase] = match name.as_str() {
            DEFAULT_BLOCK => &MEDIA_PHASES,
            name if name == Stage::TextEncoder.name() => &[Phase::TextEncode],
            name if name == Stage::Transformer.name() || name == Stage::TransformerRef.name() => {
                &[Phase::LatentPrepare, Phase::Denoise]
            }
            VIDEO_DECODER_BLOCK => &[Phase::VideoDecode],
            name if name == Stage::AudioDecoder.name() => {
                if block.ranks.first() != Some(&position) {
                    continue;
                }
                &[Phase::AudioDecode]
            }
            MEDIA_READER => &[Phase::MediaRead],
            VIDEO_CODEC_BLOCK => &[Phase::VideoEncode],
            MUXER => &[Phase::AudioEncode, Phase::Mux],
            _ => &[],
        };
        if holds {
            for phase in named {
                if !phases.contains(phase) {
                    phases.push(*phase);
                }
            }
        }
    }
    phases.sort_by_key(|phase| phase.index());
    phases
}

/// The axes of a denoiser's partition whose groups a denoising rank's runner computes over, in
/// the order `nn::family::DenoiserSplit` and the runner's graphs state them: the Ulysses group,
/// the all-gather group, the tensor group and the stages of the layer split.
pub(crate) const DENOISER_AXES: usize = 4;

/// A denoising rank's group along each axis of its denoiser's partition ([`DENOISER_AXES`]),
/// `None` on an axis of one rank and on the meta device.
pub(crate) type DenoiserGroups = [Option<Arc<ops::Group>>; DENOISER_AXES];

/// The groups of a denoiser block along each axis of its partition ([`DENOISER_AXES`]), as
/// places of the block: per axis, one list per fiber along it, its members in the order of their
/// coordinate in the block's order (`placement::denoiser_axes`).
///
/// # Errors
///
/// A degree of zero, or a block whose ranks do not number its degrees' product.
fn denoiser_fibers(block: &proto::config::BlockConfig) -> Result<[Vec<Vec<usize>>; DENOISER_AXES]> {
    // Each axis's degree and the distance between two consecutive coordinates of it.
    let (degrees, strides) = super::placement::denoiser_axes(block);
    let [ulysses, gather, tensor, layers] = degrees;
    let width: usize = degrees.iter().product();
    if degrees.contains(&0) || width != block.ranks.len() {
        bail!(
            "its {} ranks are not the product of its degrees (Ulysses {ulysses}, all-gather \
             {gather}, tensor {tensor}, layers {layers})",
            block.ranks.len()
        );
    }

    Ok(std::array::from_fn(|axis| {
        let (degree, stride) = (degrees[axis], strides[axis]);
        if degree <= 1 {
            return Vec::new();
        }
        // A fiber starts at every place whose coordinate on the axis is zero.
        (0..width)
            .filter(|place| (place / stride) % degree == 0)
            .map(|start| (0..degree).map(|step| start + step * stride).collect())
            .collect()
    }))
}

/// Adds the groups `fibers` of `block` of worker group `group`, whose first rank is `first`, to
/// `collective`: each fiber's members as places of the block, in its order, mapped to the
/// deployment's ranks; and this rank's place when `own` (its group and its position there) is a
/// member. A fiber of one member forms no group.
///
/// # Errors
///
/// A block whose ranks repeat, or a place past them.
fn add_fibers(
    collective: &mut super::Collective,
    (group, first, block): (&WorkerGroupConfig, usize, &proto::config::BlockConfig),
    fibers: Vec<Vec<usize>>,
    own: Option<(&WorkerGroupConfig, usize)>,
) -> Result<()> {
    let distinct: BTreeSet<usize> = block.ranks.iter().copied().collect();
    if distinct.len() != block.ranks.len() {
        bail!(
            "a block of worker group {} lists a rank twice: {:?}",
            group.id,
            block.ranks
        );
    }

    for fiber in fibers {
        if fiber.len() <= 1 {
            continue;
        }
        let positions = fiber
            .iter()
            .map(|place| block.ranks.get(*place).copied())
            .collect::<Option<Vec<usize>>>()
            .with_context(|| format!("a place past the ranks {:?}", block.ranks))?;

        let index = collective.groups.len();
        collective
            .groups
            .push(positions.iter().map(|position| first + position).collect());
        if let Some((own, position)) = own
            && own.id == group.id
            && let Some(place) = positions.iter().position(|held| *held == position)
        {
            collective.member = Some((index, place));
        }
    }
    Ok(())
}

/// The product kinds the rank at `position` of `group` backs, as indices into
/// `crate::batch::product::KINDS`: those its group's phases write, the conditioning only on the
/// group's first text encoder rank, an encoder output being one rank's.
///
/// # Errors
///
/// A session that numbers no conditioning kind.
pub(crate) fn backed_kinds(group: &WorkerGroupConfig, position: usize) -> Result<Vec<usize>> {
    let conditioning = product_kind(product::CONDITIONING)?;
    let phases = group_phases(group);
    let encoder = (0..group.ranks.len())
        .find(|position| served_phases(group, *position).contains(&Phase::TextEncode));

    Ok((0..product::KINDS.len())
        .filter(|&kind| {
            phases.contains(&product::KINDS[kind].1)
                && (kind != conditioning || encoder == Some(position))
        })
        .collect())
}

/// A rank of a media deployment: the index of its worker group in the deployment and its
/// position in that group.
pub(crate) type MediaRank = (usize, usize);

/// Where one worker group finds the blocks of one product kind: the ranks whose product buffers
/// its blocks of the kind lie in (`storage::runtime::Group::regions`), then the ranks whose
/// product buffers receive them in the other memories it reads the kind in
/// (`storage::runtime::Group::receives`).
pub(crate) type KindRanks = (Vec<MediaRank>, Vec<MediaRank>);

/// Where the blocks of each product kind lie for each worker group of the media deployment whose
/// groups are `groups` and whose `data_parallel` schedulers each serve a block of them
/// (`WorkerGroupConfig::scheduler_blocks`): per group, in `crate::batch::product::KINDS` order,
/// its [`KindRanks`].
///
/// A block's units exist once, in the product buffers of the ranks that back its kind, and each
/// main process's ranks reach the product buffers of that process alone
/// (`BufferMemory::connect`): a memory is the process of one host.
///
/// - A group whose phases write the kind takes its blocks from the product buffers of its ranks
///   that back it ([`backed_kinds`]).
/// - A group that reads the kind and does not write it reads the blocks of the groups of its
///   scheduler that write it, in the product buffers of their backing ranks on the hosts of its
///   own ranks that serve the kind's reading phase.
/// - In each other host where its ranks read the kind, and only when a group of its scheduler
///   writes it, a group receives the blocks in the product buffer of its first rank on that
///   host, where a crossing writes them (`storage::runtime::Request::Replicate`).
///
/// A rank that receives a kind holds a request's blocks of it in its product buffer as a rank
/// that backs it does ([`held_kinds`]).
///
/// # Errors
///
/// Serving groups that do not form `data_parallel` equal blocks, or a session that numbers no
/// conditioning kind.
pub(crate) fn product_ranks(
    groups: &[WorkerGroupConfig],
    data_parallel: usize,
) -> Result<Vec<Vec<KindRanks>>> {
    let kinds = product::KINDS.len();
    let schedulers = WorkerGroupConfig::scheduler_blocks(groups, data_parallel)?;

    // Per group, the phases its ranks serve and the ranks that back each kind.
    let phases: Vec<Vec<Phase>> = groups.iter().map(group_phases).collect();
    let mut backing: Vec<Vec<Vec<MediaRank>>> = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let mut backs = vec![Vec::new(); kinds];
        for position in 0..group.ranks.len() {
            for kind in backed_kinds(group, position)? {
                backs[kind].push((index, position));
            }
        }
        backing.push(backs);
    }
    let host = |(index, position): MediaRank| groups[index].ranks[position].host.as_str();

    let mut places: Vec<Vec<KindRanks>> = vec![vec![(Vec::new(), Vec::new()); kinds]; groups.len()];
    for scheduler in &schedulers {
        for &index in scheduler {
            let group = &groups[index];
            for (kind, (_, writer, reader)) in product::KINDS.iter().enumerate() {
                // The hosts of the group's ranks that serve the kind's reading phase, in the
                // order of their first such rank.
                let mut reading: Vec<&str> = Vec::new();
                for position in 0..group.ranks.len() {
                    let on = host((index, position));
                    if served_phases(group, position).contains(reader) && !reading.contains(&on) {
                        reading.push(on);
                    }
                }

                let writers: Vec<usize> = scheduler
                    .iter()
                    .copied()
                    .filter(|other| phases[*other].contains(writer))
                    .collect();
                let regions: Vec<MediaRank> = if phases[index].contains(writer) {
                    backing[index][kind].clone()
                } else {
                    writers
                        .iter()
                        .flat_map(|other| backing[*other][kind].iter().copied())
                        .filter(|rank| reading.contains(&host(*rank)))
                        .collect()
                };

                // Each host the group reads the kind on where none of its regions lies receives
                // it in its first rank's product buffer there.
                let mut receives = Vec::new();
                if !writers.is_empty() {
                    for on in &reading {
                        if regions.iter().any(|rank| host(*rank) == *on) {
                            continue;
                        }
                        let first = (0..group.ranks.len()).find(|at| host((index, *at)) == *on);
                        receives.extend(first.map(|position| (index, position)));
                    }
                }
                places[index][kind] = (regions, receives);
            }
        }
    }
    Ok(places)
}

/// The product kinds whose blocks rank `rank` holds in its product buffer, as indices into
/// `crate::batch::product::KINDS`, by the places of every group's kinds `places`
/// ([`product_ranks`]): those it backs, and those it receives for a group that reads them on its
/// host.
pub(crate) fn held_kinds(places: &[Vec<KindRanks>], rank: MediaRank) -> Vec<usize> {
    (0..product::KINDS.len())
        .filter(|kind| {
            places.iter().any(|group| {
                group.get(*kind).is_some_and(|(regions, receives)| {
                    regions.contains(&rank) || receives.contains(&rank)
                })
            })
        })
        .collect()
}

/// The phases any rank of `group` serves.
pub(crate) fn group_phases(group: &WorkerGroupConfig) -> Vec<Phase> {
    let mut phases: Vec<Phase> = (0..group.ranks.len())
        .flat_map(|position| served_phases(group, position))
        .collect();
    phases.sort_by_key(|phase| phase.index());
    phases.dedup();
    phases
}

/// The session's class of each product kind, in `crate::batch::product::KINDS` order, when the
/// product classes follow `first` classes of cache groups.
pub(crate) fn product_classes(first: u8) -> [u8; 10] {
    std::array::from_fn(|kind| first.saturating_add(kind as u8))
}

/// The index of product kind `name` in `crate::batch::product::KINDS`.
///
/// # Errors
///
/// A name that is no product kind of the session.
pub(crate) fn product_kind(name: &str) -> Result<usize> {
    product::kind(name).with_context(|| format!("{name} is no product kind of the session"))
}

/// The bytes of one element of `dtype`.
fn element(dtype: DType) -> Result<u64> {
    ops::abi::bytes(dtype)
        .with_context(|| format!("{} has no whole-byte element", ops::abi::name(dtype)))
}

/// Checks that `model`'s decoders read their latents in the element type its denoiser keeps the
/// trajectory in, which the last denoising step writes the latents products in.
///
/// # Errors
///
/// A decoder that reads latents of another element type.
fn check_latents(model: &dyn VideoModel) -> Result<()> {
    let sample = model.sample_type();
    for (name, decoder) in [
        ("video", model.video_decoder()),
        ("audio", model.audio_decoder()),
    ] {
        if let Some(decoder) = decoder
            && decoder.latent_type() != sample
        {
            bail!(
                "the {name} decoder reads latents of {} and the denoiser keeps its trajectory in {}",
                ops::abi::name(decoder.latent_type()),
                ops::abi::name(sample)
            );
        }
    }
    Ok(())
}

/// The requests the deployment `serving` captures its denoiser for, each with the denoiser's
/// layout of it.
///
/// # Errors
///
/// A model that places no denoiser, or a layout it refuses.
fn laid_out(model: &dyn VideoModel, serving: &VideoServing) -> Result<Vec<(VideoRequest, Layout)>> {
    let denoiser = model
        .denoiser()
        .context("the rank's share of the checkpoint holds no denoiser to lay requests out")?;
    let candidates = model.candidates(serving)?;
    if candidates.is_empty() {
        bail!("this deployment serves no request layout");
    }
    candidates
        .into_iter()
        .map(|(request, _)| {
            let layout = denoiser.layout(&request)?;
            Ok((request, layout))
        })
        .collect()
}

/// The pages of a rank's host latent region for trajectories of at most `trajectory` bytes and
/// `slots` request slots: a mebibyte of float32 samples a page, every slot's trajectory pages.
///
/// # Errors
///
/// A trajectory of more pages than a `u32` counts.
pub(crate) fn latent_pages(trajectory: u64, slots: u32) -> Result<BankPages> {
    let pages = BankPages {
        dtype: DType::F32,
        page_values: LATENT_PAGE_VALUES,
        trajectory_pages: 0,
        pages: 0,
    };
    let page_bytes = pages.page_bytes()?;
    let trajectory_pages = u32::try_from(trajectory.div_ceil(page_bytes))
        .context("a trajectory of more pages than a u32 counts")?;
    Ok(slot_pages(
        BankPages {
            trajectory_pages,
            ..pages
        },
        slots,
    ))
}

/// `pages` holding the trajectories of `slots` request slots: every slot's trajectory pages.
pub(crate) fn slot_pages(pages: BankPages, slots: u32) -> BankPages {
    BankPages {
        pages: u64::from(pages.trajectory_pages) * u64::from(slots),
        ..pages
    }
}

/// The requests a stage of `phase` holds at once on a rank of `slots` request slots, in a process
/// whose media pool runs `pool_tasks` tasks: the slots of an encode, the pool's tasks for a host
/// phase, and one request of a denoiser or a decoder.
fn phase_rows(phase: Phase, slots: u32, pool_tasks: u32) -> u32 {
    match phase {
        Phase::TextEncode => slots.max(1),
        phase if HOST_PHASES.contains(&phase) => pool_tasks.max(1),
        _ => 1,
    }
}

/// The prompt capacities a media deployment captures: its own statement, or the default ladder,
/// completed with `longest`, the prompt length it admits (`max_model_len`).
///
/// Every prompt the deployment admits rides the smallest capacity that holds it, so the largest
/// capacity is `longest` itself. The deployment's `text_capacities`, when stated, must be strictly
/// increasing and none above `longest` - a capacity above it would capture an encoder graph no
/// request can fill; a statement whose largest is below `longest` is completed with it. The
/// default ladder is `1024`, then steps of `2048`, below `longest`, completed the same way.
pub(super) fn prompt_capacities(
    video: &proto::config::VideoConfig,
    longest: u32,
) -> Result<Vec<u32>> {
    if longest == 0 {
        bail!("this media deployment states no prompt length (`max_model_len`) to capture");
    }
    let mut stated: Vec<u32> = match &video.text_capacities {
        Some(capacities) => capacities.iter().map(|value| value.get()).collect(),
        None => {
            let mut ladder = Vec::new();
            let mut capacity = 1024;
            while capacity < longest {
                ladder.push(capacity);
                capacity += 2048;
            }
            ladder
        }
    };
    for pair in stated.windows(2) {
        if pair[0] >= pair[1] {
            bail!(
                "this deployment's prompt capacities {} are not strictly increasing, so a request \
                 of {} tokens would select two captured graphs",
                stated
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                pair[0]
            );
        }
    }
    if let Some(above) = stated.iter().copied().find(|capacity| *capacity > longest) {
        bail!(
            "this deployment states the prompt capacity {above} above the {longest} tokens it \
             admits: the graph would be captured and never filled"
        );
    }
    if stated.last() != Some(&longest) {
        stated.push(longest);
    }
    Ok(stated)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    use crate::batch::Phase;
    use proto::config::{BlockConfig, BlockDistribution, WorkerGroupConfig, WorkerGroupId};

    use super::{group_phases, prompt_capacities, served_phases};

    /// Video options stating `capacities`, or none.
    fn video(capacities: Option<&[u32]>) -> proto::config::VideoConfig {
        proto::config::VideoConfig {
            text_capacities: capacities.map(|capacities| {
                capacities
                    .iter()
                    .map(|capacity| NonZeroU32::new(*capacity).unwrap())
                    .collect()
            }),
            ..proto::config::VideoConfig::default()
        }
    }

    /// The largest capacity is the prompt length the deployment admits, whether its options state
    /// the ladder or take the default, so every admitted prompt has a capacity that holds it.
    #[test]
    fn the_admitted_prompt_length_is_the_largest_capacity() {
        assert_eq!(prompt_capacities(&video(None), 1024).unwrap(), [1024]);
        assert_eq!(
            prompt_capacities(&video(None), 4096).unwrap(),
            [1024, 3072, 4096]
        );
        assert_eq!(
            prompt_capacities(&video(Some(&[1024])), 10240).unwrap(),
            [1024, 10240]
        );
        assert_eq!(
            prompt_capacities(&video(Some(&[1024, 2048])), 2048).unwrap(),
            [1024, 2048]
        );
    }

    /// A capacity above the admitted length, or a ladder that is not strictly increasing, is
    /// refused.
    #[test]
    fn a_capacity_no_prompt_fills_is_refused() {
        assert!(prompt_capacities(&video(Some(&[4096])), 2048).is_err());
        assert!(prompt_capacities(&video(Some(&[2048, 1024])), 4096).is_err());
        assert!(prompt_capacities(&video(None), 0).is_err());
    }

    /// A group of four ranks as `configs/minimax_h3/ulysses4.toml` places it: every rank encodes,
    /// denoises and decodes its share of a round and encodes video; the audio track is decoded by
    /// the audio block's first rank; the reader and the muxer are rank 0's.
    #[test]
    fn a_rank_serves_the_phases_of_the_blocks_that_hold_it() {
        let every = || BlockConfig::parallel((0..4).collect());
        let blocks = BTreeMap::from([
            ("transformer".to_owned(), every()),
            ("text_encoder".to_owned(), every()),
            (
                "video_decoder".to_owned(),
                BlockConfig {
                    distribution: Some(BlockDistribution::TemporalUnits),
                    ..every()
                },
            ),
            (
                "audio_decoder".to_owned(),
                BlockConfig {
                    distribution: Some(BlockDistribution::TemporalUnits),
                    ..every()
                },
            ),
            ("video_codec".to_owned(), every()),
            ("muxer".to_owned(), BlockConfig::parallel(vec![0])),
            ("media_reader".to_owned(), BlockConfig::parallel(vec![0])),
        ]);
        let rank = proto::config::WorkerRank {
            host: "localhost".to_owned(),
            device: "cuda:0".to_owned(),
        };
        let group = WorkerGroupConfig {
            id: WorkerGroupId("model".to_owned()),
            role: Default::default(),
            replicas: 1,
            ranks: vec![rank; 4],
            blocks,
            queue_depth: 6,
            mem_fraction_static: None,
        };

        // In the order of the phases' ids, which a rank states them in.
        let mut device = vec![
            Phase::TextEncode,
            Phase::LatentPrepare,
            Phase::Denoise,
            Phase::VideoDecode,
            Phase::VideoEncode,
        ];
        device.sort_by_key(|phase| phase.index());
        let mut first = device.clone();
        first.extend([
            Phase::MediaRead,
            Phase::AudioDecode,
            Phase::AudioEncode,
            Phase::Mux,
        ]);
        first.sort_by_key(|phase| phase.index());
        assert_eq!(served_phases(&group, 0), first);
        for position in 1..4 {
            assert_eq!(served_phases(&group, position), device, "rank {position}");
        }
        assert_eq!(group_phases(&group), first);
    }
}
