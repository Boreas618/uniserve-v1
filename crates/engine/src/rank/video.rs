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
//! The rank loads every stage its group places, split as the group's blocks state
//! ([`ModelRunner::video_split`]).
//!
//! **Runners.** One per block the rank serves, each on a stream of its own: [`EncoderRunner`],
//! [`DenoiserRunner`], a [`DecoderRunner`] for the video and one for the track, and the
//! [`MediaHost`] of its host work, which computes nothing on the device. They read and write a
//! request's products in the blocks the rank's device block tables name, in the rank's one
//! [`BufferMemory`], which every media runner of the rank shares.
//!
//! **Sizes.** A rank of the group that denoises states its group's captured request shapes with
//! the bytes of a slot's trajectory and of a block of each product kind there
//! (`Capacity::media_bytes`, in `ipc::product::KINDS` order), from its denoiser's shapes and its
//! decoders, which that group therefore holds. The session sizes every rank's product buffer
//! and its host regions from them (`crate::process::ranks::layout`).

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ipc::Capacity;
use ipc::batch::{Extent, Phase, Shape, Variant};
use ipc::capacity::variants;
use ipc::product;
use nn::VideoRequest;
use nn::family::{Stage, VideoModel, VideoPlacement, VideoServing};
use ops::Generation;
use platform::{DeviceId, Stream};
use proto::config::{
    DEFAULT_BLOCK, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK, WorkerGroupConfig, WorkerGroupRole,
};
use proto::video::Layout;
use tensor::DType;
use worker::buckets::{Bucket, Buckets};
use worker::graphs::Graphs;
use worker::memory::buffers::BufferMemory;
use worker::memory::client::StorageClient;
use worker::memory::division::{self, call_window};
use worker::memory::encoders;
use worker::memory::latent::host::BankPages;
use worker::runners::decoder::{DecoderConfig, DecoderRunner, RoundConfig};
use worker::runners::denoiser::{DenoiserConfig, DenoiserRunner};
use worker::runners::encoder::{EncoderConfig, EncoderRunner};
use worker::runners::media::{HostConfig, MediaHost};
use worker::thread_pool::ThreadPool;
use worker::{Runner, RunnerId};

use super::memory::{Peaks, unreserved};
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

/// `runners` as the executor takes them, each with the region of the graph pool it replays in.
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
    /// The text encoder's tensor group and the denoiser's sequence group; `None` for a group of
    /// one and on the meta device.
    pub(crate) tensor: Option<Arc<ops::Group>>,
    pub(crate) sequence: Option<Arc<ops::Group>>,
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
    /// On a rank that denoises: the trajectory class's `tensors` and `encoding` digests, and the
    /// pages of its host latent region.
    pub(crate) trajectory: Option<([u8; 32], [u8; 32], BankPages)>,
    /// What the rank states of its worker group: its own phases and the shapes of its runners.
    pub(crate) capacity: Capacity,
    /// The rank's media memory, from its opening until the process's memories are connected.
    pub(crate) memory: Option<BufferMemory>,
    /// Once the rank is opened, by the numbers the session's answer gave them: the regions its
    /// device block tables map (its group's slot rows, its product buffer and its parts of host
    /// regions), and the product buffers of the process's other ranks, which its device reaches.
    pub(crate) regions: (Vec<u8>, Vec<u8>),
    /// The rank's serving runners, from its opening until its executor takes them.
    pub(crate) runners: Vec<Box<dyn Runner>>,
}

impl ModelRunner {
    /// One model runner per rank of a media deployment, every rank of its worker groups in the
    /// deployment's rank order, each on the device the deployment places it on, which this
    /// process serves. A device that holds ranks of several groups serves each as a rank of its
    /// own.
    ///
    /// # Errors
    ///
    /// A rank on a device this process does not serve; an expert group, which serves no video
    /// request; and a media group with a rank on another host (`specs/engine-worker-port.md` §5
    /// D2): a product lies in the product buffer of every rank that backs it, and this build
    /// reaches no product buffer of another process.
    pub(super) fn media_ranks(&self) -> Result<Vec<Self>> {
        let groups = &self.engine_config().parallel.worker_groups;
        let names: Vec<&str> = product::KINDS.iter().map(|(name, ..)| *name).collect();
        for group in groups.iter().filter(|group| group.role.serves_requests()) {
            if let Some((position, rank)) = group
                .ranks
                .iter()
                .enumerate()
                .find(|(_, rank)| rank.host != self.launch.host)
            {
                bail!(
                    "worker group {} places its rank {position} on host {}, and this process \
                     serves host {}: the products of a video request ({}) lie in the product \
                     buffer of every rank that backs them, and no product buffer of another \
                     process is reached",
                    group.id,
                    rank.host,
                    self.launch.host,
                    names.join(", ")
                );
            }
        }

        let firsts = WorkerGroupConfig::first_ranks(groups);
        let mut ranks = Vec::new();
        for (first, group) in firsts.iter().zip(groups) {
            if group.role == WorkerGroupRole::Experts {
                bail!(
                    "worker group {} is an expert group, which serves no video request",
                    group.id
                );
            }

            for (position, placed) in group.ranks.iter().enumerate() {
                let rank = u32::try_from(first + position)
                    .context("a rank past what the protocol numbers")?;
                let device = placed
                    .device
                    .strip_prefix("cuda:")
                    .and_then(|ordinal| ordinal.parse::<i32>().ok())
                    .map(DeviceId)
                    .filter(|device| self.devices.contains(device));
                let Some(device) = device else {
                    bail!(
                        "the deployment places rank {rank} (rank {position} of worker group {}) \
                         on {}, which this process does not serve: it serves devices {:?}",
                        group.id,
                        placed.device,
                        self.devices
                            .iter()
                            .map(|device| device.0)
                            .collect::<Vec<_>>()
                    );
                };

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
        }
        Ok(ranks)
    }

    /// The share of the video pipeline this rank's checkpoint holds that its group places, split
    /// as the group's blocks state, its weights reached through `weights`.
    ///
    /// # Errors
    ///
    /// The checkpoint's or the family's refusal, or a split the deployment's groups do not state.
    pub(crate) fn load_video(&self, weights: Weights<'_>) -> Result<Arc<dyn VideoModel>> {
        let opened = self.open_video_checkpoint()?;
        let checkpoint = self.open_video_family(&opened)?;
        let (sequence, tensor) = self.video_split(&*checkpoint)?;

        let mut loader = weights.loader(&opened);
        let placement = VideoPlacement {
            stages: self.stages(),
            sequence,
            tensor,
            dtype: self.dtype(),
            serving: self.video_serving()?,
        };
        Ok(checkpoint.load(&opened, &mut loader, &placement)?)
    }

    /// The stages this rank's group places; `None` when the deployment places the whole model.
    fn stages(&self) -> Option<BTreeSet<String>> {
        self.capability_groups()?;
        let group = self.worker_group()?;
        Some(group.blocks.keys().cloned().collect())
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
    /// members are the deployment's ranks: the text encoders' tensor groups and the denoisers'
    /// sequence groups, each a block's ranks as consecutive members, with this rank's place
    /// among them. Every rank forms both, in this order, so every member of the host group takes
    /// part in every group's id broadcast.
    ///
    /// # Errors
    ///
    /// A split block whose ranks are not consecutive positions of its group.
    pub(crate) fn media_collectives(&self) -> Result<(super::Collective, super::Collective)> {
        let groups = &self.engine_config().parallel.worker_groups;
        let firsts = WorkerGroupConfig::first_ranks(groups);
        let text = Stage::TextEncoder.name();

        let mut tensor = super::Collective {
            spans: Vec::new(),
            member: None,
        };
        let mut sequence = super::Collective {
            spans: Vec::new(),
            member: None,
        };
        for (first, group) in firsts.iter().zip(groups) {
            for (name, block) in &group.blocks {
                let (collective, size) = if name == text {
                    (&mut tensor, block.tensor_parallel_size)
                } else {
                    let size = block.sequence_parallel.size().map_err(|refusal| {
                        anyhow::anyhow!("block {name} of worker group {}: {}", group.id, refusal.0)
                    })?;
                    (&mut sequence, size)
                };
                if size <= 1 {
                    continue;
                }

                // A block's ranks form its group in their order, as consecutive members.
                let Some(&lead) = block.ranks.first() else {
                    continue;
                };
                let consecutive = block
                    .ranks
                    .iter()
                    .enumerate()
                    .all(|(index, position)| *position == lead + index);
                if !consecutive || block.ranks.len() != size {
                    bail!(
                        "block {name} of worker group {} splits over its ranks {:?}, which are not \
                         {size} consecutive ranks of the group",
                        group.id,
                        block.ranks
                    );
                }

                let span = collective.spans.len();
                collective.spans.push((first + lead, size));
                if group.id.0 == self.group {
                    let position = self.position()?;
                    if let Some(place) = block.ranks.iter().position(|held| *held == position) {
                        collective.member = Some((span, place));
                    }
                }
            }
        }
        Ok((tensor, sequence))
    }

    /// How a video decode round is dealt to the ranks of the group that decodes this rank's
    /// requests: its video decoder block's ranks in order, by their rank index in the process's
    /// media memory (the deployment's rank), each decoding the block's units per rank when it
    /// deals units and one otherwise. Its width is the scheduler's (`crate::scheduler::video`).
    /// A group that places no video decoder block decodes a round of one unit on its own rank.
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
        let units = if block.distribution.is_some() {
            block.units_per_rank.max(1)
        } else {
            1
        };
        Ok(RoundConfig {
            ranks,
            units: u32::try_from(units).context("more units per rank than a round numbers")?,
        })
    }

    /// The config of the rank's text encoder of `model`, writing the conditioning and reading
    /// the vision pixels of their classes in `classes` (the session's class of each product kind,
    /// in `ipc::product::KINDS` order): one captured encode bucket per request count up to the
    /// slots and prompt capacity the deployment captures, and for an encoder that takes images
    /// the prompts with vision blocks the deployment admits, whose eager encodes it profiles.
    ///
    /// # Errors
    ///
    /// Capacities the deployment states that are not strictly increasing or above its prompt
    /// length, or prompts with vision blocks the model cannot state.
    pub(crate) fn encoder_config(
        &self,
        model: &dyn VideoModel,
        classes: &[u8; 10],
    ) -> Result<EncoderConfig> {
        let serving = self.video_serving()?;
        let slots = self.slots()?;
        let mut buckets = Vec::new();
        for capacity in &serving.text_capacities {
            for rows in 1..=slots {
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
        Ok(EncoderConfig {
            buckets: Buckets::new(buckets),
            width: model.text_width(),
            dtype: self.dtype(),
            patches,
            conditioning: classes[product_kind(product::CONDITIONING)?],
            pixels: classes[product_kind(product::VISION_PIXELS)?],
            warm,
        })
    }

    /// The config of the rank's denoiser of `model`: the requests the deployment serves, the
    /// classes it reads and writes, the ranks of its group's denoiser block, and the host latent
    /// region's page bytes. The first rank of the block shares the latents its last step writes
    /// with the process's other backing ranks and writes a saved trajectory's host copy.
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

        let (ranks, first) = match self.denoiser_block() {
            Some(block) => {
                let position = self.position()?;
                let ranks = u32::try_from(block.ranks.len()).context("a block wider than u32")?;
                (ranks, block.ranks.first() == Some(&position))
            }
            None => (1, true),
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
    /// the deployment serves, with the timelines of the denoiser's layouts, dealt as the
    /// deployment's rounds; one track bucket per timeline whose layout carries a track.
    ///
    /// # Errors
    ///
    /// A layout the denoiser refuses, a deployment the pipeline serves no request of, or a
    /// pipeline whose decoders read latents in another element type than the trajectory's.
    pub(crate) fn decoder_configs(
        &self,
        model: &dyn VideoModel,
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

        check_latents(model)?;
        let laid = laid_out(model, &self.video_serving()?)?;
        let window = model.decode_window();
        let (mut clips, mut timelines, mut tracks) = (Vec::new(), Vec::new(), Vec::new());
        let mut served: Vec<u32> = Vec::new();
        for (request, layout) in &laid {
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
    /// # Errors
    ///
    /// A deployment the pipeline serves no request of, or a model that states no media format.
    pub(crate) fn host_config(
        &self,
        model: &dyn VideoModel,
        classes: &[u8; 10],
        phases: Vec<Phase>,
    ) -> Result<HostConfig> {
        let mut timelines: Vec<(u32, Vec<u32>)> = Vec::new();
        for (request, _) in model.candidates(&self.video_serving()?)? {
            if timelines
                .iter()
                .all(|(frames, _)| *frames != request.frames)
            {
                timelines.push((request.frames, model.video_units(request.frames)?));
            }
        }

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
    /// over the tensor group `tensor`, the denoiser over the sequence group `sequence`, each at
    /// this rank's place (meta graphs take the split alone). The media host is
    /// [`ModelRunner::media_host`]'s.
    ///
    /// # Errors
    ///
    /// A config the model refuses, or a runner that cannot be built over its graphs.
    pub(crate) fn device_runners(
        &self,
        model: &Arc<dyn VideoModel>,
        memory: &Arc<BufferMemory>,
        mut graphs: impl FnMut(RunnerId, u64) -> Result<(Graphs, Option<Stream>)>,
        (tensor, sequence): (Option<Arc<ops::Group>>, Option<Arc<ops::Group>>),
        classes: &[u8; 10],
    ) -> Result<DeviceRunners> {
        let phases = self.media_phases()?;
        let shape = self.parallel_shape()?;

        let encoder = if phases.contains(&Phase::TextEncode) {
            let text = model.text_encoder().context(
                "the rank encodes prompts and its share of the checkpoint holds no text encoder",
            )?;
            let config = self.encoder_config(model.as_ref(), classes)?;
            let (graphs, stream) = graphs(ENCODER, EncoderRunner::input_bytes(&config))?;
            let (rank, ranks) = super::split(shape.tensor_rank, shape.tensor);
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
            let (rank, ranks) = super::split(shape.sequence_rank, shape.sequence);
            graphs.set_group(sequence, rank, ranks)?;
            let runner = DenoiserRunner::new(network, config, graphs, stream, Arc::clone(memory))
                .context("building the rank's denoiser")?;
            Some(runner)
        } else {
            None
        };

        let (video, audio) = self.decoder_configs(model.as_ref(), classes)?;
        let video = match video {
            Some(config) => {
                let decoder = model.video_decoder().context(
                    "the rank decodes video and its share of the checkpoint holds no video decoder",
                )?;
                let (graphs, stream) = graphs(VIDEO_DECODER, 0)?;
                let splicer = model.splicer();
                let runner = DecoderRunner::new(
                    decoder,
                    splicer,
                    config,
                    graphs,
                    stream,
                    Arc::clone(memory),
                )
                .context("building the rank's video decoder")?;
                Some(runner)
            }
            None => None,
        };
        let audio = match audio {
            Some(config) => {
                let decoder = model.audio_decoder().context(
                    "the denoiser's layouts carry a track and the rank's share of the checkpoint \
                     holds no audio decoder",
                )?;
                let (graphs, stream) = graphs(AUDIO_DECODER, 0)?;
                let runner =
                    DecoderRunner::new(decoder, None, config, graphs, stream, Arc::clone(memory))
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
    /// §1.1): its runners over meta graphs and a media memory that holds no device, and each
    /// runner's profile run. Returns the rank as its opening and its statement read it, with the
    /// parallel groups `groups` it computes over: the runners' peaks and buffers, a slot's row,
    /// on a denoising rank the trajectory class and its group's product sizes, and the rank's
    /// capacity with the media pool's `pool_tasks`; and every kernel the runners launch.
    ///
    /// # Errors
    ///
    /// What a runner or a forward of a profile run refuses, or a group that denoises and holds
    /// no decoder of the products it sizes.
    pub(crate) fn profile_media(
        &self,
        model: Arc<dyn VideoModel>,
        groups: (Option<Arc<ops::Group>>, Option<Arc<ops::Group>>),
        pool_tasks: u32,
    ) -> Result<(VideoRank, Keys)> {
        let classes = product_classes(0);
        let slots = self.slots()?;
        let row = self.row_bytes(model.as_ref(), &classes)?;

        // The region numbers the profile's memory states are none of a forward's concern: the
        // session's answer numbers the rank's memory when it opens.
        let memory = Arc::new(BufferMemory::meta(self.rank, (1, 1), (0, slots), row, &[])?);
        let unserved = Arc::new(Generation::new(0, []));
        let (encoder, denoiser, video, audio) = self.device_runners(
            &model,
            &memory,
            |_, inputs| Ok((Graphs::meta(Arc::clone(&unserved), inputs), None)),
            (None, None),
            &classes,
        )?;

        // What the runners hold beside their regions, and on a denoising rank the trajectory
        // class and its group's product sizes.
        let mut buffers: u64 = video
            .iter()
            .chain(&audio)
            .map(DecoderRunner::device_bytes)
            .sum();
        let (mut trajectory, mut media_bytes, mut units) = (None, Vec::new(), 0);
        if let Some(denoiser) = &denoiser {
            buffers += denoiser.device_bytes();
            let (tensors, encoding) = denoiser.trajectory_digests();
            let shapes = denoiser.media_shapes()?;
            let largest = shapes.iter().map(|(_, bytes, _)| *bytes).max().unwrap_or(0);
            trajectory = Some((tensors, encoding, latent_pages(largest, slots)?));
            (media_bytes, units) = self.media_bytes(model.as_ref(), &shapes)?;
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
            peaks.push((
                *id,
                Peaks {
                    graph: graphs.peak(),
                    shadow: graphs.shadow_peak(),
                    inputs: graphs.input_peak(),
                },
            ));
        }

        let capacity = self.media_capacity(model.as_ref(), &runners, media_bytes, pool_tasks)?;
        let (tensor, sequence) = groups;
        Ok((
            VideoRank {
                model,
                tensor,
                sequence,
                peaks,
                buffers,
                row,
                units,
                trajectory,
                capacity,
                memory: None,
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
    /// `ipc::product::KINDS` order (`Capacity::media_bytes`); and the most units a timeline of the
    /// deployment has, the splice tails a request declares.
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
    /// - The encoded units: the media host's rows of the timeline's units.
    /// - The audio samples: the track's PCM window, stereo `i16`; none without a track.
    ///
    /// # Errors
    ///
    /// A denoising group that holds no video decoder, or no audio decoder for a layout with a
    /// track; a pipeline whose decoders read latents in another element type than the
    /// trajectory's; a shape no candidate of the deployment lays out.
    #[expect(
        clippy::type_complexity,
        reason = "`Capacity::media_bytes` as the scheduler reads it, and the tails a request declares"
    )]
    fn media_bytes(
        &self,
        model: &dyn VideoModel,
        shapes: &[(Shape, u64, [u64; 2])],
    ) -> Result<(Vec<(Shape, u64, Vec<u64>)>, u32)> {
        let decoder = model.video_decoder().context(
            "the group that denoises sizes the decoded products, and its share of the checkpoint \
             holds no video decoder",
        )?;
        check_latents(model)?;
        let track = model.audio_decoder();

        let bounds = model.condition_bounds();
        let format = model.format().ok();
        let splicer = model.splicer();
        let laid = laid_out(model, &self.video_serving()?)?;
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
                    "the denoiser's layouts carry a track and the group that denoises holds no \
                     audio decoder"
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
        let slots = self.slots()?;

        let mut served = 0;
        let mut rows = [0; Phase::COUNT];
        for phase in &phases {
            served |= 1 << phase.index();
            rows[phase.index()] = match phase {
                Phase::TextEncode => slots.max(1),
                phase if HOST_PHASES.contains(phase) => pool_tasks.max(1),
                _ => 1,
            };
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

    /// The digests the class of a video request's conditioning states (`storage::runtime::Class`)
    /// for `model`: its tensors, rows of the text encoder's width in the deployment's element
    /// type, and its encoding, an output's bytes in row order over the host encoder region's
    /// units, which every parallel configuration lays alike (`worker::memory::encoders`).
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
            .iter()
            .map(|size| {
                size.split_once('x')
                    .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
                    .with_context(|| format!("the deployment's size {size:?} is not WxH"))
            })
            .collect::<Result<Vec<(u32, u32)>>>()?;
        Ok(VideoServing {
            sizes,
            max_seconds: video.max_seconds,
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

/// The phases any rank of `group` serves.
pub(crate) fn group_phases(group: &WorkerGroupConfig) -> Vec<Phase> {
    let mut phases: Vec<Phase> = (0..group.ranks.len())
        .flat_map(|position| served_phases(group, position))
        .collect();
    phases.sort_by_key(|phase| phase.index());
    phases.dedup();
    phases
}

/// The session's class of each product kind, in `ipc::product::KINDS` order, when the product
/// classes follow `first` classes of cache groups.
pub(crate) fn product_classes(first: u8) -> [u8; 10] {
    std::array::from_fn(|kind| first.saturating_add(kind as u8))
}

/// The index of product kind `name` in `ipc::product::KINDS`.
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
    Ok(BankPages {
        trajectory_pages,
        pages: u64::from(trajectory_pages) * u64::from(slots),
        ..pages
    })
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

    use ipc::batch::Phase;
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
