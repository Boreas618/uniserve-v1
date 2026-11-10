//! The runners of a video rank: one per block of its pipeline, and its host stage.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ipc::execution::Variant;
use nn::family::{VideoModel, VideoServing};
use ops::{DType, Generation};
use platform::DeviceId;
use worker::blocking::BlockingKind;
use worker::plan::Buckets;
use worker::runner::{Binding, NamedRunner, RunnerId};

use super::buckets::{host_phases, phase, served_phases, unit_phases};
use super::{ModelRunner, RankGroups};

impl ModelRunner {
    /// The runners a media rank serves `device` with: one per block the pipeline holds.
    ///
    /// **Three runners, one stream each.** An encoder writes the hidden states a denoise step reads,
    /// a denoiser reads them and publishes the trajectory, and a decoder reads that trajectory as a
    /// clip, so each is a separate stream on the one device - a stream per runner is what lets the
    /// store order the crossings between them.
    ///
    /// **The buckets are the deployment's canvases and prompt capacities**, not the checkpoint's: a
    /// bucket is a captured graph, and a deployment that captures a canvas it does not serve would
    /// pay a capture for nothing.
    #[expect(
        clippy::too_many_arguments,
        reason = "what `runners` takes, for the media class"
    )]
    pub(super) fn media_runners(
        &self,
        device: DeviceId,
        pipeline: &Arc<dyn VideoModel>,
        generation: &Arc<Generation>,
        blocking: &worker::blocking::ThreadPool,
        groups: &RankGroups,
        client: Option<&Arc<worker::memory::retained::Retained>>,
        identity: &str,
        report: Option<&mut ipc::MainProcessInfo>,
    ) -> Result<Vec<NamedRunner>> {
        use worker::runners::decoder as media_decoder;
        use worker::runners::denoiser as media_denoiser;
        use worker::runners::encoder as media_encoder;
        use worker::runners::media as media_host;

        if Some(&device) != self.devices.first() {
            bail!(
                "this build maps one model per rank, on device {}; a runner on device {} would \
                 forward from another device's addresses",
                self.devices.first().map_or(-1, |first| first.0),
                device.0
            );
        }
        let serving = self.video_serving()?;
        let capacities = &serving.text_capacities;
        // The model's own statement of the requests its graphs are captured for: one per layout a
        // request of this deployment may take.
        let denoiser = pipeline
            .denoiser()
            .context("the checkpoint holds no denoiser, so no denoising step could run")?;
        let candidates = pipeline.candidates(&serving)?;
        if candidates.is_empty() {
            bail!("this deployment serves no request layout, so a denoiser would capture nothing");
        }

        // The buckets the deployment captures: an encode bucket per (rows, prompt capacity) and a
        // decode bucket per canvas. A denoiser derives its own from the candidates it walks, so the
        // set here is what the encoder and the decoder are bound with.
        let mut encode_buckets: Vec<worker::plan::Bucket> = Vec::new();
        let rows = u32::try_from(self.engine_config().max_running_requests.get())
            .context("more request slots than the protocol addresses")?;
        for capacity in capacities {
            for row in 1..=rows {
                encode_buckets.push(worker::plan::Bucket::prefill(
                    row,
                    *capacity,
                    Variant::Encode,
                ));
            }
        }
        // **A decode bucket is one decode unit, not the timeline.** The video decoder decodes a clip
        // of the checkpoint's own window - `TemporalClips::window` latent frames, `frames_out` native ones -
        // and a timeline longer than one is spliced from its units. A bucket naming the deployment's
        // whole clip asked one decode call for a raster no decoder produces.
        let window = pipeline.decode_window();
        let mut decode_buckets: Vec<worker::plan::Bucket> = Vec::new();
        // The timelines a request of each canvas may ask for: a clip's splice follows its timeline's
        // schedule, so the decoder's recorded walk splices every clip of each of them.
        let mut timelines: BTreeMap<worker::plan::Bucket, std::collections::BTreeSet<u32>> =
            BTreeMap::new();
        for candidate in &candidates {
            let bucket = worker::plan::Bucket::clip(window, candidate.height, candidate.width);
            if !decode_buckets.contains(&bucket) {
                decode_buckets.push(bucket);
            }
            timelines
                .entry(bucket)
                .or_default()
                .insert(candidate.frames);
        }
        let hidden = pipeline.text_width();
        let text = pipeline.text_encoder().context(
            "the checkpoint holds no text encoder, so a media request's prompt could not be \
             encoded",
        )?;
        // The encoder writes its hidden states in the deployment's effective type, which the daemon
        // resolved against the checkpoint before it wrote this build config (a `dtype` of `auto`
        // never reaches a main process as `auto`).
        let dtype = DType::from(self.engine_config().model.dtype);

        let stream = |name: &str| -> Result<platform::Stream> {
            platform::Stream::new(device)
                .with_context(|| format!("creating device {}'s {name} stream", device.0))
        };
        let encode_set = Buckets::new(encode_buckets);
        let decode_set = Buckets::new(decode_buckets);
        // How the model reads condition media: the media stage writes a prompt's patch rows by it,
        // and the encoder's recorded walk lays out the admitted vision prompts' rows by it.
        let format = pipeline.format()?;
        // The prompts with vision blocks this deployment admits, which the encoder encodes eagerly
        // over its stream's region: its recorded walk encodes them so the region holds them.
        let vision = media_encoder::WarmVision {
            prompts: pipeline.warm_prompts(&serving)?,
            patches: format.patches,
        };
        let encoder = media_encoder::EncoderRunner::new(
            Binding {
                id: RunnerId(0),
                phases: served_phases(&encode_set, false),
                device,
                stream: stream("encoder")?,
            },
            text,
            encode_set,
            vision,
            media_encoder::EncoderShape { hidden, dtype },
            Arc::clone(generation),
        )?
        .in_group(groups.tensor.clone());
        // The host encoder pool a prompt's encoded states are stored in and found again by a
        // resumed request: one per host, shared by its ranks and epochs. This encoder writes the
        // conditioning alone; the vision and latent features are computed inside the calls that
        // read them, so they are no output of their own. The engine counts this process's ranks as
        // one, and the first writes the states it computed. A recorded start, which has no Storage
        // Server, maps none.
        let reports = report.is_some();
        let longest = u64::from(capacities.iter().copied().max().unwrap_or(0));
        let element = dtype
            .bytes()
            .context("the encoder's states have no whole-byte element")?;
        let (encoder, encoder_info) = match client {
            Some(client) => {
                let (pool, info) = worker::memory::encoders::EncoderHostPool::start(
                    client,
                    device,
                    worker::memory::encoders::pool_pages(
                        &[longest * u64::from(hidden) * element],
                        self.engine_config().max_running_requests.get(),
                    ),
                    BTreeMap::from([(ipc::product::CONDITIONING.to_owned(), identity.to_owned())]),
                    None,
                )
                .context("mapping the host encoder pool")?;
                (
                    encoder.with_host_outputs(Arc::new(pool), reports),
                    Some(info),
                )
            }
            None => (encoder, None),
        };
        let warm = worker::runners::warm::distinct(&*denoiser, &candidates)
            .context("a warm request the denoiser lays out")?;
        // The denoiser's own buckets are the layouts those candidates collapse to: its phase is
        // derived from them rather than named, so a rank that captures no denoise bucket states no
        // denoise phase.
        let denoise_buckets =
            warm.iter()
                .map(|candidate| {
                    Ok(
                        worker::plan::Bucket::denoise(1, denoiser.layout(&candidate.view())?)
                            .serving(candidate.frames, candidate.width, candidate.height),
                    )
                })
                .collect::<nn::Result<Vec<_>>>()
                .context("a warm request the denoiser lays out")?;
        let denoise_set = Buckets::new(denoise_buckets);
        // One audio decode bucket per distinct track those layouts carry. A `t2va` clip is a picture
        // **and** a soundtrack, so a deployment that captured clips alone would produce video with
        // no track; the track's extent is the layout the denoiser already states, and two canvases
        // of one duration collapse onto one track. A deployment that prepares no audio at all states
        // none, and then no audio runner is bound rather than one that could serve nothing.
        let audio_buckets: Vec<worker::plan::Bucket> = {
            let mut seen: Vec<(u32, u32)> = Vec::new();
            let mut buckets = Vec::new();
            for bucket in denoise_set.iter() {
                let ipc::execution::Extent::Latents(layout) = bucket.extent else {
                    continue;
                };
                let Some(audio) = layout.audio else {
                    continue;
                };
                if seen.contains(&(audio.frames, audio.channels)) {
                    continue;
                }
                seen.push((audio.frames, audio.channels));
                let mut track = worker::plan::Bucket::audio(1, layout);
                track.serves = bucket.serves;
                buckets.push(track);
            }
            buckets
        };
        // A latent preparation opens the trajectory the denoiser steps, so the denoiser serves its
        // phase beside its steps.
        let mut denoise_phases = served_phases(&denoise_set, false);
        denoise_phases.push(phase(scheduler::PhaseKind::LatentPrepare));
        let denoise = media_denoiser::DenoiserRunner::new(
            Binding {
                id: RunnerId(1),
                phases: denoise_phases,
                device,
                stream: stream("denoiser")?,
            },
            Arc::clone(&denoiser),
            warm,
            media_denoiser::DenoiserShape {
                stack: dtype,
                sample: pipeline.sample_type(),
                width: hidden,
                text_width: denoiser.text_width().unwrap_or(hidden),
            },
            Arc::clone(generation),
        )?
        .in_group(groups.sequence.clone());
        // The host latent pool a preempted request's trajectory is written to and a resumed one's
        // read from, at this rank's Storage Server, its pages sized for the denoiser's
        // largest trajectory. The engine counts this process's ranks as one rank of the worker
        // group, so every device reads stored pages as that rank, and the first states its pool.
        let trajectory = denoise.trajectory_bytes()?;
        let page_bytes = u64::from(LATENT_PAGE_VALUES) * 4;
        let pages = worker::memory::latent::host::BankPages {
            dtype: worker::memory::allocation::ElementType::F32,
            page_values: LATENT_PAGE_VALUES,
            trajectory_pages: u32::try_from(trajectory.div_ceil(page_bytes))
                .context("a trajectory of more pages than a u32 counts")?,
            pages: trajectory.div_ceil(page_bytes),
        };
        let (denoise, latent_info) = match client {
            Some(client) => {
                let (pool, info) = worker::memory::latent::host::LatentHostPool::start(
                    client,
                    device,
                    pages,
                    identity,
                    WHOLE_SAMPLES,
                    None,
                )
                .context("mapping the host latent pool")?;
                (
                    denoise.with_host_latents(Arc::new(pool), 0, reports),
                    Some(info),
                )
            }
            None => (denoise, None),
        };
        if let Some(report) = report {
            report.latent_pool = latent_info.map(Box::new);
            report.encoder_pool = encoder_info.map(Box::new);
        }
        let decoder = pipeline
            .video_decoder()
            .context("the checkpoint holds no video decoder, so no clip could be decoded")?;
        let decode = media_decoder::DecoderRunner::new(
            Binding {
                id: RunnerId(2),
                phases: served_phases(&decode_set, false),
                device,
                stream: stream("decoder")?,
            },
            decoder,
            decode_set,
            Arc::clone(generation),
            // The clip splicer (`nn::capability::Splicer`). A pipeline that holds none serves a clip
            // rather than a timeline, and the runner then publishes the clip.
            pipeline.splicer(),
        )?
        .serving(timelines);
        let mut runners = vec![
            NamedRunner::new(RunnerId(0), Box::new(encoder)),
            NamedRunner::new(RunnerId(1), Box::new(denoise)),
            NamedRunner::new(RunnerId(2), Box::new(decode)),
        ];
        // A pipeline whose layouts carry a soundtrack decodes it; one whose layouts carry none binds
        // no audio runner.
        if self.serves_track() && !audio_buckets.is_empty() {
            let audio_decoder = pipeline.audio_decoder().context(
                "the denoiser's layouts carry a soundtrack and the checkpoint holds no audio \
                 decoder, so every response would be a clip with no sound",
            )?;
            let audio_set = Buckets::new(audio_buckets);
            let audio = worker::runners::AudioDecoderRunner::new(
                Binding {
                    id: RunnerId(3),
                    phases: served_phases(&audio_set, false),
                    device,
                    stream: stream("audio decoder")?,
                },
                audio_decoder,
                audio_set,
                Arc::clone(generation),
            )?;
            runners.push(NamedRunner::new(RunnerId(3), Box::new(audio)));
        }
        // The artifact's three calls. They are a runner of this device because the products they
        // stage live in this device's memory and their copies are ordered on this device's stream,
        // and they own the process's media lane because encoding and muxing are not device work
        //. The two numbers they work at are the checkpoint's: the frame rate the timeline is
        // measured in and the sample clock the track was decoded at.
        //
        // A rank past the first encodes the units it decoded and nothing else: the reader, the
        // track's encoder and the muxer serve on one rank, and the units meet in the one output.
        let (output, lane) = self.codec(
            media_host::Rates {
                frame_rate: pipeline.frame_rate(),
                audio_rate: pipeline.audio_rate(),
            },
            blocking,
        )?;
        let media = media_host::MediaStage::new(
            Binding {
                id: RunnerId(4),
                phases: if self.serves_output() {
                    host_phases()
                } else {
                    unit_phases()
                },
                device,
                stream: stream("media")?,
            },
            output,
            lane,
            // A video unit's frames are the model's schedule, not this plan's arithmetic.
            {
                let pipeline = Arc::clone(pipeline);
                Arc::new(move |frames| {
                    pipeline.video_units(frames).map_err(|error| {
                        worker::Error::Invalid(format!("a video unit's frames: {error}"))
                    })
                })
            },
            format,
        );
        runners.push(NamedRunner::new(RunnerId(4), Box::new(media)));
        Ok(runners)
    }

    /// The deployment's one output and media lane, which every rank's media stage shares.
    pub(super) fn codec(
        &self,
        rates: worker::runners::media::Rates,
        blocking: &worker::blocking::ThreadPool,
    ) -> Result<SharedCodec> {
        let mut codec = self
            .codec
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(shared) = codec.as_ref() {
            return Ok(shared.clone());
        }
        let shared = (
            Arc::new(worker::runners::media::Output::from_env(rates)?),
            blocking.lane::<worker::runners::media::MediaWork>(BlockingKind::Media)?,
        );
        *codec = Some(shared.clone());
        Ok(shared)
    }

    /// Whether this rank decodes the track: the group's last.
    ///
    /// The units of a timeline go to the ranks in turn from the first, so the last rank is the one
    /// a timeline's remainder leaves free, and the track runs beside the other ranks' last clips.
    pub(super) fn serves_track(&self) -> bool {
        self.rank as usize + 1 == self.world_size
    }

    /// Whether this rank serves the blocks that run on one rank of the group: the audio decoder and
    /// the host stage. The sequence and tensor groups run on every rank, and the video decoder takes
    /// a timeline's units in turn over every rank.
    pub(crate) fn serves_output(&self) -> bool {
        self.rank == 0
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

/// One deployment's output and the media lane its stages run on.
pub(super) type SharedCodec = (
    Arc<worker::runners::media::Output>,
    worker::host::Lane<worker::runners::media::MediaWork>,
);

/// The values of one page of a host latent pool: a mebibyte of the denoiser's float32 samples.
const LATENT_PAGE_VALUES: u32 = 1 << 18;

/// The share of every request's samples a denoising rank of this build holds: the whole video and
/// audio latents, because the velocity is gathered to every rank of a sequence group before the
/// solver's update.
const WHOLE_SAMPLES: &str = "video and audio latents, whole on every rank";

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

/// The widest prompt capacity a deployment captures: what one media batch may present.
pub(super) fn prompt_capacity(video: &proto::config::VideoConfig, longest: u32) -> Result<u32> {
    prompt_capacities(video, longest)?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow::anyhow!("this deployment captures no prompt capacity"))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::prompt_capacities;

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
}
