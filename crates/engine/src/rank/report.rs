//! What a rank advertises of itself: the report its capacity plan is made against.

use anyhow::Result;
use nn::family::ConditionBounds;
use platform::DeviceId;

/// What this rank advertises of itself: the report the capacity plan is made against.
///
/// A causal-language rank states one component - the model's single block - the two forward calls
/// it serves, and the three identities that name this process. It publishes no products (`logits`
/// returns through the batch, not as an exported tensor), so no device product is exported and no
/// media component is routed: the deployment is one class of computation, and saying otherwise
/// would make the engine reserve storage for exports that never happen.
pub(crate) fn rank_report(device: DeviceId, host: &str) -> Result<worker::capacity::Report> {
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    Ok(worker::capacity::Report {
        endpoint: ipc::MainProcessEndpoint {
            host: host.to_owned(),
            // The address space this rank's tensors live in: one main process, one pid.
            address_space: std::process::id().to_string(),
            // This process's start, which the engine compares to notice a relaunched rank.
            incarnation: format!("{at}"),
        },
        device_exports: false,
        fabric_handles: platform::HandleType::probe(device)? == platform::HandleType::Fabric,
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        media_components: std::collections::BTreeMap::new(),
        // The component this rank holds, named as the rank layout names it (`schedulers` places one
        // component over every rank). The scheduler routes a call by component name, so a rank that
        // advertised a name of its own would hold nothing any pool asks for.
        components: vec![ipc::ComponentInfo {
            name: proto::config::DEFAULT_BLOCK.to_owned(),
            outputs: Vec::new(),
        }],
    })
}

/// The largest extents a request of this deployment states, which admission holds a request to.
pub(super) struct MediaBounds {
    /// The prompt's tokens: what the text encoder's largest bucket holds.
    pub(super) prompt_tokens: u32,
    /// The clips of the longest video the deployment serves.
    pub(super) video_units: u32,
    /// The extents a request's conditions may reach: the model's statement.
    pub(super) conditions: ConditionBounds,
    /// One row of the prompt's encoded states: the element type and the width the text encoder
    /// writes, which the engine sizes a stored encoding by.
    pub(super) conditioning_row: (ipc::DType, u32),
}

impl MediaBounds {
    /// The vision tower's patch rows of a prompt, at most.
    pub(super) fn vision_patches(&self) -> u32 {
        self.prompt_tokens
            .saturating_mul(self.conditions.patches_per_token)
    }
}

/// The products `component` publishes, under the names a request's graph asks for them.
///
/// A media rank's products stay in the process, in the product store, and the scheduler books
/// nothing for a rank with no buffer pool. So what this states is the name and the one axis a
/// request's extent binds, with the deployment's real bound on it: admission refuses a request past
/// that bound, which is the check a batch that fits no bucket would otherwise fail much later. The
/// element type and the inner axes are the model's and nothing on such a rank reads them, so a
/// product is stated as bytes along that axis, and a product with no request extent as one byte.
pub(super) fn media_outputs(component: &str, bounds: &MediaBounds) -> Vec<ipc::OutputInfo> {
    use ipc::product;
    let output = |name: &str, extent: Option<u32>| ipc::OutputInfo {
        name: name.to_owned(),
        dtype: ipc::DType::U8,
        shape_bound: ipc::ShapeBound {
            dims: vec![
                extent.map_or(ipc::DimBound::Static(1), |max| ipc::DimBound::Device {
                    max,
                }),
            ],
        },
        raster_axes: None,
    };
    // The prompt's encoded states are rows of the text encoder's width: the engine stores and finds
    // them in the host encoder pool by their bytes, so the row is stated whole.
    let (states, width) = bounds.conditioning_row;
    let conditioning = ipc::OutputInfo {
        name: product::CONDITIONING.to_owned(),
        dtype: states,
        shape_bound: ipc::ShapeBound {
            dims: vec![
                ipc::DimBound::Device {
                    max: bounds.prompt_tokens,
                },
                ipc::DimBound::Static(width),
            ],
        },
        raster_axes: None,
    };
    match component {
        "text_encoder" => vec![
            conditioning,
            output(product::VISION_FEATURES, Some(bounds.prompt_tokens)),
        ],
        "transformer" | "transformer_ref" => vec![
            output(product::VIDEO_LATENTS, None),
            output(product::AUDIO_LATENTS, None),
        ],
        "media_reader" => vec![
            output(product::CONDITION_PIXELS, Some(bounds.conditions.pixels)),
            output(product::CONDITION_SAMPLES, Some(bounds.conditions.samples)),
            output(product::VISION_PIXELS, Some(bounds.vision_patches())),
        ],
        "latent_encoder" => vec![
            output(
                product::CONDITION_VIDEO_LATENTS,
                Some(bounds.conditions.rows),
            ),
            output(
                product::CONDITION_AUDIO_LATENTS,
                Some(bounds.conditions.rows),
            ),
        ],
        "video_decoder" => vec![output(product::VIDEO_UNITS, Some(bounds.video_units))],
        "video_codec" => vec![output(product::ENCODED_UNITS, Some(bounds.video_units))],
        "audio_decoder" => vec![output(product::AUDIO_SAMPLES, None)],
        _ => Vec::new(),
    }
}

/// What a media rank advertises of itself: the calls its blocks serve, and which component serves
/// each.
///
/// **A media rank serves media calls, not forwards**, so the two calls a causal-language report
/// states (prefill and decode) are not among them: a scheduler routes a request's encode, its
/// denoising steps and its decode by call, and each is served by one block of the pipeline.
///
/// **No device product is exported**, for the same reason the text class exports none: what this
/// class hands on - the encoded conditioning rows, the latent trajectory, the decoded clip - stays
/// in the process, through the product store, and a report that claimed otherwise would have the
/// capacity plan reserve storage for transfers no rank makes.
pub(super) fn media_rank_report(
    device: DeviceId,
    host: &str,
    bounds: MediaBounds,
    denoiser: &str,
) -> Result<worker::capacity::Report> {
    use ipc::MediaCall;
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    // The component names are the deployment's own, which the pipeline's blocks answer to: a
    // scheduler routes a call by component name, and a rank that advertised names of its own would
    // hold nothing any pool asks for.
    let mut media_components = std::collections::BTreeMap::new();
    media_components.insert(MediaCall::TextEncoding, "text_encoder".to_owned());
    media_components.insert(MediaCall::LatentPreparation, denoiser.to_owned());
    media_components.insert(MediaCall::Denoising, denoiser.to_owned());
    // A conditioned request's own calls: a rank that takes none states an empty bound for them, and
    // a request's graph holds them only when it carries conditions.
    if bounds.conditions.rows > 0 {
        media_components.insert(MediaCall::MediaReading, "media_reader".to_owned());
        media_components.insert(MediaCall::VisionEncoding, "text_encoder".to_owned());
        media_components.insert(MediaCall::LatentEncoding, "latent_encoder".to_owned());
    }
    media_components.insert(MediaCall::VideoDecoding, "video_decoder".to_owned());
    media_components.insert(MediaCall::AudioDecoding, "audio_decoder".to_owned());
    media_components.insert(MediaCall::VideoEncoding, "video_codec".to_owned());
    media_components.insert(MediaCall::AudioEncoding, "muxer".to_owned());
    media_components.insert(MediaCall::Muxing, "muxer".to_owned());
    let supported_calls: Vec<ipc::CallKind> = media_components
        .keys()
        .copied()
        .map(ipc::CallKind::Media)
        .collect();
    let components: Vec<ipc::ComponentInfo> = media_components
        .values()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|name| ipc::ComponentInfo {
            name: name.clone(),
            outputs: media_outputs(name, &bounds),
        })
        .collect();
    Ok(worker::capacity::Report {
        endpoint: ipc::MainProcessEndpoint {
            host: host.to_owned(),
            address_space: std::process::id().to_string(),
            incarnation: format!("{at}"),
        },
        device_exports: false,
        fabric_handles: platform::HandleType::probe(device)? == platform::HandleType::Fabric,
        supported_calls,
        media_components,
        components,
    })
}
