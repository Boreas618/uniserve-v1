//! Startup of an image model's lanes: the executor's `image` section, the
//! facts the lanes over the prefix cache lay prompts out with, what the
//! executor's manifest states for capacity planning ([`ImageManifest`]),
//! and what binding reports for the engine ([`ImageInfo`]), including the
//! agreement of every rank's capture order.
//!
//! An image model (BAGEL, SenseNova) serves its vision and latent encoders,
//! its denoiser and its decoder from the executor's prepared lanes
//! (`crate::executor::image`); the worker states the deployment's bounds
//! and graph configuration, and the executor answers with its buckets.

use std::collections::BTreeMap;

use ipc::{CapturedImage, ImageCapacity, ImageWarmup, WorkerInfo};
use serde::Deserialize;

use crate::config::WorkerConfig;
use crate::config::placement::{Capability, Components};
use crate::ffi::{DType, Handle, Interpreter};
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// The capabilities an image model's lanes serve.
const IMAGE: [Capability; 4] = [
    Capability::ImageDenoiser,
    Capability::PatchEncoder,
    Capability::PatchAutoencoder,
    Capability::ImageDecoder,
];

/// Whether the rank holds a device component of an image model: one that
/// declares an image denoiser, a vision or latent encoder, or an image
/// decoder, of a model that is not a video model (`video`).
pub(crate) fn holds(components: &Components, video: bool) -> bool {
    !video
        && components
            .placed()
            .iter()
            .any(|placed| !placed.host() && IMAGE.iter().any(|&capability| placed.has(capability)))
}

/// The executor's `image` section: the deployment's batch bounds, its
/// request slots and the image sizes the latent encoder and the decoder
/// capture (`flow_shapes`, `[height, width]`). Whether the lanes capture
/// graphs is the settings' top-level `graphs`.
pub(crate) fn settings(config: &WorkerConfig) -> serde_json::Value {
    serde_json::json!({
        "max_batch_calls": config.max_batch_calls,
        "max_batch_tokens": config.max_batch_tokens,
        "max_request_slots": config.max_request_pool_size,
        "max_model_len": config.max_model_len,
        "flow_shapes": config
            .flow_graph_shapes
            .iter()
            .map(|&(height, width)| [height, width])
            .collect::<Vec<_>>(),
    })
}

/// What the lanes over the prefix cache lay a model's prompts out with.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub(crate) struct TextFacts {
    /// The model prepares request images (it encodes them).
    pub(crate) image_processor: bool,
    /// The rotary axes of rows placing image features: one, or three
    /// (temporal, row and column); `None` for a model placing none.
    pub(crate) media_axes: Option<usize>,
}

impl TextFacts {
    /// The facts of a loaded model (`uniserve.executor.image.text_settings`).
    pub(crate) fn of(loaded: &Handle) -> Result<Self> {
        let function = Interpreter::import("uniserve.executor.image.text_settings")?;
        let text: tvm_ffi::String = function.call_as(&[loaded.view()])?;
        serde_json::from_str(text.as_str())
            .map_err(|error| Error::Interface(format!("{}: {error}", function.name())))
    }
}

/// The latent rows of an image denoiser, which size its latent pool.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LatentManifest {
    /// Latent rows of the longest image sequence.
    pub(crate) rows: usize,
    /// Values per latent row.
    pub(crate) width: usize,
    /// The latent values' element type.
    pub(crate) dtype: String,
}

impl LatentManifest {
    pub(crate) fn dtype(&self) -> Result<DType> {
        DType::from_name(&self.dtype).ok_or_else(|| {
            Error::Interface(format!(
                "the executor names the unknown latent dtype {:?}",
                self.dtype
            ))
        })
    }
}

/// What the executor's manifest states of an image model's lanes for
/// capacity planning (`manifest["image"]`).
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ImageManifest {
    /// The image denoiser's latents; `None` without one.
    #[serde(default)]
    pub(crate) latent: Option<LatentManifest>,
    /// Bytes of one image's vision features.
    #[serde(default)]
    pub(crate) vision_feature_bytes: u64,
    /// Bytes of one image's latent features.
    #[serde(default)]
    pub(crate) latent_feature_bytes: u64,
    /// Device bytes the image lanes allocate whatever the slot count.
    #[serde(default)]
    pub(crate) fixed_bytes: u64,
    /// Device bytes they allocate per request slot.
    #[serde(default)]
    pub(crate) slot_bytes: u64,
}

impl ImageManifest {
    /// Device bytes the image lanes allocate for `request_slots` slots.
    pub(crate) fn reserved_bytes(&self, request_slots: usize) -> u64 {
        self.fixed_bytes
            .saturating_add(self.slot_bytes.saturating_mul(request_slots as u64))
    }
}

/// What binding the executor reports of an image model's lanes
/// (`bind()["image"]`), for the engine.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ImageInfo {
    #[serde(default)]
    pub(crate) warmups: Vec<ImageWarmup>,
    #[serde(default)]
    pub(crate) captured_images: Option<Vec<CapturedImage>>,
    #[serde(default)]
    pub(crate) captured_latents: Option<Vec<[u32; 2]>>,
    #[serde(default)]
    pub(crate) unconditional_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub(crate) capacity: Option<ImageCapacity>,
    /// Each component's SHA-256 of its graph keys in capture order.
    #[serde(default)]
    pub(crate) digests: BTreeMap<String, String>,
}

impl ImageInfo {
    /// Report the lanes' shapes and extents in `info`.
    pub(crate) fn apply(&self, info: &mut WorkerInfo) {
        info.image_warmups.clone_from(&self.warmups);
        info.captured_images.clone_from(&self.captured_images);
        info.captured_latents.clone_from(&self.captured_latents);
        info.image_unconditional_token_ids
            .clone_from(&self.unconditional_token_ids);
        info.image_capacity = self.capacity;
    }
}

/// Check, collectively over `group` (every rank of the worker group, at the
/// same point of startup), that every rank holding a component of an image
/// model captures that component's graph keys in the same order: a capture
/// in warmup may run the component's collectives, which its ranks must
/// reach in one order. `digests` holds this rank's digest of each component
/// it serves image lanes of; ranks holding other components state others.
pub(crate) fn agree(digests: &BTreeMap<String, String>, group: &HostGroup) -> Result<()> {
    let encoded = serde_json::to_vec(digests)
        .map_err(|error| Error::Invariant(format!("encoding image graph digests: {error}")))?;
    let mut agreed: BTreeMap<String, String> = BTreeMap::new();
    for stated in group.all_gather_bytes(&encoded)? {
        let stated: BTreeMap<String, String> = serde_json::from_slice(&stated)
            .map_err(|error| Error::Interface(format!("a rank's image graph digests: {error}")))?;
        for (component, digest) in stated {
            match agreed.get(&component) {
                Some(known) if *known != digest => {
                    return Err(Error::Invalid(format!(
                        "the ranks of {component:?} capture different image graph keys: \
                         digest {known} differs from {digest}"
                    )));
                }
                Some(_) => {}
                None => {
                    agreed.insert(component, digest);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::time::Duration;

    use super::*;
    use crate::parallel::host::Rendezvous;

    /// Run `agree` on a loopback group, one member per entry of `stated`;
    /// each member's result.
    fn agreed(stated: &[&[(&str, &str)]]) -> Vec<bool> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut listener = Some(listener);
        let size = stated.len();
        std::thread::scope(|scope| {
            let handles: Vec<_> = stated
                .iter()
                .enumerate()
                .map(|(rank, digests)| {
                    let rendezvous = match listener.take().filter(|_| rank == 0) {
                        Some(listener) => Rendezvous::serve(address.clone(), listener),
                        None => Rendezvous::connect(address.clone()),
                    };
                    let digests: BTreeMap<String, String> = digests
                        .iter()
                        .map(|&(component, digest)| (component.to_owned(), digest.to_owned()))
                        .collect();
                    scope.spawn(move || {
                        let group =
                            HostGroup::connect(rendezvous, rank, size, Duration::from_secs(30))
                                .unwrap();
                        agree(&digests, &group).is_ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    /// The ranks of each component agree on its capture order; ranks
    /// holding other components, or none, state other digests or none.
    #[test]
    fn each_components_ranks_agree_on_its_capture_order() {
        let model = [("model", "a")];
        let both = [("model", "a"), ("vision", "v")];
        let vision = [("vision", "v")];
        assert_eq!(
            agreed(&[&model, &both, &vision, &[]]),
            [true, true, true, true]
        );
        // A rank capturing the model's keys in another order is refused on
        // every rank.
        let other = [("model", "b")];
        assert_eq!(agreed(&[&model, &other]), [false, false]);
    }

    /// Binding's report fills the engine's image fields, and the manifest's
    /// image section reserves its fixed and per-slot bytes.
    #[test]
    fn bound_image_info_reaches_the_report() -> Result<()> {
        let info: ImageInfo = serde_json::from_value(serde_json::json!({
            "warmups": [{"height": 512, "width": 512, "branches": 1, "negative_token_ids": []}],
            "captured_images": [{"height": 512, "width": 512, "branches": 2}],
            "captured_latents": [[512, 512]],
            "unconditional_token_ids": [1, 2],
            "capacity": {
                "max_step_tokens": 4096, "max_step_rows": 3, "max_condition_tokens": 2048,
                "max_condition_rows": 4, "max_vision_patch_rows": 8192, "max_vision_images": 4
            },
            "digests": {"model": "abc"}
        }))
        .map_err(|error| Error::Interface(error.to_string()))?;
        let mut report = WorkerInfo::default();
        info.apply(&mut report);
        assert_eq!(report.image_warmups.len(), 1);
        assert_eq!(report.captured_latents, Some(vec![[512, 512]]));
        assert_eq!(report.image_unconditional_token_ids, Some(vec![1, 2]));
        assert_eq!(
            report.image_capacity.map(|capacity| capacity.max_step_rows),
            Some(3)
        );
        report
            .validate()
            .map_err(|error| Error::Invalid(error.to_string()))?;

        let manifest: ImageManifest = serde_json::from_value(serde_json::json!({
            "latent": {"rows": 4096, "width": 64, "dtype": "bfloat16"},
            "fixed_bytes": 100, "slot_bytes": 10
        }))
        .map_err(|error| Error::Interface(error.to_string()))?;
        assert_eq!(manifest.reserved_bytes(3), 130);
        assert_eq!(
            manifest
                .latent
                .as_ref()
                .map(LatentManifest::dtype)
                .transpose()?,
            Some(DType::BF16)
        );
        Ok(())
    }
}
