//! A model's declared components, the call kinds they serve, and placement
//! rules.
//!
//! A model declares one entry per IPC-addressable component: the module
//! paths and methods serving ranks may invoke. `model.describe` reports
//! them with the capability tags of each call's module (the classes it is an
//! instance of, as fixed tag names) and the axis selections each call
//! communicates over. Everything else is decided here, without inspecting
//! Python objects: the call kinds a call serves, which components a rank
//! holds, which media routes it advertises and which placements are legal.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use uniserve_core::ComponentConfig;
use uniserve_ipc::{CallKind, ForwardMode, MediaCall, TransferMode};

use crate::ffi::Record;
use crate::parallel::topology::{HOST_COMPONENTS, is_host_component};
use crate::{Error, Result};

/// A model capability, reported as the tag of the base class a call's
/// module is an instance of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Capability {
    CausalLm,
    DiffusionLm,
    Denoiser,
    ImageDenoiser,
    VideoDenoiser,
    VideoPostprocessor,
    TextEncoder,
    PatchEncoder,
    PatchAutoencoder,
    VideoEncoder,
    AudioEncoder,
    Encoder,
    ImageDecoder,
    VideoDecoder,
    AudioDecoder,
}

impl Capability {
    pub const ALL: [Self; 15] = [
        Self::CausalLm,
        Self::DiffusionLm,
        Self::Denoiser,
        Self::ImageDenoiser,
        Self::VideoDenoiser,
        Self::VideoPostprocessor,
        Self::TextEncoder,
        Self::PatchEncoder,
        Self::PatchAutoencoder,
        Self::VideoEncoder,
        Self::AudioEncoder,
        Self::Encoder,
        Self::ImageDecoder,
        Self::VideoDecoder,
        Self::AudioDecoder,
    ];

    pub const fn tag(self) -> &'static str {
        match self {
            Self::CausalLm => "causal_lm",
            Self::DiffusionLm => "diffusion_lm",
            Self::Denoiser => "denoiser",
            Self::ImageDenoiser => "image_denoiser",
            Self::VideoDenoiser => "video_denoiser",
            Self::VideoPostprocessor => "video_postprocessor",
            Self::TextEncoder => "text_encoder",
            Self::PatchEncoder => "patch_encoder",
            Self::PatchAutoencoder => "patch_autoencoder",
            Self::VideoEncoder => "video_encoder",
            Self::AudioEncoder => "audio_encoder",
            Self::Encoder => "encoder",
            Self::ImageDecoder => "image_decoder",
            Self::VideoDecoder => "video_decoder",
            Self::AudioDecoder => "audio_decoder",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|capability| capability.tag() == tag)
    }
}

/// Which pipeline stages run a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    All,
    First,
    Last,
}

/// The numerical runner family that serves a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RunnerKind {
    Text,
    Diffusion,
    DiffusionLM,
    Decoder,
    Encoder,
}

/// What an encoder call produces, which fixes its input preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    Text,
    Vision,
    VideoCondition,
    AudioCondition,
    Latent,
    Conditioning,
}

/// One declared numerical call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallDecl {
    /// Module path within the model; empty for the root.
    pub path: String,
    pub method: String,
    pub stage: Stage,
    pub capabilities: BTreeSet<Capability>,
    /// Axis selections the call communicates over; empty on a rank that
    /// does not place the component's mesh.
    pub axes: Vec<Vec<String>>,
}

impl CallDecl {
    pub fn is(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// The call kinds this call serves, first matching capability per method.
    pub fn kinds(&self) -> BTreeSet<CallKind> {
        use CallKind::{Forward, Media};
        use MediaCall::*;

        let mut kinds = BTreeSet::new();
        match self.method.as_str() {
            "forward" => {
                if self.is(Capability::CausalLm) {
                    kinds.extend([
                        Forward(ForwardMode::Prefill),
                        Forward(ForwardMode::Decode),
                        Forward(ForwardMode::Verify),
                    ]);
                } else if self.is(Capability::DiffusionLm) {
                    kinds.insert(Forward(ForwardMode::TokenDenoising));
                } else if self.is(Capability::Denoiser) {
                    kinds.extend([Media(LatentPreparation), Media(Denoising)]);
                } else if self.is(Capability::VideoPostprocessor) {
                    kinds.insert(Media(VideoDecoding));
                }
            }
            "encode" => {
                let kind = if self.is(Capability::TextEncoder) {
                    Some(TextEncoding)
                } else if self.is(Capability::PatchEncoder) {
                    Some(VisionEncoding)
                } else if self.is(Capability::PatchAutoencoder)
                    || self.is(Capability::VideoEncoder)
                    || self.is(Capability::AudioEncoder)
                {
                    Some(LatentEncoding)
                } else if self.is(Capability::Encoder) {
                    Some(LatentPreparation)
                } else {
                    None
                };
                kinds.extend(kind.map(Media));
            }
            "decode" => {
                let kind = if self.is(Capability::ImageDecoder) {
                    Some(ImageDecoding)
                } else if self.is(Capability::VideoDecoder) {
                    Some(VideoDecoding)
                } else if self.is(Capability::AudioDecoder) {
                    Some(AudioDecoding)
                } else {
                    None
                };
                kinds.extend(kind.map(Media));
            }
            _ => {}
        }
        kinds
    }

    /// Helper methods a runner calls beside a serving method: a causal LM's
    /// embedding and logits, and a token denoiser's logits.
    fn auxiliary(&self) -> bool {
        (self.is(Capability::CausalLm)
            && matches!(self.method.as_str(), "embed_input_ids" | "compute_logits"))
            || (self.is(Capability::DiffusionLm) && self.method == "compute_logits")
    }

    /// The runner family that serves this call's module, if any.
    pub fn runner(&self) -> Option<RunnerKind> {
        if self.is(Capability::CausalLm) {
            Some(RunnerKind::Text)
        } else if self.is(Capability::Denoiser) {
            Some(RunnerKind::Diffusion)
        } else if self.is(Capability::DiffusionLm) {
            Some(RunnerKind::DiffusionLM)
        } else if [
            Capability::AudioDecoder,
            Capability::ImageDecoder,
            Capability::VideoDecoder,
            Capability::VideoPostprocessor,
        ]
        .iter()
        .any(|capability| self.is(*capability))
        {
            Some(RunnerKind::Decoder)
        } else if [
            Capability::Encoder,
            Capability::PatchAutoencoder,
            Capability::VideoEncoder,
            Capability::AudioEncoder,
        ]
        .iter()
        .any(|capability| self.is(*capability))
        {
            Some(RunnerKind::Encoder)
        } else {
            None
        }
    }

    /// The encoder family of an encoder call.
    pub fn encoder(&self) -> EncoderKind {
        if self.is(Capability::TextEncoder) {
            EncoderKind::Text
        } else if self.is(Capability::PatchEncoder) {
            EncoderKind::Vision
        } else if self.is(Capability::VideoEncoder) {
            EncoderKind::VideoCondition
        } else if self.is(Capability::AudioEncoder) {
            EncoderKind::AudioCondition
        } else if self.is(Capability::PatchAutoencoder) {
            EncoderKind::Latent
        } else {
            EncoderKind::Conditioning
        }
    }
}

/// Call kinds of the host components, which own no numerical method.
pub fn host_kinds(name: &str) -> &'static [MediaCall] {
    match name {
        "media_reader" => &[MediaCall::MediaReading],
        "video_codec" => &[MediaCall::VideoEncoding],
        "muxer" => &[MediaCall::AudioEncoding, MediaCall::Muxing],
        _ => &[],
    }
}

/// Every declared component and its calls, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Declarations {
    entries: IndexMap<String, Vec<CallDecl>>,
}

impl Declarations {
    /// Decode `model.describe`'s `entries`.
    ///
    /// A call no runner can execute (other than a runner's helper methods)
    /// is refused. A model that reconstructs video also declares the host
    /// components that encode and assemble its units.
    pub fn from_record(description: &Record) -> Result<Self> {
        let mut entries = IndexMap::new();
        for entry in description.records("entries")? {
            let name = entry.string("name")?;
            let mut calls = Vec::new();
            for call in entry.records("calls")? {
                let stage = match call.string("stage")?.as_str() {
                    "all" => Stage::All,
                    "first" => Stage::First,
                    "last" => Stage::Last,
                    other => {
                        return Err(Error::Interface(format!(
                            "model.describe reports the unknown stage {other:?}"
                        )));
                    }
                };
                let capabilities = call
                    .strings("capabilities")?
                    .iter()
                    .map(|tag| {
                        Capability::from_tag(tag).ok_or_else(|| {
                            Error::Interface(format!(
                                "model.describe reports the unknown capability {tag:?}"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?;
                let axes = call
                    .list("axes")?
                    .into_iter()
                    .map(|axes| {
                        tvm_ffi::Array::<tvm_ffi::String>::try_from(axes)
                            .map(|axes| axes.iter().map(|axis| axis.as_str().to_owned()).collect())
                            .map_err(|error| Error::numerical("model.describe", &error))
                    })
                    .collect::<Result<_>>()?;
                calls.push(CallDecl {
                    path: call.string("path")?,
                    method: call.string("method")?,
                    stage,
                    capabilities,
                    axes,
                });
            }
            entries.insert(name, calls);
        }
        Self::new(entries)
    }

    /// Check calls and add the host components a video model needs.
    pub fn new(mut entries: IndexMap<String, Vec<CallDecl>>) -> Result<Self> {
        for call in entries.values().flatten() {
            if call.kinds().is_empty() && !call.auxiliary() {
                return Err(Error::Unsupported(format!(
                    "worker cannot execute capability {}.{}",
                    call.path, call.method
                )));
            }
        }
        if entries
            .values()
            .flatten()
            .any(|call| call.is(Capability::VideoPostprocessor))
        {
            for name in HOST_COMPONENTS {
                entries.insert(name.to_owned(), Vec::new());
            }
        }
        Ok(Self { entries })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn calls(&self, name: &str) -> &[CallDecl] {
        self.entries.get(name).map_or(&[], Vec::as_slice)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[CallDecl])> {
        self.entries
            .iter()
            .map(|(name, calls)| (name.as_str(), calls.as_slice()))
    }

    /// Any call of any component has `capability`.
    pub fn has(&self, capability: Capability) -> bool {
        self.entries
            .values()
            .flatten()
            .any(|call| call.is(capability))
    }

    /// Check a deployment's placement against the declarations, before any
    /// process group exists.
    pub fn validate(&self, placements: &BTreeMap<String, ComponentConfig>) -> Result<()> {
        let unknown: Vec<&String> = placements
            .keys()
            .filter(|name| !self.entries.contains_key(name.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(Error::Unsupported(format!(
                "unknown components {unknown:?}"
            )));
        }

        for (name, placement) in placements {
            let calls = self.calls(name);
            let temporal = placement.distribution.is_some();
            if calls.is_empty() {
                if !is_host_component(name) {
                    return Err(Error::Unsupported(format!(
                        "component {name:?} has no numerical methods"
                    )));
                }
                if name == "muxer" && temporal {
                    return Err(Error::Unsupported(
                        "the muxer assembles one artifact and is not distributed".into(),
                    ));
                }
                if name == "media_reader" && temporal {
                    return Err(Error::Unsupported(
                        "the media reader reads one request's conditions on one rank and is not distributed".into(),
                    ));
                }
                if name == "video_codec" && (!temporal || placement.units_per_rank != 1) {
                    return Err(Error::Unsupported(
                        "video encoding distributes by temporal_units with one media unit per rank, each rank one codec slot".into(),
                    ));
                }
                continue;
            }

            let video = calls.iter().any(|call| call.is(Capability::VideoDecoder));
            let media = calls
                .iter()
                .any(|call| call.is(Capability::AudioDecoder) || call.is(Capability::VideoEncoder));
            if video {
                if !temporal || placement.units_per_rank != 1 {
                    return Err(Error::Unsupported(
                        "video decoding requires temporal_units with one native unit per rank"
                            .into(),
                    ));
                }
            } else if !media && temporal {
                return Err(Error::Unsupported(format!(
                    "entry {name:?} requires model-parallel membership"
                )));
            }
        }
        Ok(())
    }

    /// The call kinds of one component.
    pub fn component_kinds(&self, name: &str) -> BTreeSet<CallKind> {
        if is_host_component(name) {
            return host_kinds(name)
                .iter()
                .copied()
                .map(CallKind::Media)
                .collect();
        }
        self.calls(name).iter().flat_map(CallDecl::kinds).collect()
    }

    /// Call kinds the components in `held` serve, with tensor transfer and,
    /// for a causal LM, KV export and install.
    pub fn supported(&self, held: &[&str]) -> BTreeSet<CallKind> {
        let mut result = BTreeSet::from([CallKind::Transfer(TransferMode::Tensor)]);
        for &name in held {
            result.extend(self.component_kinds(name));
            if self
                .calls(name)
                .iter()
                .any(|call| call.is(Capability::CausalLm))
            {
                result.extend([
                    CallKind::Transfer(TransferMode::KvExport),
                    CallKind::Transfer(TransferMode::KvInstall),
                ]);
            }
        }
        result
    }

    /// The unique component serving each media call among `held`.
    /// Ambiguous media calls have no route; latent preparation must share
    /// the denoiser's component.
    pub fn routes(&self, held: &[&str]) -> Result<BTreeMap<MediaCall, String>> {
        let mut owners = BTreeMap::<MediaCall, Vec<&str>>::new();
        for &name in held {
            for kind in self.component_kinds(name) {
                if let CallKind::Media(kind) = kind {
                    owners.entry(kind).or_default().push(name);
                }
            }
        }
        let routes: BTreeMap<MediaCall, String> = owners
            .into_iter()
            .filter_map(|(kind, names)| match names.as_slice() {
                [name] => Some((kind, (*name).to_owned())),
                _ => None,
            })
            .collect();
        if let (Some(prepare), Some(denoise)) = (
            routes.get(&MediaCall::LatentPreparation),
            routes.get(&MediaCall::Denoising),
        ) && prepare != denoise
        {
            return Err(Error::Unsupported(
                "latent preparation must participate in the denoiser component".into(),
            ));
        }
        Ok(routes)
    }

    /// Module paths this rank must read from the checkpoint: every path its
    /// placed components call.
    pub fn resident(
        &self,
        rank: usize,
        placements: &BTreeMap<String, ComponentConfig>,
    ) -> BTreeSet<String> {
        placements
            .iter()
            .filter(|(_, placement)| placement.ranks.contains(&rank))
            .flat_map(|(name, _)| self.calls(name))
            .map(|call| call.path.clone())
            .collect()
    }

    /// Axis selections per component for the group plan: every call's
    /// selections, plus `tp` for a causal LM, whose sampling needs it even
    /// with replicated logits.
    pub fn selections(&self) -> BTreeMap<String, BTreeSet<Vec<String>>> {
        self.entries
            .iter()
            .map(|(name, calls)| {
                let mut selections: BTreeSet<Vec<String>> = calls
                    .iter()
                    .flat_map(|call| call.axes.iter().cloned())
                    .collect();
                if calls.iter().any(|call| call.is(Capability::CausalLm)) {
                    selections.insert(vec!["tp".to_owned()]);
                }
                (name.clone(), selections)
            })
            .collect()
    }

    /// Outermost declared paths of a component: a path nested in another
    /// declared path is bound by its parent's traversal, and the empty path
    /// (the model root) contains every other path.
    pub fn roots(&self, name: &str) -> Vec<String> {
        let paths: BTreeSet<&str> = self
            .calls(name)
            .iter()
            .map(|call| call.path.as_str())
            .collect();
        paths
            .iter()
            .filter(|path| {
                !paths.iter().any(|parent| {
                    parent != *path
                        && (parent.is_empty() || path.starts_with(&format!("{parent}.")))
                })
            })
            .map(|path| (*path).to_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{ComponentDistribution, ParallelConfig};

    use super::*;

    fn call(path: &str, method: &str, capabilities: &[Capability]) -> CallDecl {
        CallDecl {
            path: path.into(),
            method: method.into(),
            stage: Stage::All,
            capabilities: capabilities.iter().copied().collect(),
            axes: Vec::new(),
        }
    }

    fn declarations(entries: Vec<(&str, Vec<CallDecl>)>) -> Result<Declarations> {
        Declarations::new(
            entries
                .into_iter()
                .map(|(name, calls)| (name.to_owned(), calls))
                .collect(),
        )
    }

    fn local(ranks: Vec<usize>) -> ComponentConfig {
        ComponentConfig::parallel(ranks, ParallelConfig::default())
    }

    #[test]
    fn a_causal_lm_serves_token_calls_and_kv_transfers() {
        let model = declarations(vec![(
            "model",
            vec![
                call("", "forward", &[Capability::CausalLm]),
                call("", "compute_logits", &[Capability::CausalLm]),
            ],
        )])
        .unwrap();
        let supported = model.supported(&["model"]);
        for kind in [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Verify),
            CallKind::Transfer(TransferMode::Tensor),
            CallKind::Transfer(TransferMode::KvExport),
            CallKind::Transfer(TransferMode::KvInstall),
        ] {
            assert!(supported.contains(&kind), "{kind:?}");
        }
        assert_eq!(supported.len(), 6);
        assert_eq!(
            model.selections()["model"],
            BTreeSet::from([vec!["tp".to_owned()]])
        );
    }

    #[test]
    fn an_unserviceable_capability_is_refused() {
        let error = declarations(vec![(
            "model",
            vec![call("head", "project", &[Capability::CausalLm])],
        )])
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "worker cannot execute capability head.project"
        );
    }

    #[test]
    fn a_video_model_declares_its_host_components() {
        let model = declarations(vec![(
            "video_decoder",
            vec![
                call("decoder", "decode", &[Capability::VideoDecoder]),
                call(
                    "postprocessor",
                    "forward",
                    &[Capability::VideoPostprocessor],
                ),
            ],
        )])
        .unwrap();
        let names: Vec<&str> = model.names().collect();
        assert_eq!(
            names,
            vec!["video_decoder", "media_reader", "video_codec", "muxer"]
        );
        let routes = model.routes(&["video_decoder", "muxer"]).unwrap();
        assert_eq!(routes[&MediaCall::VideoDecoding], "video_decoder");
        assert_eq!(routes[&MediaCall::Muxing], "muxer");

        // Video decoding must be distributed one unit per rank.
        let mut placements = BTreeMap::from([("video_decoder".to_owned(), local(vec![0]))]);
        assert!(model.validate(&placements).is_err());
        placements.insert(
            "video_decoder".to_owned(),
            ComponentConfig {
                distribution: Some(ComponentDistribution::TemporalUnits),
                ..local(vec![0, 1])
            },
        );
        model.validate(&placements).unwrap();
        placements.insert(
            "muxer".to_owned(),
            ComponentConfig {
                distribution: Some(ComponentDistribution::TemporalUnits),
                ..local(vec![0])
            },
        );
        assert!(model.validate(&placements).is_err());
    }

    #[test]
    fn unknown_placements_and_split_latent_routes_are_refused() {
        let model = declarations(vec![
            (
                "text",
                vec![call(
                    "text",
                    "encode",
                    &[Capability::TextEncoder, Capability::Encoder],
                )],
            ),
            (
                "denoiser",
                vec![call("flow", "forward", &[Capability::Denoiser])],
            ),
            (
                "conditioner",
                vec![call("cond", "encode", &[Capability::Encoder])],
            ),
        ])
        .unwrap();
        let placements = BTreeMap::from([("decoder".to_owned(), local(vec![0]))]);
        assert!(model.validate(&placements).is_err());
        // Two owners of latent preparation leave it without a route; with a
        // single foreign owner the route must follow the denoiser.
        let routes = model.routes(&["denoiser", "conditioner"]).unwrap();
        assert!(!routes.contains_key(&MediaCall::LatentPreparation));
        assert_eq!(routes[&MediaCall::Denoising], "denoiser");
        assert_eq!(
            model.component_kinds("text"),
            BTreeSet::from([CallKind::Media(MediaCall::TextEncoding)])
        );
    }

    #[test]
    fn roots_exclude_nested_and_rooted_paths() {
        let model = declarations(vec![
            (
                "model",
                vec![
                    call("backbone", "forward", &[Capability::CausalLm]),
                    call("backbone.head", "compute_logits", &[Capability::CausalLm]),
                    call("vision", "encode", &[Capability::PatchEncoder]),
                ],
            ),
            (
                "root",
                vec![
                    call("", "forward", &[Capability::CausalLm]),
                    call("vision", "encode", &[Capability::PatchEncoder]),
                ],
            ),
        ])
        .unwrap();
        assert_eq!(
            model.roots("model"),
            vec!["backbone".to_owned(), "vision".to_owned()]
        );
        assert_eq!(model.roots("root"), vec![String::new()]);
        let placements = BTreeMap::from([("model".to_owned(), local(vec![1]))]);
        assert_eq!(model.resident(1, &placements).len(), 3);
        assert!(model.resident(0, &placements).is_empty());
    }
}
