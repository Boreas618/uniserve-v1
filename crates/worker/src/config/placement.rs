//! Component placement: the components a model declares, the call kinds
//! they serve, the placement rules, and the components placed on this rank.
//!
//! A model declares one entry per IPC-addressable component: the module
//! paths and methods serving ranks may invoke. The description `read_config`
//! returns reports them, one flat list per method, with the capability tags
//! of each call's module (the classes it is an instance of, as fixed tag
//! names) and the axis selections each component communicates over.
//! Everything else is decided here, without inspecting Python objects: the
//! call kinds a call serves, which components a rank holds, which media
//! routes it advertises and which placements are legal.
//!
//! [`Components`] records the components this rank holds once start has
//! built their meshes; the runners look up the components serving a call
//! kind or runner family there.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use tvm_ffi::Any;
use uniserve_core::ComponentConfig;
use uniserve_ipc::{CallKind, ForwardMode, MediaCall, TransferMode};

use super::Device;
use crate::ffi::{Handle, Interpreter, integers, texts};
use crate::parallel::topology::{HOST_COMPONENTS, is_host_component};
use crate::{Error, Result};

/// A model capability, reported as the tag of the base class a call's
/// module is an instance of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Capability {
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
    pub(crate) const ALL: [Self; 15] = [
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

    pub(crate) const fn tag(self) -> &'static str {
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

    pub(crate) fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|capability| capability.tag() == tag)
    }
}

/// Which pipeline stages run a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    All,
    First,
    Last,
}

/// The numerical runner family that serves a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum RunnerKind {
    Text,
    Diffusion,
    DiffusionLM,
    Decoder,
    Encoder,
}

/// One declared numerical call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CallDecl {
    /// Module path within the model; empty for the root.
    pub path: String,
    pub method: String,
    pub stage: Stage,
    pub capabilities: BTreeSet<Capability>,
}

impl CallDecl {
    pub(crate) fn is(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// The call kinds this call serves, first matching capability per method.
    pub(crate) fn kinds(&self) -> BTreeSet<CallKind> {
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
    pub(crate) fn runner(&self) -> Option<RunnerKind> {
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
}

/// Call kinds of the host components, which own no numerical method.
pub(crate) fn host_kinds(name: &str) -> &'static [MediaCall] {
    match name {
        "media_reader" => &[MediaCall::MediaReading],
        uniserve_ipc::VIDEO_CODEC_COMPONENT => &[MediaCall::VideoEncoding],
        "muxer" => &[MediaCall::AudioEncoding, MediaCall::Muxing],
        _ => &[],
    }
}

/// Every declared component and its calls, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Declarations {
    entries: IndexMap<String, Vec<CallDecl>>,
    /// Axis selections each component's calls communicate over; empty for a
    /// component whose mesh this rank does not place.
    axes: BTreeMap<String, BTreeSet<Vec<String>>>,
}

/// The library class whose methods report a description's declarations.
const DESCRIPTION: &str = "uniserve.loading.rank.Description";

/// Split a concatenated column into consecutive runs of `lengths`.
fn runs<T>(lengths: &[i64], values: Vec<T>) -> Vec<Vec<T>> {
    let mut values = values.into_iter();
    lengths
        .iter()
        .map(|&length| values.by_ref().take(length as usize).collect())
        .collect()
}

impl Declarations {
    /// Read the declarations of the `description` that `read_config`
    /// returned.
    ///
    /// A call no runner can execute (other than a runner's helper methods)
    /// is refused. A model that reconstructs video also declares the host
    /// components that encode and assemble its units.
    pub(crate) fn read(description: &Handle) -> Result<Self> {
        let call = |method: &str, component: Option<&str>| -> Result<Any> {
            let mut arguments = vec![description.to_any()];
            if let Some(component) = component {
                arguments.push(Any::from(tvm_ffi::String::from(component)));
            }
            Interpreter::import(&format!("{DESCRIPTION}.{method}"))?.call_values(&arguments)
        };
        let strings = |method: &str, component: Option<&str>| -> Result<Vec<String>> {
            texts(call(method, component)?).map_err(|error| Error::numerical(method, &error))
        };
        let counts = |method: &str, component: &str| -> Result<Vec<i64>> {
            integers(call(method, Some(component))?)
                .map_err(|error| Error::numerical(method, &error))
        };

        let mut entries = IndexMap::new();
        let mut axes = BTreeMap::new();
        for name in strings("components", None)? {
            let component = Some(name.as_str());
            let paths = strings("call_paths", component)?;
            let methods = strings("call_methods", component)?;
            let stages = strings("call_stages", component)?;
            let capabilities = runs(
                &counts("capability_counts", &name)?,
                strings("capabilities", component)?,
            );

            let mut calls = Vec::with_capacity(paths.len());
            for (((path, method), stage), tags) in
                paths.into_iter().zip(methods).zip(stages).zip(capabilities)
            {
                let stage = match stage.as_str() {
                    "all" => Stage::All,
                    "first" => Stage::First,
                    "last" => Stage::Last,
                    other => {
                        return Err(Error::Interface(format!(
                            "{DESCRIPTION}.call_stages reports the unknown stage {other:?}"
                        )));
                    }
                };
                let capabilities = tags
                    .iter()
                    .map(|tag| {
                        Capability::from_tag(tag).ok_or_else(|| {
                            Error::Interface(format!(
                                "{DESCRIPTION}.capabilities reports the unknown capability {tag:?}"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?;
                calls.push(CallDecl {
                    path,
                    method,
                    stage,
                    capabilities,
                });
            }
            let selections = runs(
                &counts("selection_sizes", &name)?,
                strings("selection_axes", component)?,
            );
            axes.insert(name.clone(), selections.into_iter().collect());
            entries.insert(name, calls);
        }
        Ok(Self {
            axes,
            ..Self::new(entries)?
        })
    }

    /// Check calls and add the host components a video model needs. The
    /// components communicate over no axes.
    pub(crate) fn new(mut entries: IndexMap<String, Vec<CallDecl>>) -> Result<Self> {
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
        Ok(Self {
            entries,
            axes: BTreeMap::new(),
        })
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub(crate) fn calls(&self, name: &str) -> &[CallDecl] {
        self.entries.get(name).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &[CallDecl])> {
        self.entries
            .iter()
            .map(|(name, calls)| (name.as_str(), calls.as_slice()))
    }

    /// Any call of any component has `capability`.
    pub(crate) fn has(&self, capability: Capability) -> bool {
        self.entries
            .values()
            .flatten()
            .any(|call| call.is(capability))
    }

    /// Check a deployment's placement against the declarations, before any
    /// process group exists.
    pub(crate) fn validate(&self, placements: &BTreeMap<String, ComponentConfig>) -> Result<()> {
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
                if name == uniserve_ipc::VIDEO_CODEC_COMPONENT
                    && (!temporal || placement.units_per_rank != 1)
                {
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
    pub(crate) fn component_kinds(&self, name: &str) -> BTreeSet<CallKind> {
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
    /// for a causal LM, cache export and install.
    pub(crate) fn supported(&self, held: &[&str]) -> BTreeSet<CallKind> {
        let mut result = BTreeSet::from([CallKind::Transfer(TransferMode::Tensor)]);
        for &name in held {
            result.extend(self.component_kinds(name));
            if self
                .calls(name)
                .iter()
                .any(|call| call.is(Capability::CausalLm))
            {
                result.extend([
                    CallKind::Transfer(TransferMode::CacheExport),
                    CallKind::Transfer(TransferMode::CacheInstall),
                ]);
            }
        }
        result
    }

    /// The unique component serving each media call among `held`.
    /// Ambiguous media calls have no route; latent preparation must share
    /// the denoiser's component.
    pub(crate) fn routes(&self, held: &[&str]) -> Result<BTreeMap<MediaCall, String>> {
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
    pub(crate) fn resident(
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

    /// Axis selections per component for the group plan: the selections
    /// its calls communicate over, plus `tp` for a causal LM, whose sampling
    /// needs it even with replicated logits.
    pub(crate) fn selections(&self) -> BTreeMap<String, BTreeSet<Vec<String>>> {
        self.entries
            .iter()
            .map(|(name, calls)| {
                let mut selections = self.axes.get(name).cloned().unwrap_or_default();
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
    pub(crate) fn roots(&self, name: &str) -> Vec<String> {
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

/// One component placed on this rank.
pub(crate) struct Placed {
    pub name: String,
    pub config: ComponentConfig,
    /// Declared calls; empty for a host component.
    pub calls: Vec<CallDecl>,
    pub kinds: BTreeSet<CallKind>,
    /// The device the component's bindings compute on: the rank's primary
    /// device. Modules a tower placement puts on a generation device run
    /// there through the bindings' transfers.
    pub device: Device,
    /// The component's numerical mesh (a `DeviceMesh` handle), if bound.
    pub mesh: Option<Handle>,
    /// The temporal unit ring of a distributed component.
    pub units: Option<Handle>,
}

impl Placed {
    pub(crate) fn host(&self) -> bool {
        is_host_component(&self.name)
    }

    /// The runner families serving this component's calls.
    pub(crate) fn runners(&self) -> BTreeSet<RunnerKind> {
        self.calls.iter().filter_map(CallDecl::runner).collect()
    }

    pub(crate) fn has(&self, capability: Capability) -> bool {
        self.calls.iter().any(|call| call.is(capability))
    }
}

/// The placement a rank's group was launched with, which every rank of the
/// group shares: each component's members by name, and the group's rank
/// count. The rank reads it from its launch; its report does not carry it.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GroupPlacement {
    pub(crate) components: BTreeMap<String, ComponentConfig>,
    pub(crate) world_size: usize,
}

impl GroupPlacement {
    /// Whether `rank` is one of `component`'s members.
    pub(crate) fn holds(&self, rank: usize, component: &str) -> bool {
        self.components
            .get(component)
            .is_some_and(|config| config.ranks.contains(&rank))
    }

    /// Whether `rank` reports `component`'s host results: the component's
    /// first member. A component the launch does not place is reported by
    /// the only rank of a one-rank group, and by no rank of a larger one.
    pub(crate) fn reports_output(&self, rank: usize, component: &str) -> Result<bool> {
        match self.components.get(component) {
            Some(config) => Ok(config.ranks.first() == Some(&rank)),
            None if self.world_size == 1 => Ok(true),
            None => Err(Error::Unsupported(format!(
                "component {component:?} has no output rank"
            ))),
        }
    }
}

/// The components placed on this rank.
#[derive(Default)]
pub(crate) struct Components {
    placed: Vec<Placed>,
}

impl Components {
    /// Record the components of `declarations` placed on this rank.
    /// `meshes` supplies each member component's mesh and unit ring.
    pub(crate) fn new(
        declarations: &Declarations,
        placements: &BTreeMap<String, ComponentConfig>,
        rank: usize,
        primary: Device,
        mut meshes: impl FnMut(&str) -> (Option<Handle>, Option<Handle>),
    ) -> Self {
        let placed = placements
            .iter()
            .filter(|(_, config)| config.ranks.contains(&rank))
            .map(|(name, config)| {
                let kinds = declarations.component_kinds(name);
                let (mesh, units) = meshes(name);
                Placed {
                    name: name.clone(),
                    config: config.clone(),
                    calls: declarations.calls(name).to_vec(),
                    kinds,
                    device: primary,
                    mesh,
                    units,
                }
            })
            .collect();
        Self { placed }
    }

    pub(crate) fn placed(&self) -> &[Placed] {
        &self.placed
    }

    pub(crate) fn get(&self, name: &str) -> Option<&Placed> {
        self.placed.iter().find(|placed| placed.name == name)
    }

    /// Components a runner family serves on this rank.
    pub(crate) fn for_runner(&self, runner: RunnerKind) -> impl Iterator<Item = &Placed> {
        self.placed
            .iter()
            .filter(move |placed| placed.runners().contains(&runner))
    }

    /// Whether this rank holds a host component (media reading, video
    /// encoding or muxing).
    pub(crate) fn holds_host(&self) -> bool {
        self.placed
            .iter()
            .any(|placed| !host_kinds(&placed.name).is_empty())
    }

    /// Whether every held component is a host component or has no calls,
    /// so the rank needs no numerical weights.
    pub(crate) fn host_only(&self) -> bool {
        self.placed
            .iter()
            .all(|placed| placed.host() || placed.calls.is_empty())
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
    fn a_causal_lm_serves_token_calls_and_cache_transfers() {
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
            CallKind::Transfer(TransferMode::CacheExport),
            CallKind::Transfer(TransferMode::CacheInstall),
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
