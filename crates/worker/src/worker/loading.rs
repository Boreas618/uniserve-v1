//! The model-loading steps of [`Worker::start`](super::Worker::start).
//!
//! Loading runs in two phases around process-group creation:
//!
//! 1. [`Description::describe`] reads the checkpoint configuration and
//!    describes a meta-device skeleton: components, calls and capability
//!    tags. Placement is validated here, before any collective, so a bad
//!    placement cannot leave peers waiting in one.
//! 2. [`LoadedModel::load`] narrows the checkpoint to the modules this rank
//!    calls, applies the launch's weight choices, materializes the modules
//!    on the component meshes and resolves each declared call on the loaded
//!    model.
//!
//! The rank reads what it needs of a loaded model from the owner of each fact:
//! the loaded model's own (its attention backend and numerical formats) here,
//! what its computations serve from the executor's manifest
//! (`crate::worker::computations`).

use std::collections::BTreeSet;

use tvm_ffi::Any;

use crate::config::WorkerLaunch;
use crate::ffi::{Handle, Interpreter, integer_list, list, strings, texts};
use crate::memory::weights::Fingerprints;

/// The library module that reads and loads a rank's model.
const LOADING: &str = "uniserve.loading.rank";
use crate::config::placement::Declarations;
use crate::{Error, Result};

/// The meta-device description of a model and its declarations.
pub(crate) struct Description {
    handle: Handle,
    pub(crate) declarations: Declarations,
    /// Routed-expert module paths, when experts are disaggregated.
    pub(crate) expert_paths: Vec<String>,
    /// The model decodes video, which bounds its request concurrency.
    pub(crate) video_decoder: bool,
}

impl Description {
    /// Read the model's configuration and validate the placement against
    /// it (`read_config`).
    pub(crate) fn describe(launch: &WorkerLaunch) -> Result<Self> {
        let execution = &launch.execution;
        let model = launch.model.as_ref();
        let components = serde_json::to_string(&launch.components).map_err(|error| {
            Error::Invalid(format!("encoding the component placement: {error}"))
        })?;
        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let optional = |value: Option<&str>| value.map(text).unwrap_or_default();
        let generation = execution.generation_device.map(|device| device.to_string());
        let handle = Handle::new(
            Interpreter::import(&format!("{LOADING}.read_config"))?.call_values(&[
                Any::from(launch.stub),
                optional(model.map(|model| model.path.as_str())),
                text(&launch.load.format),
                text(&launch.load.mode),
                optional(launch.load.download_dir.as_deref()),
                launch
                    .load
                    .num_threads
                    .map(|threads| Any::from(i64::from(threads)))
                    .unwrap_or_default(),
                text(&execution.device.to_string()),
                optional(generation.as_deref()),
                text(&components),
                Any::from(execution.rank as i64),
                text(&execution.role),
                Any::from(
                    launch
                        .expert_parallel
                        .as_ref()
                        .is_some_and(|union| union.disaggregated()),
                ),
            ])?,
        )?;
        let method = |name: &str| Interpreter::import(&format!("{LOADING}.Description.{name}"));
        let expert_paths = texts(method("routed_experts")?.call_values(&[handle.to_any()])?)
            .map_err(|error| Error::numerical("routed_experts", &error))?;
        let video_decoder = method("decodes_video")?.call_as(&[handle.view()])?;
        let declarations = Declarations::read(&handle)?;
        declarations.validate(&launch.components)?;
        Ok(Self {
            handle,
            declarations,
            expert_paths,
            video_decoder,
        })
    }
}

/// One component's meshes, passed to loading.
pub(crate) struct ComponentMesh {
    pub(crate) component: String,
    /// Outermost declared paths of the component.
    pub(crate) paths: Vec<String>,
    pub(crate) mesh: Handle,
    pub(crate) units: Option<Handle>,
    /// The sequence-parallel kind of the placement.
    pub(crate) sequence: &'static str,
    /// The Ulysses head projection mode (`gather` or `exchange`) and its
    /// exchange step in rows; other kinds gather.
    pub(crate) projection: (&'static str, Option<usize>),
}

/// The expert-parallel union a model partitions its routed experts over.
pub(crate) struct ExpertPartition {
    /// The torch world holding the union communicator.
    pub(crate) world: Handle,
    pub(crate) attention_ranks: usize,
    pub(crate) dedicated: bool,
}

/// A loaded model and its declarations.
pub(crate) struct LoadedModel {
    handle: Handle,
    pub(crate) declarations: Declarations,
    pub(crate) video_decoder: bool,
    /// The attention backend the model was loaded with.
    pub(crate) attention_backend: String,
    /// Why serving does not capture the model's calls in CUDA graphs: its
    /// attention provider's or one of its modules'; empty when it captures
    /// them all.
    pub(crate) uncaptured: String,
}

/// The module paths a rank loads and those left unloaded below them:
/// dedicated expert ranks hold only the routed experts, and attention ranks
/// of a disaggregated union exclude them.
fn selection<'a>(
    description: &'a Description,
    launch: &WorkerLaunch,
) -> (Vec<String>, &'a [String]) {
    let execution = &launch.execution;
    let dedicated = execution.role == "experts";
    let split = launch
        .expert_parallel
        .as_ref()
        .is_some_and(|union| union.disaggregated());
    let modules: Vec<String> = if dedicated {
        description.expert_paths.clone()
    } else {
        description
            .declarations
            .resident(execution.rank, &launch.components)
            .into_iter()
            .collect()
    };
    let excluded: &[String] = if split && !dedicated {
        &description.expert_paths
    } else {
        &[]
    };
    (modules, excluded)
}

/// The plugin weight source the launch's model streams its tensors from, as
/// an argument: `None` reads the checkpoint's files.
fn weights(launch: &WorkerLaunch) -> Any {
    launch
        .model
        .as_ref()
        .and_then(|model| model.weights.as_deref())
        .map(|weights| Any::from(tvm_ffi::String::from(weights)))
        .unwrap_or_default()
}

/// The precision selectors of the launch's model, as JSON object text.
fn quantization(launch: &WorkerLaunch) -> &str {
    launch
        .model
        .as_ref()
        .map_or("{}", |model| model.quantization.as_str())
}

impl LoadedModel {
    /// The digests of what determines this rank's loaded weights besides
    /// its placement and GPUs (`fingerprint`); `None` for the stub model,
    /// whose weights are not retained.
    pub(crate) fn fingerprint(
        description: &Description,
        launch: &WorkerLaunch,
    ) -> Result<Option<Fingerprints>> {
        let (modules, excluded) = selection(description, launch);
        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let union = launch.expert_parallel.as_ref();
        let fingerprint: tvm_ffi::String = Interpreter::import(&format!("{LOADING}.fingerprint"))?
            .call_values(&[
                description.handle.to_any(),
                Any::from(strings(&modules)),
                Any::from(strings(excluded)),
                text(quantization(launch)),
                text(&launch.execution.model_dtype),
                union.map(|union| text(&union.exchange)).unwrap_or_default(),
                Any::from(union.map_or(0, |union| union.attention_ranks as i64)),
                weights(launch),
            ])?
            .try_into()
            .map_err(|error| Error::numerical("fingerprint", &error))?;
        if fingerprint.as_str().is_empty() {
            return Ok(None);
        }
        serde_json::from_str(fingerprint.as_str())
            .map(Some)
            .map_err(|error| Error::Interface(format!("the model's fingerprints: {error}")))
    }

    /// Materialize this rank's modules (`load_model`), or bind `retained`,
    /// the library's tensors this rank retained when it last loaded the
    /// same model on the same placement, and read the tensors `load` names
    /// from the checkpoint. The checkpoint is narrowed to the rank's
    /// resident modules (`selection`).
    pub(crate) fn load(
        description: Description,
        launch: &WorkerLaunch,
        meshes: &[ComponentMesh],
        experts: Option<&ExpertPartition>,
        host_only: bool,
        retained: Option<&Handle>,
        load: &BTreeSet<String>,
    ) -> Result<Self> {
        let handle = Self::materialize(
            &description,
            launch,
            meshes,
            experts,
            host_only,
            retained,
            load,
            false,
        )?;
        Self::from_handle(description, handle)
    }

    /// Read numerical facts from a materialized rank, including a rank
    /// whose storage has been allocated without checkpoint values.
    pub(crate) fn from_handle(description: Description, handle: Handle) -> Result<Self> {
        let backend: tvm_ffi::String =
            Interpreter::import(&format!("{LOADING}.attention_backend"))?
                .call_as(&[handle.view()])?;
        let uncaptured: tvm_ffi::String =
            Interpreter::import(&format!("{LOADING}.uncaptured"))?.call_as(&[handle.view()])?;
        Ok(Self {
            handle,
            declarations: description.declarations,
            video_decoder: description.video_decoder,
            attention_backend: backend.as_str().to_owned(),
            uncaptured: uncaptured.as_str().to_owned(),
        })
    }

    /// This rank's modules materialized as [`LoadedModel::load`] loads
    /// them, but without reading the checkpoint (`rank.load_model`'s
    /// `unread`): every tensor has its storage, shape, element type and
    /// encoding, not its value. Returns the `rank.Loaded` handle, from
    /// which a worker plans the layout of the weights it receives from
    /// another world.
    pub(crate) fn unread(
        description: &Description,
        launch: &WorkerLaunch,
        meshes: &[ComponentMesh],
        experts: Option<&ExpertPartition>,
    ) -> Result<Handle> {
        Self::materialize(
            description,
            launch,
            meshes,
            experts,
            false,
            None,
            &BTreeSet::new(),
            true,
        )
    }

    /// The `rank.Loaded` handle of `rank.load_model` with the launch's
    /// arguments.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call of the library's loader with the launch's choices"
    )]
    fn materialize(
        description: &Description,
        launch: &WorkerLaunch,
        meshes: &[ComponentMesh],
        experts: Option<&ExpertPartition>,
        host_only: bool,
        retained: Option<&Handle>,
        load: &BTreeSet<String>,
        unread: bool,
    ) -> Result<Handle> {
        let execution = &launch.execution;
        let (modules, excluded) = selection(description, launch);

        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let optional = |value: Option<i64>| value.map(Any::from).unwrap_or_default();
        let generation = execution.generation_device.map(|device| device.to_string());
        let attention = &launch.attention;
        // Each component's paths cross back to back, with their counts.
        let path_counts: Vec<i64> = meshes.iter().map(|mesh| mesh.paths.len() as i64).collect();
        let paths: Vec<&String> = meshes.iter().flat_map(|mesh| &mesh.paths).collect();
        let (world, attention_ranks, expert_dedicated) = match &experts {
            Some(experts) => (
                experts.world.to_any(),
                experts.attention_ranks as i64,
                experts.dedicated,
            ),
            None => (Any::default(), 0, false),
        };
        Handle::new(
            Interpreter::import(&format!("{LOADING}.load_model"))?.call_values(&[
                description.handle.to_any(),
                Any::from(strings(&modules)),
                Any::from(strings(excluded)),
                text(quantization(launch)),
                text(&execution.model_dtype),
                text(&execution.device.to_string()),
                generation.as_deref().map(text).unwrap_or_default(),
                text(&execution.role),
                Any::from(host_only),
                Any::from(strings(meshes.iter().map(|mesh| &mesh.component))),
                Any::from(integer_list(path_counts)),
                Any::from(strings(paths)),
                Any::from(list(meshes.iter().map(|mesh| mesh.mesh.to_any()))),
                Any::from(list(meshes.iter().map(|mesh| {
                    mesh.units.as_ref().map(Handle::to_any).unwrap_or_default()
                }))),
                Any::from(strings(meshes.iter().map(|mesh| mesh.sequence))),
                Any::from(strings(meshes.iter().map(|mesh| mesh.projection.0))),
                Any::from(list(
                    meshes
                        .iter()
                        .map(|mesh| optional(mesh.projection.1.map(|rows| rows as i64))),
                )),
                world,
                Any::from(attention_ranks),
                Any::from(expert_dedicated),
                launch
                    .expert_parallel
                    .as_ref()
                    .map(|union| text(&union.exchange))
                    .unwrap_or_default(),
                Any::from(launch.tensor_experts),
                retained.map(Handle::to_any).unwrap_or_default(),
                Any::from(strings(load)),
                text(execution.attention_backend.as_str()),
                Any::from(attention.workspace_size as i64),
                text(&attention.decode_backend),
                text(&attention.prefill_backend),
                Any::from(attention.disable_split_kv),
                optional(attention.decode_split_tile_size.map(i64::from)),
                optional(attention.prefill_split_tile_size.map(i64::from)),
                attention.use_tensor_core.map(Any::from).unwrap_or_default(),
                weights(launch),
                Any::from(unread),
            ])?,
        )
    }

    pub(crate) fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The storage formats of the loaded parameters (`weight_formats`).
    pub(crate) fn weight_formats(&self) -> Result<Vec<String>> {
        self.formats("weight_formats")
    }

    /// The input formats of the loaded linear layers
    /// (`activation_formats`).
    pub(crate) fn activation_formats(&self) -> Result<Vec<String>> {
        self.formats("activation_formats")
    }

    fn formats(&self, function: &str) -> Result<Vec<String>> {
        let value = Interpreter::import(&format!("{LOADING}.{function}"))?
            .call_values(&[self.handle.to_any()])?;
        texts(value).map_err(|error| Error::numerical(function, &error))
    }
}

/// The Ulysses head projection of a placement: its mode name and exchange
/// step in rows. Kinds other than `ulysses` gather.
pub(crate) fn head_projection(placement: &ipc::ComponentConfig) -> (&'static str, Option<usize>) {
    match placement.parallel_config.sequence_parallel {
        ipc::SequenceParallel::Ulysses {
            projection,
            projection_rows,
            ..
        } => (
            match projection {
                ipc::HeadProjection::Gather => "gather",
                ipc::HeadProjection::Exchange => "exchange",
            },
            projection_rows,
        ),
        _ => ("gather", None),
    }
}

/// The sequence-parallel kind name of a placement.
pub(crate) fn sequence_kind(placement: &ipc::ComponentConfig) -> &'static str {
    match placement.parallel_config.sequence_parallel {
        ipc::SequenceParallel::Local => "local",
        ipc::SequenceParallel::Ulysses { .. } => "ulysses",
        ipc::SequenceParallel::Allgather { .. } => "allgather",
        ipc::SequenceParallel::Hybrid { .. } => "hybrid",
    }
}
