//! Model description, loading and component bindings.
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
//! The rank reads what it needs of a loaded model from the owner of each
//! fact: the loaded model's own (its attention backend and numerical
//! formats) here, a capability's from that capability's runner.
//!
//! Component bindings ([`bind`]) and the registry of placed components by
//! capability ([`registry`]) follow.

pub mod bind;
pub mod components;
pub mod interface;
pub mod registry;

use tvm_ffi::Any;

use crate::config::{Device, WorkerLaunch};
use crate::ffi::{Handle, integer_list, list, strings, texts};

/// The library module that reads and loads a rank's model.
const LOADING: &str = "uniserve_models.interface.loading";
use crate::{Error, Result};
use components::Declarations;
use interface::Interface;

/// The meta-device description of a model and its declarations.
pub struct Description {
    handle: Handle,
    /// `module.QualName` of the model class.
    pub model_name: String,
    pub declarations: Declarations,
    /// Routed-expert module paths, when experts are disaggregated.
    pub expert_paths: Vec<String>,
    /// The model decodes video, which bounds its request concurrency.
    pub video_decoder: bool,
}

impl Description {
    /// Read the model's configuration and validate the placement against
    /// it (`read_config`).
    pub fn describe(interface: &Interface, launch: &WorkerLaunch) -> Result<Self> {
        let execution = &launch.execution;
        let model = launch.model.as_ref();
        let components = serde_json::to_string(&launch.components).map_err(|error| {
            Error::Invalid(format!("encoding the component placement: {error}"))
        })?;
        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let optional = |value: Option<&str>| value.map(text).unwrap_or_default();
        let generation = execution.generation_device.map(|device| device.to_string());
        let handle = Handle::new(
            interface
                .import(&format!("{LOADING}.read_config"))?
                .call_values(&[
                    Any::from(launch.stub),
                    optional(model.map(|model| model.path.as_str())),
                    optional(model.and_then(|model| model.base_model.as_deref())),
                    text(&launch.load.format),
                    text(&launch.load.mode),
                    optional(launch.load.download_dir.as_deref()),
                    launch
                        .load
                        .num_threads
                        .map(|threads| Any::from(i64::from(threads)))
                        .unwrap_or_default(),
                    optional(launch.load.checksum_manifest.as_deref()),
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
        let method = |name: &str| interface.import(&format!("{LOADING}.Description.{name}"));
        let model_name: tvm_ffi::String = method("model_name")?.call_as(&[handle.view()])?;
        let expert_paths = texts(method("routed_experts")?.call_values(&[handle.to_any()])?)
            .map_err(|error| Error::numerical("routed_experts", &error))?;
        let video_decoder = method("decodes_video")?.call_as(&[handle.view()])?;
        let declarations = Declarations::read(interface, &handle)?;
        declarations.validate(&launch.components)?;
        Ok(Self {
            handle,
            model_name: model_name.as_str().to_owned(),
            declarations,
            expert_paths,
            video_decoder,
        })
    }
}

/// One component's meshes, passed to loading.
pub struct ComponentMesh {
    pub component: String,
    /// Outermost declared paths of the component.
    pub paths: Vec<String>,
    pub mesh: Handle,
    pub units: Option<Handle>,
    /// The sequence-parallel kind of the placement.
    pub sequence: &'static str,
}

/// The expert-parallel union a model partitions its routed experts over.
pub struct ExpertPartition {
    /// The torch world holding the union communicator.
    pub world: Handle,
    pub attention_ranks: usize,
    pub dedicated: bool,
}

/// A loaded model and its declarations.
pub struct LoadedModel {
    handle: Handle,
    pub model_name: String,
    pub declarations: Declarations,
    pub video_decoder: bool,
    /// The attention backend the model was loaded with.
    pub attention_backend: String,
}

impl LoadedModel {
    /// Materialize this rank's modules (`load_model`). The checkpoint is
    /// narrowed to the rank's resident modules: dedicated expert ranks hold
    /// only the routed experts, and attention ranks of a disaggregated
    /// union exclude them.
    pub fn load(
        interface: &Interface,
        description: Description,
        launch: &WorkerLaunch,
        meshes: Vec<ComponentMesh>,
        experts: Option<ExpertPartition>,
        host_only: bool,
    ) -> Result<Self> {
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
        let handle = Handle::new(
            interface
                .import(&format!("{LOADING}.load_model"))?
                .call_values(&[
                    description.handle.to_any(),
                    Any::from(strings(&modules)),
                    Any::from(strings(excluded)),
                    text(
                        launch
                            .model
                            .as_ref()
                            .map_or("{}", |model| model.quantization.as_str()),
                    ),
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
                    world,
                    Any::from(attention_ranks),
                    Any::from(expert_dedicated),
                    text(execution.attention_backend.as_deref().unwrap_or("auto")),
                    Any::from(attention.workspace_size as i64),
                    text(&attention.decode_backend),
                    text(&attention.prefill_backend),
                    Any::from(attention.disable_split_kv),
                    optional(attention.decode_split_tile_size.map(i64::from)),
                    optional(attention.prefill_split_tile_size.map(i64::from)),
                    attention.use_tensor_core.map(Any::from).unwrap_or_default(),
                ])?,
        )?;
        let backend: tvm_ffi::String = interface
            .import(&format!("{LOADING}.attention_backend"))?
            .call_as(&[handle.view()])?;
        Ok(Self {
            handle,
            model_name: description.model_name,
            declarations: description.declarations,
            video_decoder: description.video_decoder,
            attention_backend: backend.as_str().to_owned(),
        })
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The storage formats of the loaded parameters (`weight_formats`).
    pub fn weight_formats(&self, interface: &Interface) -> Result<Vec<String>> {
        self.formats(interface, "weight_formats")
    }

    /// The input formats of the loaded linear layers
    /// (`activation_formats`).
    pub fn activation_formats(&self, interface: &Interface) -> Result<Vec<String>> {
        self.formats(interface, "activation_formats")
    }

    fn formats(&self, interface: &Interface, function: &str) -> Result<Vec<String>> {
        let value = interface
            .import(&format!("{LOADING}.{function}"))?
            .call_values(&[self.handle.to_any()])?;
        texts(value).map_err(|error| Error::numerical(function, &error))
    }
}

/// The sequence-parallel kind name of a placement.
pub fn sequence_kind(placement: &uniserve_core::ComponentConfig) -> &'static str {
    match placement.parallel_config.sequence_parallel {
        uniserve_core::SequenceParallel::Local => "local",
        uniserve_core::SequenceParallel::Ulysses { .. } => "ulysses",
        uniserve_core::SequenceParallel::Allgather { .. } => "allgather",
        uniserve_core::SequenceParallel::Hybrid { .. } => "hybrid",
    }
}

/// The CUDA ordinal of a device name, or `-1` for the host.
pub fn device_index(name: &str) -> i64 {
    Device::parse(name, "device")
        .ok()
        .and_then(Device::cuda_index)
        .map_or(-1, i64::from)
}

/// An integer list argument.
pub fn shape_argument(values: &[usize]) -> tvm_ffi::Array<Any> {
    integer_list(values.iter().map(|&value| value as i64))
}
