//! Model description, loading and component bindings.
//!
//! Loading runs in phases around process-group creation:
//!
//! 1. [`Description::describe`] reads the checkpoint configuration and
//!    describes a meta-device skeleton: components, calls and capability
//!    tags. Placement is validated here, before any collective, so a bad
//!    placement cannot leave peers waiting in one.
//! 2. [`Description::resolve`] narrows the checkpoint to the modules this
//!    rank calls and applies the launch's weight choices.
//! 3. [`LoadedModel::load`] materializes those modules on the component
//!    meshes and resolves each declared call on the loaded model.
//! 4. [`LoadedModel::facts`] reports the model-derived quantities capacity
//!    planning and reporting need, so Rust never inspects modules.
//!
//! Component bindings ([`bind`]) and the registry of placed components by
//! capability ([`registry`]) follow.

pub mod bind;
pub mod canvas;
pub mod components;
pub mod image;
pub mod interface;
pub mod registry;
pub mod text;

use std::collections::BTreeMap;

use tvm_ffi::Any;
use uniserve_ipc::{DType, DimBound, OutputInfo, RasterAxes, ShapeBound};

use crate::config::{Device, WorkerLaunch};
use crate::ffi::{Fields, Handle, Record, integer_list, list, strings};
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
    /// Describe the model and validate the placement against it.
    pub fn describe(interface: &Interface, launch: &WorkerLaunch) -> Result<Self> {
        let execution = &launch.execution;
        let model = launch.model.as_ref();
        let components = serde_json::to_value(&launch.components).map_err(|error| {
            Error::Invalid(format!("encoding the component placement: {error}"))
        })?;
        let request = Fields::new()
            .with("stub", launch.stub)
            .optional_text("path", model.map(|model| model.path.as_str()))
            .optional_text(
                "base_model",
                model.and_then(|model| model.base_model.as_deref()),
            )
            .with(
                "io",
                Fields::new()
                    .text("format", &launch.load.format)
                    .text("mode", &launch.load.mode)
                    .optional_text("download_dir", launch.load.download_dir.as_deref())
                    .optional("num_threads", launch.load.num_threads.map(i64::from))
                    .optional_text(
                        "checksum_manifest",
                        launch.load.checksum_manifest.as_deref(),
                    )
                    .map(),
            )
            .text("device", &execution.device)
            .optional_text("generation_device", execution.generation_device.as_deref())
            .with("components", crate::ffi::json(&components)?)
            .with("rank", execution.rank as i64)
            .text("role", &execution.role)
            .with(
                "expert_split",
                launch
                    .expert_parallel
                    .as_ref()
                    .is_some_and(|union| union.disaggregated()),
            )
            .into_any();
        let result: tvm_ffi::Array<Any> = interface.describe.call_as(&[(&request).into()])?;
        if result.len() != 2 {
            return Err(Error::Interface(
                "model.describe returns [description, info]".into(),
            ));
        }
        let handle = Handle::new(
            result
                .get(0)
                .map_err(|error| Error::numerical("model.describe", &error))?,
        );
        let info = Record::new(
            "model.describe",
            result
                .get(1)
                .map_err(|error| Error::numerical("model.describe", &error))?,
        )?;
        let declarations = Declarations::from_record(&info)?;
        declarations.validate(&launch.components)?;
        Ok(Self {
            handle,
            model_name: info.string("model_name")?,
            declarations,
            expert_paths: info.strings("expert_paths")?,
            video_decoder: info.flag("video_decoder")?,
        })
    }

    /// Narrow the checkpoint to this rank's resident modules. Dedicated
    /// expert ranks hold only the routed experts; attention ranks of a
    /// disaggregated union exclude them.
    pub fn resolve(&self, interface: &Interface, launch: &WorkerLaunch) -> Result<Handle> {
        let dedicated = launch.execution.role == "experts";
        let split = launch
            .expert_parallel
            .as_ref()
            .is_some_and(|union| union.disaggregated());
        let modules: Vec<String> = if dedicated {
            self.expert_paths.clone()
        } else {
            self.declarations
                .resident(launch.execution.rank, &launch.components)
                .into_iter()
                .collect()
        };
        let excluded: &[String] = if split && !dedicated {
            &self.expert_paths
        } else {
            &[]
        };
        let request = Fields::new()
            .with("modules", strings(&modules))
            .with("exclude_modules", strings(excluded))
            .text(
                "quantization",
                launch
                    .model
                    .as_ref()
                    .map_or("{}", |model| model.quantization.as_str()),
            )
            .text("model_dtype", &launch.execution.model_dtype)
            .into_any();
        Ok(Handle::new(
            interface
                .resolve
                .call(&[self.handle.view(), (&request).into()])?,
        ))
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
}

impl LoadedModel {
    /// Materialize this rank's modules.
    pub fn load(
        interface: &Interface,
        description: Description,
        resolved: Handle,
        launch: &WorkerLaunch,
        meshes: Vec<ComponentMesh>,
        experts: Option<ExpertPartition>,
        host_only: bool,
    ) -> Result<Self> {
        let attention = &launch.attention;
        let meshes = list(meshes.into_iter().map(|mesh| {
            Fields::new()
                .text("component", &mesh.component)
                .with("paths", strings(&mesh.paths))
                .with("mesh", mesh.mesh.to_any())
                .optional("units", mesh.units.map(|units| units.to_any()))
                .text("sequence", mesh.sequence)
                .into_any()
        }));
        let experts = experts.map(|experts| {
            Fields::new()
                .with("world", experts.world.to_any())
                .with("attention_ranks", experts.attention_ranks as i64)
                .with("dedicated", experts.dedicated)
                .into_any()
        });
        let request = Fields::new()
            .text("device", &launch.execution.device)
            .optional_text(
                "generation_device",
                launch.execution.generation_device.as_deref(),
            )
            .text("role", &launch.execution.role)
            .with("host_only", host_only)
            .with("meshes", meshes)
            .optional("experts", experts)
            .text(
                "attention_backend",
                launch
                    .execution
                    .attention_backend
                    .as_deref()
                    .unwrap_or("auto"),
            )
            .with(
                "attention",
                Fields::new()
                    .with("workspace_size", attention.workspace_size as i64)
                    .text("decode_backend", &attention.decode_backend)
                    .text("prefill_backend", &attention.prefill_backend)
                    .with("disable_split_kv", attention.disable_split_kv)
                    .optional(
                        "decode_split_tile_size",
                        attention.decode_split_tile_size.map(i64::from),
                    )
                    .optional(
                        "prefill_split_tile_size",
                        attention.prefill_split_tile_size.map(i64::from),
                    )
                    .optional("use_tensor_core", attention.use_tensor_core)
                    .map(),
            )
            .into_any();
        let handle = Handle::new(interface.load.call(&[resolved.view(), (&request).into()])?);
        Ok(Self {
            handle,
            model_name: description.model_name,
            declarations: description.declarations,
            video_decoder: description.video_decoder,
        })
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// Model-derived quantities for capacity and reporting.
    pub fn facts(&self, interface: &Interface, launch: &WorkerLaunch) -> Result<ModelFacts> {
        let execution = &launch.execution;
        let bounds = Fields::new()
            .with("max_sequence_tokens", execution.max_sequence_tokens as i64)
            .text("model_dtype", &execution.model_dtype)
            .optional_text("kv_cache_dtype", execution.kv_cache_dtype.as_deref())
            .text("role", &execution.role)
            .with(
                "deployment_components",
                strings(&execution.deployment_components),
            )
            .with("max_condition_rows", execution.max_condition_rows as i64)
            .into_any();
        let record = Record::new(
            "model.facts",
            interface
                .facts
                .call(&[self.handle.view(), (&bounds).into()])?,
        )?;
        ModelFacts::from_record(&record)
    }

    /// The KV layout for `block_size`-token pages of the widest rows, as
    /// `model.cache_plan` returns it. `kv_dtype` is `None` for the compute
    /// dtype.
    pub fn cache_plan(
        &self,
        interface: &Interface,
        block_size: usize,
        kv_dtype: Option<&str>,
    ) -> Result<Any> {
        let value = interface.function("model.cache_plan")?.call_values(&[
            self.handle.to_any(),
            Any::from(block_size as i64),
            Any::from(tvm_ffi::String::from(kv_dtype.unwrap_or(""))),
        ])?;
        if crate::ffi::is_none(&value) {
            return Err(Error::Interface(
                "model.cache_plan describes no cache for a causal LM".into(),
            ));
        }
        Ok(value)
    }

    /// Whether the attention `backend` reads every resident layer's pages
    /// of a `cache_plan` layout.
    pub fn reads_pages(&self, interface: &Interface, backend: &str, plan: &Any) -> Result<bool> {
        interface.function("model.reads_pages")?.call_as(&[
            self.handle.view(),
            (&Any::from(tvm_ffi::String::from(backend))).into(),
            plan.into(),
        ])
    }
}

/// The image flow of a model with an image denoiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageFlow {
    /// Longest image sequence the network accepts, in patch rows.
    pub max_tokens: usize,
    pub latent_channels: usize,
    pub patch_size: usize,
}

/// Model-derived quantities reported by `model.facts`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFacts {
    pub model_name: String,
    /// The attention backend the model was loaded with.
    pub attention_backend: String,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    /// Products each component publishes, by component.
    pub outputs: BTreeMap<String, Vec<OutputInfo>>,
    pub weight_formats: Vec<String>,
    pub activation_formats: Vec<String>,
    pub image_processor: bool,
    pub vision_feature_bytes: u64,
    pub image_flow: Option<ImageFlow>,
}

impl ModelFacts {
    fn from_record(record: &Record) -> Result<Self> {
        let outputs_record = record.record("outputs")?;
        let mut outputs = BTreeMap::new();
        for component in outputs_record.keys() {
            let entries = outputs_record
                .records(&component)?
                .iter()
                .map(|output| output_info(&component, output))
                .collect::<Result<Vec<_>>>()?;
            outputs.insert(component, entries);
        }
        let image_flow = record
            .optional_record("image_flow")?
            .map(|flow| -> Result<ImageFlow> {
                Ok(ImageFlow {
                    max_tokens: flow.count("max_tokens")?,
                    latent_channels: flow.count("latent_channels")?,
                    patch_size: flow.count("patch_size")?,
                })
            })
            .transpose()?;
        Ok(Self {
            model_name: record.string("model_name")?,
            attention_backend: record.string("attention_backend")?,
            vocab_size: record.optional_count("vocab_size")?,
            hidden_size: record.optional_count("hidden_size")?,
            outputs,
            weight_formats: record.strings("weight_formats")?,
            activation_formats: record.strings("activation_formats")?,
            image_processor: record.flag("image_processor")?,
            vision_feature_bytes: record.count("vision_feature_bytes")? as u64,
            image_flow,
        })
    }
}

/// Translate a numerical output layout into a bounded protocol product.
fn output_info(component: &str, output: &Record) -> Result<OutputInfo> {
    let name = output.string("name")?;
    let dtype = match output.string("dtype")?.as_str() {
        "uint8" => DType::U8,
        "int16" => DType::I16,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        "float32" => DType::F32,
        _ => {
            return Err(Error::Unsupported(format!(
                "result {component}.{name} has no protocol dtype"
            )));
        }
    };
    let shape = output.shape("shape")?;
    let variable = output.shape("variable_axes")?;
    if variable.len() > 1 {
        return Err(Error::Unsupported(format!(
            "result {component}.{name} exceeds the protocol dynamic axes"
        )));
    }
    let extent = |value: i64| {
        u32::try_from(value).map_err(|_| {
            Error::Unsupported(format!("result {component}.{name} has an invalid extent"))
        })
    };
    let dims = shape
        .iter()
        .enumerate()
        .map(|(axis, &size)| {
            Ok(if variable.contains(&(axis as i64)) {
                DimBound::Device { max: extent(size)? }
            } else {
                DimBound::Static(extent(size)?)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let raster_axes = match output.optional_value("raster_axes")? {
        Some(value) => {
            let axes = crate::ffi::integers(value)
                .map_err(|error| Error::numerical("model.facts", &error))?;
            let [height, width] = axes[..] else {
                return Err(Error::Interface(format!(
                    "result {component}.{name} raster axes name two axes"
                )));
            };
            Some(RasterAxes {
                height: extent(height)?,
                width: extent(width)?,
            })
        }
        None => None,
    };
    Ok(OutputInfo {
        name,
        dtype,
        shape_bound: ShapeBound { dims },
        raster_axes,
    })
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
