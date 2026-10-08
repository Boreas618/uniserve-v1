//! Distributed expert weights (DWDP): the weight-prefetch runtime of a rank.
//!
//! Replicas run independently; each keeps its expert shard resident and
//! reads the other ranks' shards from peer memory before the layer that
//! needs them ([`ExpertWeights`]). Setup is collective over the expert
//! union, after loading and before capacity planning:
//!
//! 1. `experts.weight_fields` places each layer's weights for its kernels
//!    and lists this rank's shards;
//! 2. [`ExpertWeights::build`] publishes them and maps every field's full
//!    extent;
//! 3. `experts.bind_weights` installs the full views as the layers'
//!    parameters, and `experts.bind_prefetch` returns the object bindings
//!    pass to their expert calls, over this runtime's hooks.

use std::sync::Arc;

use tvm_ffi::{Any, AnyView};
use uniserve_cuda::{DeviceId, Stream};

use crate::config::{Device, WorkerLaunch};
use crate::ffi::{Fields, Handle, Record};
use crate::model::LoadedModel;
use crate::model::interface::Interface;
use crate::parallel::world::World;
use crate::parallel::{ExpertWeights, ShardField};
use crate::worker::Lanes;
use crate::{Error, Result};

/// The rank's distributed expert weights and their binding-side object.
pub(crate) struct WeightRuntime {
    weights: Arc<ExpertWeights>,
    /// `experts.bind_prefetch`'s object, passed to every binding.
    prefetch: Handle,
    closed: bool,
}

/// A worker error raised back into the calling numerical code.
fn ffi_error(error: &Error) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

fn count(value: &AnyView<'_>) -> std::result::Result<usize, tvm_ffi::Error> {
    let value = i64::try_from(*value)?;
    usize::try_from(value)
        .map_err(|_| tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, "expected a count", ""))
}

impl WeightRuntime {
    /// Set up the rank's distributed expert weights, collectively over the
    /// expert union. `None` unless the union prefetches weights.
    pub(crate) fn start(
        interface: &Interface,
        launch: &WorkerLaunch,
        world: &World,
        model: &LoadedModel,
        lanes: &Lanes,
        node: &str,
    ) -> Result<Option<Self>> {
        let Some(union) = &launch.expert_parallel else {
            return Ok(None);
        };
        if union.exchange != "dwdp" {
            return Ok(None);
        }
        let Device::Cuda(device) = launch.device() else {
            return Err(Error::Unsupported("DWDP requires CUDA peer memory".into()));
        };
        let stream = match lanes.lanes() {
            [] => lanes.preparation(DeviceId(device)),
            [lane] => Some(&lane.stream),
            _ => {
                return Err(Error::Unsupported(
                    "DWDP runs every forward on exactly one execution lane".into(),
                ));
            }
        }
        .ok_or(Error::State("DWDP requires a CUDA stream"))?;
        // SAFETY: the lanes own the stream and outlive this setup.
        let stream = unsafe { Stream::borrowed(DeviceId(device), stream.handle()?) }?;
        let union_group = world
            .union
            .as_ref()
            .ok_or(Error::State("DWDP requires its union's host group"))?;

        let records: tvm_ffi::Array<Any> = interface
            .function("experts.weight_fields")?
            .call_as(&[model.handle().view()])?;
        // Shard tensors stay referenced by `shards` through the setup copies.
        let mut shards = Vec::new();
        let mut layers = Vec::new();
        for (index, record) in records.iter().enumerate() {
            let record = Record::new(format!("experts.weight_fields[{index}]"), record)?;
            let mut fields = Vec::new();
            for field in record.records("fields")? {
                let shard: tvm_ffi::Tensor = field.get("shard")?;
                if !shard.is_contiguous() {
                    return Err(Error::Interface(
                        "experts.weight_fields returns contiguous shards".into(),
                    ));
                }
                let bits = usize::from(shard.dtype().bits) * usize::from(shard.dtype().lanes);
                fields.push(ShardField {
                    name: field.string("name")?,
                    address: shard.data_ptr() as u64,
                    bytes: shard.numel() * bits / 8,
                });
                shards.push(shard);
            }
            layers.push(fields);
        }
        let (weights, views) = ExpertWeights::build(union_group, device, node, &layers, &stream)?;
        drop(shards);
        let bind = interface.function("experts.bind_weights")?;
        for (layer, (fields, views)) in layers.iter().zip(views).enumerate() {
            let mut tensors = Fields::new();
            for (field, view) in fields.iter().zip(views) {
                tensors = tensors.with(&field.name, view.to_tensor()?);
            }
            bind.call_values(&[
                model.handle().to_any(),
                Any::from(layer as i64),
                tensors.into_any(),
            ])?;
        }
        tracing::info!(
            target: "uniserve_worker.launch",
            "DWDP expert weights: {} layers, {} resident and {} slot bytes, {} bytes read per invocation",
            layers.len(),
            weights.resident_bytes,
            weights.buffer_bytes,
            weights.transfer_bytes,
        );

        let weights = Arc::new(weights);
        let prefetch = Handle::new(
            interface
                .function("experts.bind_prefetch")?
                .call_values(&[model.handle().to_any(), hooks(&weights)])?,
        );
        Ok(Some(Self {
            weights,
            prefetch,
            closed: false,
        }))
    }

    /// The object bindings pass to their expert calls.
    pub(crate) fn prefetch(&self) -> &Handle {
        &self.prefetch
    }

    /// Drain the copy stream and release the weights after every binding
    /// and graph retired. An aborted close keeps every mapping until exit:
    /// peers may still read this rank's pages.
    #[allow(clippy::mem_forget)]
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        if aborted {
            std::mem::forget(Arc::clone(&self.weights));
            return Ok(());
        }
        self.weights.close()
    }
}

impl Drop for WeightRuntime {
    fn drop(&mut self) {
        let _ = self.close(true);
    }
}

/// The prefetch hooks as TVM-FFI functions closed over `weights`:
/// `begin()`, `end(stream)`, `before(layer, stream) -> [copy]`,
/// `after(layer, stream)` and `copy(index)`. They block on no peer.
fn hooks(weights: &Arc<ExpertWeights>) -> Any {
    type Hook = dyn Fn(&ExpertWeights, &[AnyView<'_>]) -> std::result::Result<Any, tvm_ffi::Error>;
    let function = |call: Box<Hook>| {
        let weights = Arc::clone(weights);
        tvm_ffi::Function::from_packed(move |args: &[AnyView<'_>]| call(&weights, args))
    };
    let unit = |result: Result<()>| {
        result
            .map(|()| Any::default())
            .map_err(|error| ffi_error(&error))
    };
    Fields::new()
        .with(
            "begin",
            function(Box::new(move |weights, _| unit(weights.begin()))),
        )
        .with(
            "end",
            function(Box::new(move |weights, args| {
                let [stream] = args else {
                    return Err(ffi_error(&Error::Invalid("end takes a stream".into())));
                };
                unit(weights.end(count(stream)?))
            })),
        )
        .with(
            "before",
            function(Box::new(|weights, args| {
                let [layer, stream] = args else {
                    return Err(ffi_error(&Error::Invalid(
                        "before takes a layer and a stream".into(),
                    )));
                };
                let recorded = weights
                    .before(count(layer)?, count(stream)?)
                    .map_err(|error| ffi_error(&error))?;
                Ok(Any::from(crate::ffi::integer_list(
                    recorded.into_iter().map(|index| index as i64),
                )))
            })),
        )
        .with(
            "after",
            function(Box::new(move |weights, args| {
                let [layer, stream] = args else {
                    return Err(ffi_error(&Error::Invalid(
                        "after takes a layer and a stream".into(),
                    )));
                };
                unit(weights.after(count(layer)?, count(stream)?))
            })),
        )
        .with(
            "copy",
            function(Box::new(move |weights, args| {
                let [index] = args else {
                    return Err(ffi_error(&Error::Invalid("copy takes an index".into())));
                };
                unit(weights.copy(count(index)?))
            })),
        )
        .into_any()
}
