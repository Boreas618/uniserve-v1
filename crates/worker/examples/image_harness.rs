//! A TVM-FFI test harness that drives the image runners with the real
//! numerical library: Python loads this library with
//! `tvm_ffi.load_module`, passes the rank's import function
//! (`uniserve.worker.import_path`) and a loaded model,
//! and compares the Rust-owned results with direct library computation
//! (`tests/python/integration/runtime/test_image_runners.py`).
//!
//! Every function binds one component on the host (CPU, no stream), runs
//! one runner operation and returns the Rust-owned results as DLPack
//! tensors.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use tvm_ffi::{Any, Array, Map, String as FfiString, Tensor};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_worker::config::Device;
use uniserve_worker::ffi::{Backing, DType, Handle, Interpreter, Record};
use uniserve_worker::storage::memory::{Memory, StorageView};
use uniserve_worker::{
    DecodeImage, DecoderRunner, EncoderConfig, EncoderInfo, EncoderRunner, GraphPool, ImageSetup,
    Result, Tower, prepare_input,
};

/// Bytes of the host product arena; every product of these cases fits.
const ARENA_BYTES: usize = 1 << 23;
/// Bytes reserved for each product.
const PRODUCT_BYTES: u64 = 1 << 16;

fn failure(error: impl std::fmt::Display) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

/// The shared work areas of one stream, as the worker holds them.
fn scratch() -> Result<Handle> {
    Handle::new(Interpreter::import("uniserve.bindings.context.Scratch")?.call_values(&[])?)
}

/// The module path `path` below `root`.
fn module_path(root: &str, path: &str) -> String {
    [root, path]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}

fn arena(device: Device) -> Result<StorageView> {
    let memory: Arc<dyn Backing> = match device {
        Device::Cuda(index) => Memory::device(index, ARENA_BYTES)?,
        Device::Cpu => Memory::host(ARENA_BYTES),
    };
    StorageView::contiguous(memory, 0, DType::U8, &[ARENA_BYTES as i64])
}

/// Product `index` of an arena: `[elements]` of `dtype`.
fn product(arena: &StorageView, index: usize, dtype: DType) -> Result<StorageView> {
    StorageView::contiguous(
        Arc::clone(arena.memory()),
        index * PRODUCT_BYTES as usize,
        dtype,
        &[(PRODUCT_BYTES / dtype.bytes() as u64) as i64],
    )
}

/// The leading `shape` elements of a product.
fn view(product: &StorageView, shape: &[i64]) -> Result<Tensor> {
    StorageView::contiguous(
        Arc::clone(product.memory()),
        (product.address() - product.memory().address()) as usize,
        product.dtype(),
        shape,
    )?
    .to_tensor()
}

/// Encode inline images (`tower` is `vit` or `vae`) with the encoder at
/// `path` below `root` of the model, named by `component`, on `device` (a CUDA
/// ordinal, the host when absent), capturing the encoder's graphs first
/// when `capture` is set. Each entry of `orders` (default: every image in
/// order) is one batch of image indices. Returns, per batch, each image's
/// result and the batch's invocation counts `[replays, eager]`.
fn encode(config: Map<FfiString, Any>) -> std::result::Result<Array<Any>, tvm_ffi::Error> {
    (|| -> Result<Array<Any>> {
        let config = Record::new("encode", Any::from(config))?;
        Interpreter::install_import(config.get("import")?);
        let loaded = Handle::new(config.value("loaded")?)?;
        let path = config.string("path")?;
        let tower = config.string("tower")?;
        let payloads = config.strings("payloads")?;
        let orders: Vec<Vec<usize>> = match config.optional_value("orders")? {
            Some(_) => config
                .list("orders")?
                .into_iter()
                .map(|order| {
                    uniserve_worker::ffi::integers(order)
                        .map(|order| order.into_iter().map(|index| index as usize).collect())
                        .map_err(|error| uniserve_worker::Error::numerical("orders", &error))
                })
                .collect::<Result<_>>()?,
            None => vec![(0..payloads.len()).collect()],
        };
        let device = match config.optional::<i64>("device")? {
            Some(index) if index >= 0 => Device::Cuda(index as i32),
            _ => Device::Cpu,
        };
        let stream = match device {
            Device::Cuda(index) => Some(Stream::new(DeviceId(index))?),
            Device::Cpu => None,
        };
        let root = config.string("root")?;
        let module = module_path(&root, &path);
        let info = EncoderInfo::describe(&loaded, &module, DType::BF16)?;
        let settings = EncoderConfig {
            max_images: orders.iter().map(Vec::len).max().unwrap_or(1),
            ..EncoderConfig::default()
        };
        let component = config.string("component")?;
        let setup = ImageSetup {
            loaded: &loaded,
            component: &component,
            path: &module,
            device,
            stream: stream.clone(),
            scratch: scratch()?,
        };
        let products = arena(device)?;
        let mut runner = EncoderRunner::image(&setup, info, &settings)?;
        if config.optional::<bool>("capture")?.unwrap_or(false) {
            let Device::Cuda(index) = device else {
                return Err(uniserve_worker::Error::Invalid(
                    "capture requires a CUDA device".into(),
                ));
            };
            let pool = pool(index)?;
            runner.capture(Some(&pool), &mut || Ok(()))?;
        }
        runner.seal();
        let tower = if tower.as_str() == "vit" {
            Tower::Vit
        } else {
            Tower::Vae
        };
        let function = Interpreter::import("uniserve.runner.encoder.EncoderRunner.prepare_image")?;
        let facts = Interpreter::import("uniserve.runner.encoder.EncoderRunner.image_facts")?;
        let images = payloads
            .iter()
            .map(|payload| {
                prepare_input(
                    &function,
                    &facts,
                    &loaded,
                    tower,
                    payload.as_str(),
                    payloads.len() as u32,
                    stream.is_some(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = match runner.info() {
            EncoderInfo::Vision(vision) => vision.output_dtype,
            EncoderInfo::Latent(latent) => latent.output_dtype,
            EncoderInfo::Text(text) => text.output_dtype,
            EncoderInfo::Condition => {
                return Err(uniserve_worker::Error::Invalid(
                    "the harness encodes images".into(),
                ));
            }
        };
        let mut results = Vec::new();
        let mut next = 0;
        for order in &orders {
            let outputs = (next..next + order.len())
                .map(|index| product(&products, index, dtype))
                .collect::<Result<Vec<_>>>()?;
            next += order.len();
            let batch: Vec<_> = order.iter().map(|&index| &images[index]).collect();
            let encoded = runner.encode(&batch, &outputs)?;
            for (shape, output) in encoded.shapes.iter().zip(&outputs) {
                results.push(Any::from(view(output, shape)?));
            }
            results.push(Any::from(uniserve_worker::ffi::integer_list([
                encoded.replays as i64,
                encoded.eager as i64,
            ])));
        }
        if let Some(stream) = &stream {
            stream.synchronize()?;
        }
        runner.close(false)?;
        Ok(results.into_iter().collect())
    })()
    .map_err(failure)
}

/// A graph allocator pool on CUDA `device` (`uniserve.cuda_graph.pool`).
fn pool(device: i32) -> Result<Arc<GraphPool<Handle>>> {
    let value: Array<Any> = Interpreter::import("uniserve.cuda_graph.pool")?
        .call_as(&[(&Any::from(i64::from(device))).into()])?;
    let number = |index: usize| -> Result<u64> {
        let value = value.get(index).map_err(|error| {
            uniserve_worker::Error::numerical("uniserve.cuda_graph.pool", &error)
        })?;
        i64::try_from(value)
            .map(|value| value as u64)
            .map_err(|error| uniserve_worker::Error::numerical("uniserve.cuda_graph.pool", &error))
    };
    Ok(Arc::new(GraphPool {
        device,
        id: (number(1)?, number(2)?),
        value: Handle::new(value.get(0).map_err(|error| {
            uniserve_worker::Error::numerical("uniserve.cuda_graph.pool", &error)
        })?)?,
    }))
}

/// Decode `latents` (`[units, width]` per image, consecutive) of images of
/// `sizes` (`[height, width]` each) with the image decoder at `path` of the
/// `component` below `root` of the model, through a staging of
/// `staging_bytes`. Returns each image's staged pixels and its product.
fn decode(config: Map<FfiString, Any>) -> std::result::Result<Array<Any>, tvm_ffi::Error> {
    (|| -> Result<Array<Any>> {
        let config = Record::new("decode", Any::from(config))?;
        Interpreter::install_import(config.get("import")?);
        let loaded = Handle::new(config.value("loaded")?)?;
        let path = config.string("path")?;
        let latents: Tensor = config.get("latents")?;
        let sizes = config.list("sizes")?;
        let staging_bytes = config.count("staging_bytes")?;
        let component = config.string("component")?;
        let module = module_path(&config.string("root")?, &path);
        let setup = ImageSetup {
            loaded: &loaded,
            component: &component,
            path: &module,
            device: Device::Cpu,
            stream: None,
            scratch: scratch()?,
        };
        let products = arena(Device::Cpu)?;
        let [units, width] = latents.shape()[..] else {
            return Err(uniserve_worker::Error::Invalid(
                "latents are [units, width]".into(),
            ));
        };
        let bytes = (units * width) as usize * 2;
        let memory = Memory::host(bytes);
        // SAFETY: both ranges hold `bytes` bytes of BF16 latents.
        unsafe {
            std::ptr::copy_nonoverlapping(
                latents.data_ptr().cast::<u8>(),
                memory.address() as *mut u8,
                bytes,
            );
        }
        let workspace = StorageView::contiguous(memory as Arc<_>, 0, DType::BF16, &[units, width])?;
        let runner = DecoderRunner::image(&setup, (-1.0, 1.0), workspace, staging_bytes)?;
        let mut images = Vec::new();
        let mut sample = 0;
        for (index, size) in sizes.iter().enumerate() {
            let size = uniserve_worker::ffi::integers(size.clone())
                .map_err(|error| uniserve_worker::Error::numerical("sizes", &error))?;
            let (height, width) = (size[0] as u32, size[1] as u32);
            let rows = (height / 16) * (width / 16);
            images.push(DecodeImage {
                sample,
                units: rows,
                height,
                width,
                product: Some(product(&products, index, DType::BF16)?),
            });
            sample += rows;
        }
        let mut staged = vec![None; images.len()];
        runner.decode(&images, |index, pixels| {
            // Copy the staged bytes out before the next sub-batch reuses
            // them, as a device-to-host copy on the runner's stream would.
            let memory = Memory::host(pixels.nbytes() as usize);
            // SAFETY: host staging and a fresh host allocation of its size.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels.address() as *const u8,
                    memory.address() as *mut u8,
                    pixels.nbytes() as usize,
                );
            }
            staged[index] = Some(StorageView::contiguous(
                memory as Arc<_>,
                0,
                DType::U8,
                pixels.shape(),
            )?);
            Ok(())
        })?;
        let mut results = Vec::new();
        for (image, pixels) in images.iter().zip(staged) {
            results.push(Any::from(pixels.expect("staged image").to_tensor()?));
            let product = image.product.as_ref().expect("product");
            results.push(Any::from(view(
                product,
                &[3, i64::from(image.height), i64::from(image.width)],
            )?));
        }
        Ok(results.into_iter().collect())
    })()
    .map_err(failure)
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(encode, encode);
tvm_ffi::tvm_ffi_dll_export_typed_func!(decode, decode);
