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

#[path = "harness/config.rs"]
pub mod config;

use std::sync::Arc;

use config::Config;
use cuda::{DeviceId, Stream};
use tvm_ffi::{Any, Array, Map, String as FfiString, Tensor};
use worker::config::Device;
use worker::ffi::{Backing, DType, Handle, Interpreter};
use worker::memory::{Storage, StorageView};
use worker::{
    DecodeImage, DecoderRunner, EncoderConfig, EncoderInfo, EncoderRunner, GraphPool,
    ImagePreparation, ImageSetup, Result, Tower,
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
        Device::Cuda(cuda) => Storage::device(cuda, ARENA_BYTES)?,
        Device::Cpu => Storage::host(ARENA_BYTES),
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
        let config = Config::new("encode", Any::from(config))?;
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
                    worker::ffi::integers(order)
                        .map(|order| order.into_iter().map(|index| index as usize).collect())
                        .map_err(|error| worker::Error::numerical("orders", &error))
                })
                .collect::<Result<_>>()?,
            None => vec![(0..payloads.len()).collect()],
        };
        let device = match config.optional::<i64>("device")? {
            Some(index) if index >= 0 => Device::Cuda(DeviceId(index as i32)),
            _ => Device::Cpu,
        };
        let stream = match device {
            Device::Cuda(cuda) => Some(Stream::new(cuda)?),
            Device::Cpu => None,
        };
        let root = config.string("root")?;
        let module = module_path(&root, &path);
        let info = EncoderInfo::describe(&loaded, &module)?;
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
        // The captured graphs address the pool's memory, which it releases
        // when dropped: declared before the runner, it outlives it.
        let mut capture_pool = None;
        let mut runner = EncoderRunner::image(&setup, info, &settings)?;
        if config.optional::<bool>("capture")?.unwrap_or(false) {
            let Device::Cuda(cuda) = device else {
                return Err(worker::Error::Invalid(
                    "capture requires a CUDA device".into(),
                ));
            };
            let pool = capture_pool.insert(pool(cuda)?);
            runner.capture(Some(pool), &mut || Ok(()))?;
        }
        runner.seal();
        let tower = if tower.as_str() == "vit" {
            Tower::Vit
        } else {
            Tower::Vae
        };
        let preparation = ImagePreparation::new(loaded.clone())?;
        let dtype = runner
            .pixel_dtype()
            .ok_or_else(|| worker::Error::Invalid("the harness encodes images".into()))?;
        let images = payloads
            .iter()
            .map(|payload| {
                preparation.prepare(
                    tower,
                    payload.as_str(),
                    payloads.len() as u32,
                    runner.pinned(),
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = match info {
            EncoderInfo::Vision(vision) => vision.output_dtype,
            EncoderInfo::Latent(latent) => latent.output_dtype,
        };
        let mut results = Vec::new();
        let mut next = 0;
        for order in &orders {
            let outputs = (next..next + order.len())
                .map(|index| product(&products, index, dtype))
                .collect::<Result<Vec<_>>>()?;
            next += order.len();
            let batch: Vec<_> = order.iter().map(|&index| &images[index]).collect();
            // A sampled posterior draws each image's noise at seed 0 and the
            // image's index.
            let draws: Vec<_> = order.iter().map(|&index| (0, index as u64)).collect();
            let encoded = runner.encode(&batch, &draws, &outputs)?;
            for (shape, output) in encoded.shapes.iter().zip(&outputs) {
                results.push(Any::from(view(output, shape)?));
            }
            results.push(Any::from(worker::ffi::integer_list([
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
fn pool(device: DeviceId) -> Result<Arc<GraphPool>> {
    Ok(Arc::new(GraphPool::new(device)?))
}

/// Decode `latents` (`[units, width]` per image, consecutive) of images of
/// `sizes` (`[height, width]` each) with the image decoder at `path` of the
/// `component` below `root` of the model, through a staging of
/// `staging_bytes`. Returns each image's staged pixels and its product.
fn decode(config: Map<FfiString, Any>) -> std::result::Result<Array<Any>, tvm_ffi::Error> {
    (|| -> Result<Array<Any>> {
        let config = Config::new("decode", Any::from(config))?;
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
            return Err(worker::Error::Invalid("latents are [units, width]".into()));
        };
        let bytes = (units * width) as usize * 2;
        let memory = Storage::host(bytes);
        // SAFETY: both ranges hold `bytes` bytes of BF16 latents.
        unsafe {
            std::ptr::copy_nonoverlapping(
                latents.data_ptr().cast::<u8>(),
                memory.address() as *mut u8,
                bytes,
            );
        }
        let workspace = StorageView::contiguous(memory as Arc<_>, 0, DType::BF16, &[units, width])?;
        // The harness decodes eagerly, at any size.
        let runner = DecoderRunner::image(&setup, (-1.0, 1.0), workspace, staging_bytes, 0, 0)?;
        let mut images = Vec::new();
        let mut sample = 0;
        for (index, size) in sizes.iter().enumerate() {
            let size = worker::ffi::integers(size.clone())
                .map_err(|error| worker::Error::numerical("sizes", &error))?;
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
        // Each image's staged bytes, copied out before the next sub-batch
        // reuses them.
        let staged = runner.decode(&images)?;
        let mut results = Vec::new();
        for (image, pixels) in images.iter().zip(staged) {
            results.push(Any::from(pixels.to_tensor()?));
            let product = image.product.as_ref().ok_or_else(|| {
                worker::Error::Invariant("a decoded image lost its product".into())
            })?;
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
