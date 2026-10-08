//! A TVM-FFI test harness that drives the image runners with the real
//! numerical interface: Python loads this library with
//! `tvm_ffi.load_module`, passes the interface table and a loaded model,
//! and compares the Rust-owned results with direct library computation
//! (`tests/python/integration/runtime/test_image_runners.py`).
//!
//! Every function binds one component on the host (CPU, no stream), runs
//! one runner operation and returns the Rust-owned results as DLPack
//! tensors.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use tvm_ffi::{Any, Array, Map, String as FfiString, Tensor};
use uniserve_worker::config::Device;
use uniserve_worker::ffi::{Backing, DType, Handle, Record, Region};
use uniserve_worker::model::bind::{
    self, Backends, BindingRequest, ComponentBinding, StreamWorkspaces,
};
use uniserve_worker::model::interface::Interface;
use uniserve_worker::storage::memory::{Memory, StorageView};
use uniserve_worker::{
    DecodeImage, DecoderRunner, EncoderConfig, EncoderRunner, Result, Tower, prepare_input,
};

/// Bytes of the host product arena; every product of these cases fits.
const ARENA_BYTES: usize = 1 << 22;
/// Bytes reserved for each product region.
const PRODUCT_BYTES: u64 = 1 << 18;

fn failure(error: impl std::fmt::Display) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

fn binding(
    interface: &Interface,
    loaded: &Handle,
    component: &str,
    root: &str,
) -> Result<ComponentBinding> {
    let prepared = bind::prepare(
        interface,
        loaded,
        BindingRequest {
            component: component.to_owned(),
            path: root.to_owned(),
            device: Device::Cpu,
            stream: 0,
            size: None,
            backends: Backends::default(),
            derive_host_lengths: true,
            experts: None,
            weights: None,
            cache: None,
        },
    )?;
    let mut bound = bind::bind(interface, &mut StreamWorkspaces::default(), vec![prepared])?;
    Ok(bound.remove(0))
}

fn arena() -> Result<StorageView> {
    StorageView::contiguous(
        Memory::host(ARENA_BYTES) as Arc<_>,
        0,
        DType::U8,
        &[ARENA_BYTES as i64],
    )
}

/// The product region `index` of an arena: `[elements]` of `dtype`.
fn region(index: usize, dtype: DType) -> Result<Region> {
    Region::new(
        index as u64 * PRODUCT_BYTES,
        dtype,
        vec![(PRODUCT_BYTES / dtype.bytes() as u64) as i64],
    )
}

/// The arena bytes a region's leading `shape` elements hold.
fn view(arena: &StorageView, region: &Region, shape: &[i64]) -> Result<Tensor> {
    StorageView::contiguous(
        Arc::clone(arena.memory()),
        region.offset() as usize,
        region.dtype(),
        shape,
    )?
    .to_tensor()
}

/// Encode inline images (`tower` is `vit` or `vae`) with the encoder at
/// `path` of the `component` binding rooted at `root`; returns each image's
/// result and the runner's invocation counts `[replays, eager]`.
fn encode(config: Map<FfiString, Any>) -> std::result::Result<Array<Any>, tvm_ffi::Error> {
    (|| -> Result<Array<Any>> {
        let config = Record::new("encode", Any::from(config))?;
        let interface = Interface::from_table(config.get("table")?)?;
        let loaded = Handle::new(config.value("loaded")?);
        let path = config.string("path")?;
        let tower = config.string("tower")?;
        let payloads = config.strings("payloads")?;
        let binding = binding(
            &interface,
            &loaded,
            &config.string("component")?,
            &config.string("root")?,
        )?;
        let products = arena()?;
        let mut runner = EncoderRunner::bind(
            &interface,
            &binding,
            &loaded,
            &path,
            &EncoderConfig {
                max_images: payloads.len(),
                ..EncoderConfig::default()
            },
            products.clone(),
            None,
        )?;
        let tower = if tower.as_str() == "vit" {
            Tower::Vit
        } else {
            Tower::Vae
        };
        let function = interface.function("image.prepare_input")?;
        let images = payloads
            .iter()
            .map(|payload| {
                prepare_input(
                    &function,
                    &loaded,
                    tower,
                    payload.as_str(),
                    payloads.len() as u32,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = match runner.info() {
            uniserve_worker::EncoderInfo::Vision(vision) => vision.output_dtype,
            uniserve_worker::EncoderInfo::Latent(latent) => latent.output_dtype,
            uniserve_worker::EncoderInfo::Text(text) => text.output_dtype,
        };
        let regions = (0..images.len())
            .map(|index| region(index, dtype))
            .collect::<Result<Vec<_>>>()?;
        let encoded = runner.encode(&images.iter().collect::<Vec<_>>(), &regions)?;
        let mut results: Vec<Any> = encoded
            .shapes
            .iter()
            .zip(&regions)
            .map(|(shape, region)| view(&products, region, shape).map(Any::from))
            .collect::<Result<_>>()?;
        results.push(Any::from(uniserve_worker::ffi::integer_list([
            encoded.replays as i64,
            encoded.eager as i64,
        ])));
        runner.close(false)?;
        Ok(results.into_iter().collect())
    })()
    .map_err(failure)
}

/// Decode `latents` (`[units, width]` per image, consecutive) of images of
/// `sizes` (`[height, width]` each) with the image decoder at `path` of the
/// `component` binding rooted at `root`, through a staging of
/// `staging_bytes`. Returns each image's staged pixels and its product.
fn decode(config: Map<FfiString, Any>) -> std::result::Result<Array<Any>, tvm_ffi::Error> {
    (|| -> Result<Array<Any>> {
        let config = Record::new("decode", Any::from(config))?;
        let interface = Interface::from_table(config.get("table")?)?;
        let loaded = Handle::new(config.value("loaded")?);
        let path = config.string("path")?;
        let latents: Tensor = config.get("latents")?;
        let sizes = config.list("sizes")?;
        let staging_bytes = config.count("staging_bytes")?;
        let binding = binding(
            &interface,
            &loaded,
            &config.string("component")?,
            &config.string("root")?,
        )?;
        let products = arena()?;
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
        let runner = DecoderRunner::bind(
            &interface,
            &binding,
            &loaded,
            &path,
            (-1.0, 1.0),
            workspace,
            products.clone(),
            staging_bytes,
            None,
        )?;
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
                product: Some(region(index, DType::BF16)?),
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
            let product = image.product.as_ref().expect("product region");
            results.push(Any::from(view(
                &products,
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
