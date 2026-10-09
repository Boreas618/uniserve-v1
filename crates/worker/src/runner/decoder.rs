//! Image decoding: completed latent trajectories to delivered pixels.
//!
//! A [`DecoderRunner`] serves one bound `ImageDecoder`. It decodes images
//! whose latent rows the caller gathered into the latent pool's step
//! workspace, optionally publishes the decoded values as image products,
//! and quantizes every image to packed 8-bit HWC pixels in its staging
//! bytes (`uniserve_models.interface.decoder`). The caller copies each
//! image's staged pixels to host memory on the runner's stream and encodes
//! them there ([`crate::host::image::png_base64`]).
//!
//! Staging holds the pixels of as many of the largest images as fit its
//! bytes; a batch with more images decodes in consecutive sub-batches.
//! Images of one size decode together within a sub-batch.

use tvm_ffi::Any;
use uniserve_cuda::Stream;

use crate::config::Device;
use crate::ffi::{DType, Fields, Function, Functions, Handle, Region, integer_list, list};
use crate::model::bind::ComponentBinding;
use crate::model::interface::Interface;
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// One image to decode.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeImage {
    /// The image's first latent row in the step workspace and its rows.
    pub sample: u32,
    pub units: u32,
    pub height: u32,
    pub width: u32,
    /// The image product receiving the decoded values, if published.
    pub product: Option<Region>,
}

impl DecodeImage {
    /// Bytes of the image's quantized pixels.
    pub fn pixel_bytes(&self) -> usize {
        self.height as usize * self.width as usize * 3
    }
}

/// The runner of one bound image decoder.
pub struct DecoderRunner {
    owner: String,
    device: Device,
    stream: Option<Stream>,
    decode: Function,
    quantize: Function,
    /// The decoder's pixel range, `(low, high)`.
    range: (f32, f32),
    staging: StorageView,
    /// Buffers bound to the binding, retained for its lifetime.
    _buffers: Vec<StorageView>,
}

impl DecoderRunner {
    /// Bind the image decoder at `path` of `binding` over the latent pool's
    /// step workspace `latents`, the device's product arena and
    /// `staging_bytes` of staging. `stream` borrows the binding's stream.
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding's complete resources"
    )]
    pub fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        loaded: &Handle,
        path: &str,
        range: (f32, f32),
        latents: StorageView,
        products: StorageView,
        staging_bytes: usize,
        stream: Option<Stream>,
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            binding.component.clone()
        } else {
            format!("{}.{path}", binding.component)
        };
        let device = binding.device;
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an image decoder runs on a stream exactly on a CUDA device".into(),
            ));
        }
        let memory: std::sync::Arc<dyn crate::ffi::Backing> = match device {
            Device::Cuda(index) => Memory::device(index, staging_bytes.max(1))?,
            Device::Cpu => Memory::host(staging_bytes.max(1)),
        };
        let staging = StorageView::contiguous(memory, 0, DType::U8, &[staging_bytes as i64])?;
        let buffers = vec![
            ("latents", latents),
            ("staging", staging.clone()),
            ("products", products),
        ];
        let mut tensors = Fields::new();
        for (name, view) in &buffers {
            tensors = tensors.with(name, view.to_tensor()?);
        }
        let function = interface.function("decoder.bind")?;
        let functions = Functions::new(
            function.name(),
            function.call_values(&[
                binding.prepared().to_any(),
                loaded.to_any(),
                Any::from(tvm_ffi::String::from(path)),
                tensors.into_any(),
            ])?,
        )?;
        Ok(Self {
            owner,
            device,
            stream,
            decode: functions.get("decode")?,
            quantize: functions.get("quantize")?,
            range,
            staging,
            _buffers: buffers.into_iter().map(|(_, view)| view).collect(),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The runner's stream; `None` on the host.
    pub fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// The decoder's pixel range, `(low, high)`.
    pub fn range(&self) -> (f32, f32) {
        self.range
    }

    /// Decode `images`, handing each image's staged pixels (in input
    /// order) to `stage`, which must enqueue their consumption on the
    /// runner's stream before it returns: the next sub-batch reuses the
    /// staging bytes.
    pub fn decode(
        &self,
        images: &[DecodeImage],
        mut stage: impl FnMut(usize, StorageView) -> Result<()>,
    ) -> Result<()> {
        let capacity = self.staging.numel() as usize;
        let mut first = 0;
        while first < images.len() {
            // The longest prefix of the remaining images that fits staging.
            let mut bytes = 0;
            let mut end = first;
            while end < images.len() && bytes + images[end].pixel_bytes() <= capacity {
                bytes += images[end].pixel_bytes();
                end += 1;
            }
            if end == first {
                return Err(Error::Invalid(format!(
                    "{}: a {}x{} image exceeds the decoder staging of {capacity} bytes",
                    self.owner, images[first].height, images[first].width
                )));
            }
            let mut offsets = Vec::with_capacity(end - first);
            let mut offset = 0;
            let rows = list(images[first..end].iter().map(|image| {
                offsets.push(offset);
                let row = integer_list([
                    i64::from(image.sample),
                    i64::from(image.units),
                    i64::from(image.height),
                    i64::from(image.width),
                    offset as i64,
                ]);
                offset += image.pixel_bytes();
                Any::from(row)
            }));
            let products = list(images[first..end].iter().map(|image| {
                image
                    .product
                    .as_ref()
                    .map_or_else(Any::default, Region::to_any)
            }));
            let counts = crate::ffi::integers(
                self.decode
                    .call_values(&[Any::from(rows), Any::from(products)])?,
            )
            .map_err(|error| Error::numerical(self.decode.name(), &error))?;
            if counts.len() != end - first {
                return Err(Error::Interface(format!(
                    "{} returns one byte count per image",
                    self.decode.name()
                )));
            }
            for ((index, image), (&count, &offset)) in (first..end)
                .zip(&images[first..end])
                .zip(counts.iter().zip(&offsets))
            {
                if count as usize != image.pixel_bytes() {
                    return Err(Error::Interface(format!(
                        "{} staged {count} bytes of a {}x{} image",
                        self.decode.name(),
                        image.height,
                        image.width
                    )));
                }
                stage(index, self.staging.narrow(0, offset as i64, count)?)?;
            }
            first = end;
        }
        Ok(())
    }

    /// Quantize a resident image product holding values in `range` and
    /// hand its staged pixels to `stage` (see [`Self::decode`]).
    pub fn quantize(
        &self,
        image: &Region,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
    ) -> Result<()> {
        let count: i64 = self.quantize.call_as(&[
            (&image.to_any()).into(),
            (&Any::from(f64::from(range.0))).into(),
            (&Any::from(f64::from(range.1))).into(),
            (&Any::from(0_i64)).into(),
        ])?;
        if count <= 0 || count as u64 > self.staging.numel() {
            return Err(Error::Interface(format!(
                "{} staged {count} bytes",
                self.quantize.name()
            )));
        }
        stage(self.staging.narrow(0, 0, count)?)
    }
}
