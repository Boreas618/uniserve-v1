//! Installing one module of a program layer by operator name.
//!
//! A program layer arrives as modules keyed by operator name and descriptor key (specs/worker.md
//! 5.3), and the executor's precopy has to turn each one into a kernel of a generation without
//! knowing any operator's type. This is that entry: it takes the operator's name, the descriptor
//! key the module was compiled for, and the module's image or path, and hands back the loaded
//! kernel inside a [`KernelEntry`] for the caller to put in a [`Generation`].
//!
//! Nothing here allocates device memory or launches anything, so it is safe at precopy time and
//! outside any capture.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::generation::KernelEntry;
use crate::kernel::{CubinMetadata, Kernel};
use crate::platform::NativeLibrary;
use crate::{Op, Result, platform};

/// One module of a program layer, as the package delivered it.
pub enum Module<'a> {
    /// A cubin, fatbin or PTX image, with its Triton metadata as JSON when the package carries it,
    /// and the symbol to resolve.
    Cubin {
        /// The image itself.
        image: &'a [u8],
        /// Triton's metadata record for the image, as the package wrote it.
        metadata: Option<&'a [u8]>,
        /// The kernel's symbol in the image.
        entry: &'a str,
    },
    /// A shared object the platform loads, and the C entry the package declares.
    Native {
        /// The library on disk.
        path: &'a Path,
        /// The entry point's name, constants and launch scalars as the package declared them.
        entry: &'a str,
    },
}

/// Every operator this build of `ops` has, by the name its programs are filed under.
///
/// A build without an operator's program packages still names it here: the table lists the crate's
/// operators, not the loaded modules, and [`install`] answers `None` for a name this table lacks.
const OPERATORS: &[&str] = &[
    <crate::activation::ActAndMul as Op>::NAME,
    <crate::activation::ValueFirstSwiglu as Op>::NAME,
    <crate::activation::ActAndMulE4M3 as Op>::NAME,
    <crate::activation::ValueFirstSwigluE4M3 as Op>::NAME,
    <crate::activation::Swiglu as Op>::NAME,
    <crate::block_copy::CacheBlockSnapshot as Op>::NAME,
    <crate::block_copy::CacheBlockScatter as Op>::NAME,
    <crate::commit::SlotCommit as Op>::NAME,
    <crate::conv::Conv3d as Op>::NAME,
    <crate::conv::Conv1d as Op>::NAME,
    <crate::conv::ConvTranspose1d as Op>::NAME,
    <crate::embedding::Embedding as Op>::NAME,
    <crate::frame::FramePad as Op>::NAME,
    <crate::frame::FrameMoments as Op>::NAME,
    <crate::frame::FrameNormPad as Op>::NAME,
    <crate::gemm::Gemm as Op>::NAME,
    <crate::kv::KvWrite as Op>::NAME,
    <crate::logits::LogitsGather as Op>::NAME,
    <crate::logits::VocabUnpad as Op>::NAME,
    <crate::metadata::AttentionGatherRows as Op>::NAME,
    <crate::metadata::AttentionGatherDecodeInputs as Op>::NAME,
    <crate::modulation::ModulatedRmsNorm as Op>::NAME,
    <crate::modulation::GatedResidual as Op>::NAME,
    <crate::noise::PhiloxNoise as Op>::NAME,
    <crate::norm::RmsNorm as Op>::NAME,
    <crate::norm::AddRmsNorm as Op>::NAME,
    <crate::norm::LayerNorm as Op>::NAME,
    <crate::norm::Softmax as Op>::NAME,
    <crate::prefix_cache::PageConversion as Op>::NAME,
    <crate::prefix_cache::CacheCopyRegion as Op>::NAME,
    <crate::residual::WeightedRmsNorm as Op>::NAME,
    <crate::residual::ScaledResidualRmsNorm as Op>::NAME,
    <crate::residual::ScaledResidualLayerNorm as Op>::NAME,
    <crate::residual::ScaledResidual as Op>::NAME,
    <crate::residual::ScaledAdd as Op>::NAME,
    <crate::rope::RotaryFactors as Op>::NAME,
    <crate::rope::QkNormRope as Op>::NAME,
    <crate::sampler::LogitsShape as Op>::NAME,
    <crate::sampler::TopK as Op>::NAME,
    <crate::sampler::FilterCandidates as Op>::NAME,
    <crate::sampler::InverseCdf as Op>::NAME,
    <crate::sampler::ArgMax as Op>::NAME,
    <crate::sampler::LogSoftmax as Op>::NAME,
    <crate::sampler::LogprobGather as Op>::NAME,
    <crate::segments::VsaZeroPadding as Op>::NAME,
    <crate::segments::VsaAddGatedTiles as Op>::NAME,
    <crate::segments::VsaComposeGatedRows as Op>::NAME,
    <crate::segments::VsaSelectTiles as Op>::NAME,
    <crate::segments::VsaPrepareTiles as Op>::NAME,
    <crate::attention::dense::DenseAttention as Op>::NAME,
    <crate::attention::PagedAttention as Op>::NAME,
    <crate::attention::sparse::SparseAttention as Op>::NAME,
    <crate::moe::combine::MoeUnpermute as Op>::NAME,
    <crate::moe::exchange::ExpertExchange as Op>::NAME,
    <crate::moe::grouped::GatherGemm as Op>::NAME,
    <crate::moe::grouped::RouteGemm as Op>::NAME,
    <crate::moe::routing::TopkSoftmax as Op>::NAME,
    <crate::moe::routing::ExpertGather as Op>::NAME,
    <crate::moe::sort::MoeSort as Op>::NAME,
    <crate::fp4::Nvfp4Quantize as Op>::NAME,
    <crate::fp4::Nvfp4Gemm as Op>::NAME,
    <crate::snake_beta::SnakeBeta as Op>::NAME,
    <crate::replicate_pad::ReplicatePad as Op>::NAME,
    <crate::audio_resample::AudioResample as Op>::NAME,
    <crate::resize::AntialiasedResizeUint8 as Op>::NAME,
    <crate::snake::Snake as Op>::NAME,
    <crate::bias::BiasAdd as Op>::NAME,
    <crate::gelu::Gelu as Op>::NAME,
    <crate::activation::Silu as Op>::NAME,
    <crate::adaptive_pool::AdaptivePool as Op>::NAME,
    <crate::rows::Rows as Op>::NAME,
    <crate::position::PositionInterpolate as Op>::NAME,
    <crate::audio_activation::AudioActivation as Op>::NAME,
    <crate::patch::Patch as Op>::NAME,
];

/// `name`, interned, when this build has an operator by that name.
///
/// The executor's precopy reads a package's operator names as text; this gives back the static name
/// the generation is keyed by, or `None` when no operator of this build answers to `name`.
pub fn operator_name(name: &str) -> Option<&'static str> {
    OPERATORS
        .iter()
        .copied()
        .find(|candidate| *candidate == name)
}

/// Every operator name this build has, in a stable order for a package's coverage check.
pub fn operators() -> &'static [&'static str] {
    OPERATORS
}

/// Load `module` as the kernel of `operator` for `descriptor_key`.
///
/// `Ok(None)` means this build of `ops` has no operator called `operator`: the module belongs to a
/// build that has, and installing it here would let a generation launch a kernel no layer asks for.
/// `Err` means the name is ours and the module does not load.
pub fn install(
    device: &platform::Device,
    operator: &str,
    descriptor_key: &str,
    module: Module<'_>,
) -> Result<Option<KernelEntry>> {
    let Some(name) = operator_name(operator) else {
        return Ok(None);
    };
    let kernel = load(device, module)?;
    Ok(Some(KernelEntry::keyed(
        name,
        descriptor_key,
        Arc::new(kernel),
    )))
}

/// Load one module into a kernel.
fn load(device: &platform::Device, module: Module<'_>) -> Result<Kernel> {
    match module {
        Module::Cubin {
            image,
            metadata,
            entry,
        } => {
            let mut metadata = match metadata {
                Some(json) => CubinMetadata::parse(json)?,
                None => CubinMetadata::named(entry),
            };
            metadata.name = entry.to_owned();
            Kernel::load_cubin(device, image, metadata)
        }
        Module::Native { path, entry } => {
            let entry = intern(entry)?;
            let library = NativeLibrary::open(path)?;
            Kernel::load_native(library, entry)
        }
    }
}

/// `entry` as a `&'static str`, one allocation per distinct name.
///
/// The platform resolves a native entry by a name that outlives the library, and a package may
/// install the same entry many times over (one per descriptor key), so the names are kept once.
fn intern(entry: &str) -> Result<&'static str> {
    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let names = NAMES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut names = names.lock().map_err(|_| crate::Error::Unsupported {
        operator: "install",
        problem: "an entry name was interned while another thread panicked".to_owned(),
    })?;
    if let Some(name) = names.get(entry) {
        return Ok(name);
    }
    let leaked: &'static str = Box::leak(entry.to_owned().into_boxed_str());
    names.insert(entry.to_owned(), leaked);
    Ok(leaked)
}
