//! Launches one installed kernel module on a device, from a JSON recipe.
//!
//! `program-server`'s `cubin_launch` module reads the recipe and checks its parameter words
//! against the module's published signature. This binary is the other half: it allocates the
//! buffers, copies the host files in, loads the cubin through `platform::Library`, launches it
//! with `Function::launch_ex` and a `LaunchConfig`, waits, and copies the buffers to be compared
//! back out.
//!
//! It is the one device-side member of this crate, and it exists because three packages are a
//! cubin with no launcher of their own — `topk_softmax`'s `nvcc` module and `paged_attention`'s
//! and `gather_gemm_nvfp4`'s `trtllm` modules — so no other route in the tree can exercise
//! their arithmetic. It takes `platform`'s launch surface as it is: no attribute list of its
//! own, no `cuLaunchKernel`, one `LaunchConfig` whose block, shared memory, cluster, PDL and
//! cooperative flag come from the module's `metadata.json`.
//!
//! ```text
//! cargo run -p program-server --bin cubin-launch -- --recipe case.json --dump artifacts/case
//! ```
//!
//! It needs a device: `Library::load` refuses an image the device cannot take, so a host without
//! one fails at that call with the driver's own answer rather than pretending to succeed.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use platform::memcpy_async;
use platform::{
    Cluster, Device, DeviceAllocation, DeviceId, LaunchConfig, Library, PinnedBuffer, Stream,
};
use program_server::cubin_launch::{Baked, Launch, Metadata, Recipe};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cubin-launch: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let Some(flags) = arguments()? else {
        return Ok(());
    };
    let mut recipe = Recipe::read(&flags.recipe)?;
    let recipe_path = flags.recipe.clone();
    if let Some(module) = flags.module {
        // A run's store root is its own, so a committed recipe names the module's path inside
        // the store and the run states the store.
        recipe.module = std::path::absolute(&module)
            .map_err(|error| format!("{}: {error}", module.display()))?;
    }
    if let Some(device) = flags.device {
        recipe.device = device;
    }
    let dump = flags.dump;
    let module = recipe.module_dir();
    let metadata = Metadata::read(&module)?;
    let baked = Baked::read(&module)?;
    let mut launch = Launch::plan(&recipe, &metadata, &baked)?;
    let checked = launch.check_signature(&metadata)?;
    println!("{}", launch.describe(&recipe, &metadata, &baked));
    if !checked {
        eprintln!(
            "{}: metadata.json states no signature, so the parameter words are launched unchecked",
            recipe.entry
        );
    }

    let device = Device::new(DeviceId(recipe.device))
        .map_err(|error| format!("device {}: {error}", recipe.device))?;
    let stream = device
        .stream()
        .map_err(|error| format!("device {}: {error}", recipe.device))?;

    let buffers = allocate(&recipe, &device, &stream, &module, &recipe_path)?;
    for word in &mut launch.words {
        if let Some(name) = word.buffer() {
            let buffer = buffers
                .get(name)
                .ok_or_else(|| format!("buffer {name} was not allocated"))?;
            word.set_address(buffer.address());
        }
    }

    let image_path = module.join(&recipe.cubin);
    let image =
        std::fs::read(&image_path).map_err(|error| format!("{}: {error}", image_path.display()))?;
    let library = Library::load(&device, &image)
        .map_err(|error| format!("{}: {error}", image_path.display()))?;
    let function = library
        .kernel(&launch.entry)
        .map_err(|error| format!("{}: {error}", launch.entry))?;

    let config = LaunchConfig {
        grid: launch.grid,
        block: launch.block,
        shared_memory: launch.shared,
        cluster: launch.cluster.map(Cluster::new),
        pdl: launch.pdl,
        cooperative: launch.cooperative,
    };
    let mut parameters: Vec<*mut c_void> = launch
        .words
        .iter_mut()
        .map(|word| word.parameter())
        .collect();
    function
        .launch_ex(&stream, &config, &mut parameters)
        .map_err(|error| format!("launching {}: {error}", launch.entry))?;
    stream
        .synchronize()
        .map_err(|error| format!("the launch: {error}"))?;

    let written = dump_buffers(&recipe, &buffers, &stream, &module, dump)?;
    println!(
        "launched {} on device {}: grid {:?} block {:?} shared {} cluster {:?} pdl {} \
         cooperative {}; {} buffers written",
        launch.entry,
        recipe.device,
        launch.grid,
        launch.block,
        launch.shared,
        launch.cluster,
        launch.pdl,
        launch.cooperative,
        written
    );
    Ok(())
}

/// What a run states on its command line.
struct Flags {
    recipe: PathBuf,
    /// The module directory, overriding the recipe's own.
    module: Option<PathBuf>,
    /// The device, overriding the recipe's own.
    device: Option<i32>,
    /// Where buffers with a `dump` are written, overriding the recipe's own directory.
    dump: Option<PathBuf>,
}

/// The flags, or nothing when help was asked for.
fn arguments() -> Result<Option<Flags>, String> {
    let mut recipe = None;
    let mut module = None;
    let mut device = None;
    let mut dump = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--recipe" => recipe = Some(PathBuf::from(next_value(&mut arguments, &flag)?)),
            "--module" => module = Some(PathBuf::from(next_value(&mut arguments, &flag)?)),
            "--device" => {
                let value = next_value(&mut arguments, &flag)?;
                device = Some(
                    value
                        .parse()
                        .map_err(|_| format!("--device takes a device number, not {value}"))?,
                );
            }
            "--dump" => dump = Some(PathBuf::from(next_value(&mut arguments, &flag)?)),
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(None);
            }
            other => return Err(format!("unknown flag {other}; try --help")),
        }
    }
    let recipe = recipe.ok_or("--recipe is required")?;
    Ok(Some(Flags {
        recipe,
        module,
        device,
        dump,
    }))
}

/// The flag's value, or the error that names the flag.
fn next_value(
    arguments: &mut std::iter::Skip<std::env::Args>,
    flag: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} needs a value"))
}

const HELP: &str = "\
cubin-launch --recipe RECIPE [--dump DIRECTORY]

Launches one installed kernel module on a device, from a JSON recipe, through `platform`'s
library and launch surface. The recipe names the module directory, the entry, the grid, the
device buffers and the parameter words in declaration order; the block, shared memory, cluster,
PDL and cooperative flag come from the module's own metadata.json.

  --recipe PATH   the JSON recipe
  --module PATH   the module directory, overriding the recipe's own (a store's module path)
  --device N      the device, overriding the recipe's own (0 by default)
  --dump PATH     where buffers with a `dump` are written; the recipe's directory by default
  -h, --help      this text
";

/// Allocates the recipe's buffers and copies their host files in.
fn allocate(
    recipe: &Recipe,
    device: &Device,
    stream: &Stream,
    module: &Path,
    recipe_path: &Path,
) -> Result<BTreeMap<String, DeviceAllocation>, String> {
    let mut buffers = BTreeMap::new();
    for (name, plan) in &recipe.buffers {
        let buffer = DeviceAllocation::linear(device.id(), plan.bytes)
            .map_err(|error| format!("buffer {name} of {} bytes: {error}", plan.bytes))?;
        if let Some(fill) = &plan.fill {
            let source = host_path(fill, module, beside(recipe_path));
            let bytes =
                std::fs::read(&source).map_err(|error| format!("{}: {error}", source.display()))?;
            if bytes.len() > plan.bytes {
                return Err(format!(
                    "{} holds {} bytes, buffer {name} holds {}",
                    source.display(),
                    bytes.len(),
                    plan.bytes
                ));
            }
            let staged = PinnedBuffer::new(device.id(), bytes.len().max(1))
                .map_err(|error| format!("staging {name}: {error}"))?;
            // SAFETY: the staged buffer holds `bytes.len()` bytes, and the source is that long.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), staged.as_ptr(), bytes.len());
            }
            // SAFETY: both ranges are live until the copy completes, which the synchronize below
            // waits for; the destination is this buffer's storage on the stream's device.
            unsafe { memcpy_async(buffer.address(), staged.address(), bytes.len(), stream) }
                .map_err(|error| format!("filling {name}: {error}"))?;
            stream
                .synchronize()
                .map_err(|error| format!("filling {name}: {error}"))?;
        }
        buffers.insert(name.clone(), buffer);
    }
    Ok(buffers)
}

/// Copies the buffers the recipe asks for back to the host.
fn dump_buffers(
    recipe: &Recipe,
    buffers: &BTreeMap<String, DeviceAllocation>,
    stream: &Stream,
    module: &Path,
    dump: Option<PathBuf>,
) -> Result<usize, String> {
    let directory = dump.unwrap_or_else(|| recipe.dump_dir());
    let mut written = 0;
    for (name, plan) in &recipe.buffers {
        let Some(destination) = &plan.dump else {
            continue;
        };
        let buffer = buffers
            .get(name)
            .ok_or_else(|| format!("buffer {name} was not allocated"))?;
        let staged = PinnedBuffer::new(buffer.device(), plan.bytes)
            .map_err(|error| format!("staging {name}: {error}"))?;
        // SAFETY: the device range is the buffer's storage and the host range the staged buffer's,
        // both live until the synchronized copy completes.
        unsafe { memcpy_async(staged.address(), buffer.address(), plan.bytes, stream) }
            .map_err(|error| format!("reading {name}: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("reading {name}: {error}"))?;
        let path = host_path(destination, module, &directory);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        // SAFETY: the staged buffer holds `plan.bytes` bytes beyond `as_ptr`.
        let bytes = unsafe { std::slice::from_raw_parts(staged.as_ptr(), plan.bytes) };
        std::fs::write(&path, bytes).map_err(|error| format!("{}: {error}", path.display()))?;
        written += 1;
    }
    Ok(written)
}

/// The directory a recipe at `recipe` names its relative files from.
fn beside(recipe: &Path) -> &Path {
    recipe.parent().unwrap_or_else(|| Path::new("."))
}

/// A recipe's relative path, taken against its module directory first and the recipe's second.
fn host_path(path: &Path, module: &Path, base: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let beside_module = module.join(path);
    if beside_module.exists() {
        return beside_module;
    }
    base.join(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recipe written beside its inputs names them relatively, and they are found there.
    #[test]
    fn a_relative_fill_is_taken_from_the_recipes_directory() {
        let recipe = Path::new("/cases/topk/recipe.json");
        assert_eq!(
            host_path(
                Path::new("logits.bin"),
                Path::new("/modules/none"),
                beside(recipe)
            ),
            PathBuf::from("/cases/topk/logits.bin")
        );
        assert_eq!(beside(Path::new("recipe.json")), Path::new(""));
    }

    #[test]
    fn a_file_beside_the_module_comes_first_and_an_absolute_path_stands() {
        let module = tempfile::tempdir().unwrap();
        std::fs::write(module.path().join("table.bin"), b"").unwrap();
        assert_eq!(
            host_path(Path::new("table.bin"), module.path(), Path::new("/cases")),
            module.path().join("table.bin")
        );
        assert_eq!(
            host_path(
                Path::new("/data/table.bin"),
                module.path(),
                Path::new("/cases")
            ),
            PathBuf::from("/data/table.bin")
        );
    }
}
