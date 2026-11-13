//! The tensors a reference implementation wrote to a safetensors file, read with the checkpoint
//! reader the families load with.

use std::collections::HashMap;
use std::path::Path;

use nn::{Checkpoint, Device, Tensor};

/// The tensors of the safetensors file at `path`, by name, as CPU tensors of the element type
/// and extents the file states.
///
/// The file's directory is opened as a checkpoint, whose shards are every safetensors file it
/// holds; the tensors of the other shards are left out.
pub(crate) fn tensors(path: &Path) -> HashMap<String, Tensor> {
    let directory = path
        .parent()
        .unwrap_or_else(|| panic!("{} is in no directory", path.display()));
    let checkpoint = Checkpoint::open(directory)
        .unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
    let file = checkpoint
        .files()
        .iter()
        .position(|file| file.path() == path)
        .unwrap_or_else(|| panic!("{} is not a shard of its directory", path.display()));

    checkpoint
        .names()
        .filter_map(|name| {
            let location = checkpoint.locate(name).unwrap();
            if location.file != file {
                return None;
            }

            let bytes = checkpoint.read(name).unwrap();
            let tensor = Tensor::from_raw_buffer(
                &bytes,
                location.dtype,
                location.shape.dims(),
                &Device::Cpu,
            )
            .unwrap_or_else(|error| panic!("{}: {name}: {error}", path.display()));
            Some((name.to_owned(), tensor))
        })
        .collect()
}
