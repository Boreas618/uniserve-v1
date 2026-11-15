//! The tensors a reference implementation wrote to a safetensors file, read with the checkpoint
//! reader the families load with.

use std::collections::HashMap;
use std::path::Path;

use nn::{Checkpoint, Device, Tensor};

/// The tensors of the safetensors file at `path`, by name, as CPU tensors of the element type
/// and extents the file states.
pub(crate) fn tensors(path: &Path) -> HashMap<String, Tensor> {
    let checkpoint =
        Checkpoint::open_file(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));

    checkpoint
        .names()
        .map(|name| {
            let location = checkpoint.locate(name).unwrap();
            let bytes = checkpoint.read(name).unwrap();
            let tensor = Tensor::from_raw_buffer(
                &bytes,
                location.dtype,
                location.shape.dims(),
                &Device::Cpu,
            )
            .unwrap_or_else(|error| panic!("{}: {name}: {error}", path.display()));
            (name.to_owned(), tensor)
        })
        .collect()
}
