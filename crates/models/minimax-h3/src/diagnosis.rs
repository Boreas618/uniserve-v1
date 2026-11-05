//! The container the block-by-block diagnosis writes: `.npy`, as `numpy.save` writes it.
//!
//! `notes/h3/diagnosis.md` states the protocol: both implementations write the tensors they computed
//! into their own tree, the reference's through `tools/dump_*.py` and this crate's through a
//! harness's `--diag <dir>`, and `tools/diagnosis/compare.py` pairs the two trees by relative path.
//! The two `.h3t` containers beside these files are this crate's own and are read by
//! `examples/compare_blocks.rs`; a `.npy` is what a comparison with the reference can use, so a
//! tensor that is meant to be compared is written in this one.
//!
//! **Nothing here is a diagnostic hook.** No model launch reaches this module: it encodes bytes a
//! caller already has on the host, and a run that names no `--diag` never calls it. A hook in the
//! model would put a file write in the control path of every request, which is why the protocol puts
//! the runner's dumps in the executor and leaves the model pure arithmetic.

use std::path::Path;

/// Writes one `.npy` tensor: version 1.0, little-endian floats, C order.
///
/// `payload` is the tensor's bytes exactly as the device produced them, so a `.npy` written here and
/// the `.h3t` written beside it are the same numbers in two containers rather than two computations.
/// The shape is in elements and the payload must hold `4 * product(shape)` bytes, which is why this
/// function takes `f32` data only: the reference's `numpy.save` output reads with it, and a type the
/// reader must widen is a type the comparator widens on its own side.
///
/// The header arithmetic is `numpy`'s own: a preamble of magic (6 bytes), version (2), a 2-byte
/// little-endian header length, then a dictionary padded with spaces and closed by a newline so that
/// the preamble is a multiple of 64 bytes. A reader rejects a file whose padding is short, so this is
/// not a cosmetic detail.
pub fn write_npy(path: &Path, shape: &[u64], payload: &[u8]) -> Result<(), String> {
    let elements: u64 = shape.iter().product();
    if payload.len() as u64 != elements * 4 {
        return Err(format!(
            "{}: {} bytes for {} elements of f32",
            path.display(),
            payload.len(),
            elements
        ));
    }
    let extents = shape
        .iter()
        .map(|extent| extent.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({extents},), }}");
    let preamble = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - preamble % 64) % 64));
    header.push('\n');

    let mut bytes = Vec::with_capacity(10 + header.len() + payload.len());
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

/// The `f32` bytes of a slice, which is the payload [`write_npy`] takes.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
