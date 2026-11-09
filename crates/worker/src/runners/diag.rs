//! The served run's half of the block-by-block diagnosis (`specs/worker.md` 10).
//!
//! The protocol: both implementations write the named tensors they
//! computed into their own tree at `<block>/<name>.npy`, and `tools/diagnosis/compare.py` pairs the
//! two trees by that relative path. The reference half exists - `crates/models/minimax-h3/tools/`
//! holds a dumper per block - and the served half did not: no crate read `UNISERVE_DIAG_DIR`, so a
//! served run wrote no dump and a report read every name as `missing_in_got`. This module is that
//! half.
//!
//! **Nothing here runs unless `UNISERVE_DIAG_DIR` names a directory.** The variable is read once,
//! when a runner binds; an unset variable costs no allocation, no copy and no branch inside a step,
//! which is the rule that keeps a diagnosis out of the control path of every request.
//!
//! **A failed dump never fails a request.** The design calls this a diagnosis and never an
//! acceptance gate, so a directory that cannot be written, a tensor that has no byte width and a
//! device copy that fails are all reported at `warn` and the run goes on. The alternative - a
//! diagnosis that stops serving - is the gate the design forbids.
//!
//! **The container is `.npy`, and the numbers are the reader's width.** A dump is written as version
//! 1.0, C order, `<f4` or `<f2`: the two element types `numpy.load` reads whole. A tensor of any
//! other type is **widened to FP32 here**, which is stated rather than silent - the comparator widens
//! a reader anyway ("keeps BF16/FP16 native and widens the reader"), so a BF16 deployment's dump is
//! that deployment's numbers at the only width a plain `.npy` carries.

use std::path::{Path, PathBuf};

use ops::{DType, TensorView};
use platform::{DeviceId, PinnedBuffer};

use crate::{Error, Result};

/// The directory a served run writes its diagnosis into, or `None` when the diagnosis is off.
pub(crate) struct Diag {
    dir: Option<PathBuf>,
}

impl Diag {
    /// Reads `UNISERVE_DIAG_DIR`: a directory to write, or nothing to do.
    ///
    /// An empty value is off rather than a directory named `""`: a shell that exports the variable
    /// and leaves it empty means "not now", and writing into the working directory would be a
    /// surprise no message announces.
    pub(crate) fn from_env() -> Self {
        let dir = std::env::var_os("UNISERVE_DIAG_DIR")
            .map(PathBuf::from)
            .filter(|dir| !dir.as_os_str().is_empty());
        Self { dir }
    }

    /// The same diagnosis for rank `rank` of a group: the first rank writes where one rank does, so
    /// the trees of a one-rank and a many-rank run pair by path, and each other rank beside it.
    pub(crate) fn for_rank(mut self, rank: u32) -> Self {
        if rank > 0 {
            self.dir = self.dir.map(|dir| dir.join(format!("rank{rank}")));
        }
        self
    }

    /// Whether a step runs operator by operator rather than from its graph: `UNISERVE_DIAG_EAGER`.
    pub(crate) fn eager(&self) -> bool {
        self.dir.is_some() && std::env::var_os("UNISERVE_DIAG_EAGER").is_some()
    }

    /// Whether a dump will be written.
    pub(crate) fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// Writes `view` as `<block>/<name>.npy` under the diagnosis directory.
    ///
    /// The copy is stream-ordered after the work that produced `view` and synchronized before the
    /// file is written, so the file holds the tensor the run computed rather than whatever the
    /// device had written by the time the host read it.
    pub(crate) fn write(
        &self,
        device: DeviceId,
        stream: &platform::Stream,
        block: &str,
        name: &str,
        view: TensorView,
    ) -> Result<()> {
        let Some(root) = self.dir.as_ref() else {
            return Ok(());
        };
        self.dump(root, device, stream, block, name, view)
            .map_err(|error| {
                // A diagnosis reports and never gates: the run goes on and the failure is named.
                tracing::warn!(
                    block,
                    name,
                    error = %error,
                    "a diagnosis dump was not written"
                );
                error
            })
    }

    /// Writes `bytes` as `<block>/<name>` under the diagnosis directory, as they stand.
    ///
    /// For what a tensor cannot hold: the request's own plan, which the reference half reads to run
    /// the same request.
    pub(crate) fn write_bytes(&self, block: &str, name: &str, bytes: &[u8]) {
        let Some(root) = self.dir.as_ref() else {
            return;
        };
        let directory = root.join(block);
        if let Err(error) = std::fs::create_dir_all(&directory)
            .and_then(|()| std::fs::write(directory.join(name), bytes))
        {
            tracing::warn!(block, name, error = %error, "a diagnosis dump was not written");
        }
    }

    fn dump(
        &self,
        root: &Path,
        device: DeviceId,
        stream: &platform::Stream,
        block: &str,
        name: &str,
        view: TensorView,
    ) -> Result<()> {
        let bytes = view.byte_len()?;
        let host = PinnedBuffer::new(
            device,
            usize::try_from(bytes)
                .map_err(|_| Error::Invalid("a dump is larger than memory".into()))?,
        )?;
        // SAFETY: source and destination are `view`'s own device range and this host allocation of
        // exactly its length; the copy is ordered on `stream`, which is synchronized below before
        // the host bytes are read.
        unsafe {
            platform::memcpy_async(host.address(), view.ptr.address(), host.len(), stream)?;
        }
        stream.synchronize()?;
        // SAFETY: the copy above finished, the destination is this allocation of `host.len()`
        // bytes, and nothing else reads it.
        let payload = unsafe { std::slice::from_raw_parts(host.as_ptr(), host.len()) };
        let (dtype, bytes) = widen(view.dtype, payload)?;
        let directory = root.join(block);
        std::fs::create_dir_all(&directory)
            .map_err(|error| Error::Invalid(format!("{}: {error}", directory.display())))?;
        let path = directory.join(format!("{name}.npy"));
        let file = npy(view.shape.dims(), dtype, &bytes)?;
        std::fs::write(&path, file)
            .map_err(|error| Error::Invalid(format!("{}: {error}", path.display())))?;
        Ok(())
    }
}

/// The element type a dump is written in, and its bytes: `f32`, `f16` and `i32` as they stand, and
/// `bf16` widened to `f32`.
///
/// Widening is by the element's own value: BF16 is the top sixteen bits of an FP32, so a shift is
/// the exact widening; the same is true of the low half, which is why `f16` is converted rather
/// than shifted.
fn widen(dtype: DType, payload: &[u8]) -> Result<(DType, Vec<u8>)> {
    match dtype {
        DType::F32 => Ok((DType::F32, payload.to_vec())),
        DType::F16 => Ok((DType::F16, payload.to_vec())),
        // A table of indices is compared exactly, so it keeps its integers.
        DType::I32 => Ok((DType::I32, payload.to_vec())),
        DType::BF16 => {
            let mut out = Vec::with_capacity(payload.len() * 2);
            for pair in payload.as_chunks::<2>().0 {
                let value = u16::from_le_bytes(*pair);
                out.extend_from_slice(&(u32::from(value) << 16).to_le_bytes());
            }
            Ok((DType::F32, out))
        }
        other => Err(Error::Invalid(format!(
            "a dump is `f4`, `f2` or `i4` and this tensor is {other}"
        ))),
    }
}

/// One `.npy` file: version 1.0, little-endian, C order.
///
/// The header arithmetic is `numpy`'s: magic (6 bytes), version (2), a two-byte little-endian header
/// length, then the dictionary padded with spaces and closed by a newline so the preamble ends on a
/// 64-byte boundary. A reader rejects a short pad, so the padding is not cosmetic.
fn npy(shape: &[u64], dtype: DType, payload: &[u8]) -> Result<Vec<u8>> {
    let descr = match dtype {
        DType::F32 => "<f4",
        DType::F16 => "<f2",
        DType::I32 => "<i4",
        other => {
            return Err(Error::Invalid(format!(
                "a dump is written as `f4`, `f2` or `i4`, not {other}"
            )));
        }
    };
    let width = dtype
        .bytes()
        .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?;
    let elements: u64 = shape.iter().product();
    if payload.len() as u64 != elements * width {
        return Err(Error::Invalid(format!(
            "{} bytes for {elements} elements of {dtype}",
            payload.len()
        )));
    }
    let extents = shape
        .iter()
        .map(|extent| extent.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    // A one-dimensional array still writes the trailing comma numpy writes; a scalar writes
    // `()` and reaches here as an empty shape.
    let spec = if shape.is_empty() {
        "()".to_owned()
    } else {
        format!("({extents},)")
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {spec}, }}");
    let preamble = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - preamble % 64) % 64));
    header.push('\n');
    let length = u16::try_from(header.len())
        .map_err(|_| Error::Invalid("an npy header is longer than its field".into()))?;
    let mut bytes = Vec::with_capacity(10 + header.len() + payload.len());
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_variable_is_off() {
        // SAFETY: the test binary owns this variable and no other test reads it.
        unsafe { std::env::remove_var("UNISERVE_DIAG_DIR") };
        assert!(!Diag::from_env().enabled());
        // SAFETY: as above.
        unsafe { std::env::set_var("UNISERVE_DIAG_DIR", "") };
        assert!(
            !Diag::from_env().enabled(),
            "an empty value is off, not a directory named `\"\"`"
        );
        // SAFETY: as above.
        unsafe { std::env::set_var("UNISERVE_DIAG_DIR", "/tmp/h3-diag") };
        assert!(Diag::from_env().enabled());
        // SAFETY: as above; the variable does not outlive the test.
        unsafe { std::env::remove_var("UNISERVE_DIAG_DIR") };
    }

    #[test]
    fn a_header_ends_on_a_64_byte_boundary_and_states_the_type() {
        let bytes = npy(&[2, 3], DType::F32, &[0u8; 24]).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert_eq!((bytes[6], bytes[7]), (1, 0));
        let length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(
            (10 + length) % 64,
            0,
            "the preamble does not end on a boundary"
        );
        let header = std::str::from_utf8(&bytes[10..10 + length]).unwrap();
        assert!(header.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3,), }"));
        assert!(header.ends_with('\n'));
        assert_eq!(bytes.len(), 10 + length + 24);
    }

    #[test]
    fn a_half_precision_dump_states_f2() {
        let bytes = npy(&[4], DType::F16, &[0u8; 8]).unwrap();
        let length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + length]).unwrap();
        assert!(header.contains("'<f2'"), "{header}");
        assert_eq!(bytes.len(), 10 + length + 8);
    }

    #[test]
    fn a_payload_of_the_wrong_length_is_refused_by_name() {
        let error = npy(&[3], DType::F32, &[0u8; 8]).unwrap_err();
        assert!(format!("{error}").contains("8 bytes for 3 elements of f32"));
    }

    #[test]
    fn bf16_widens_by_the_element_and_f16_is_kept() {
        // 1.0 in BF16 is 0x3F80, whose FP32 is 0x3F800000.
        let (dtype, bytes) = widen(DType::BF16, &0x3F80u16.to_le_bytes()).unwrap();
        assert_eq!(dtype, DType::F32);
        assert_eq!(bytes, 0x3F80_0000u32.to_le_bytes().to_vec());

        let half = 1.0f32.to_bits() as u16;
        let (dtype, bytes) = widen(DType::F16, &half.to_le_bytes()).unwrap();
        assert_eq!(dtype, DType::F16, "a half is written as a half");
        assert_eq!(bytes, half.to_le_bytes().to_vec());
    }

    /// A type that cannot be written as a `.npy` is refused with the type's own name, and the name
    /// is `DType`'s rendering - lower case, the same one every other refusal in the tree prints. The
    /// test pinned the name in capitals, which is what the type printed before `Debug` was made to
    /// agree with `Display`; the refusal itself never lost its reason.
    #[test]
    fn a_type_a_dump_does_not_hold_is_refused() {
        let error = widen(DType::I64, &[0u8; 8]).unwrap_err();
        let text = format!("{error}");
        assert!(
            text.contains("a dump is `f4`, `f2` or `i4`"),
            "the refusal no longer says what a dump may be: {text}"
        );
        assert!(
            text.contains(&format!("this tensor is {}", DType::I64)),
            "the refusal does not name the type it was given: {text}"
        );
    }

    /// A table of indices is compared exactly, so its integers reach the file as they stand.
    #[test]
    fn an_index_table_keeps_its_integers() {
        let table = 7_i32.to_le_bytes();
        let (dtype, bytes) = widen(DType::I32, &table).unwrap();
        assert_eq!(dtype, DType::I32);
        assert_eq!(bytes, table.to_vec());
    }
}
