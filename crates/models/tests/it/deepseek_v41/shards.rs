//! The release's tensors read straight from its safetensors shards: the bytes of a tensor, or of
//! some of its rows, by the shard index.
//!
//! The library's checkpoint reader indexes every tensor of a shard it opens and refuses the
//! UE8M0 block scales the release's shards hold beside the tensors these tests read, so the
//! tests read the format themselves: an eight-byte little-endian header length, the JSON header
//! of every tensor's element type, extents and byte range, then the bytes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One tensor's place in its shard.
struct Place {
    file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    /// The absolute byte offset of its first byte.
    start: u64,
}

/// The tensors of a release, by name.
pub(crate) struct Shards {
    places: HashMap<String, Place>,
}

impl Shards {
    /// The tensors the shard index of the snapshot `root` lists.
    pub(crate) fn open(root: &Path) -> Self {
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("model.safetensors.index.json")).expect("the shard index"),
        )
        .expect("the shard index is JSON");
        let mut files: Vec<&str> = index["weight_map"]
            .as_object()
            .expect("a weight map")
            .values()
            .filter_map(|file| file.as_str())
            .collect();
        files.sort_unstable();
        files.dedup();

        let mut places = HashMap::new();
        for file in files {
            let path = root.join(file);
            let mut shard = File::open(&path).expect("a shard");
            let mut length = [0_u8; 8];
            shard
                .read_exact(&mut length)
                .expect("a shard's header length");
            let length = u64::from_le_bytes(length);
            let mut header = vec![0_u8; length as usize];
            shard.read_exact(&mut header).expect("a shard's header");
            let header: serde_json::Value = serde_json::from_slice(&header).expect("JSON");
            for (name, entry) in header.as_object().expect("a header object") {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = entry["data_offsets"].as_array().expect("offsets");
                places.insert(
                    name.clone(),
                    Place {
                        file: path.clone(),
                        dtype: entry["dtype"].as_str().expect("a dtype").to_owned(),
                        shape: entry["shape"]
                            .as_array()
                            .expect("a shape")
                            .iter()
                            .map(|extent| extent.as_u64().expect("an extent") as usize)
                            .collect(),
                        start: 8 + length + offsets[0].as_u64().expect("an offset"),
                    },
                );
            }
        }
        Self { places }
    }

    fn place(&self, name: &str, dtype: &str) -> &Place {
        let place = self
            .places
            .get(name)
            .unwrap_or_else(|| panic!("the release holds no {name}"));
        assert_eq!(place.dtype, dtype, "{name}'s element type");
        place
    }

    /// The extents of `name`.
    pub(crate) fn shape(&self, name: &str) -> Vec<usize> {
        self.places[name].shape.clone()
    }

    /// The bytes of rows `rows` of `name`, a tensor of `dtype` whose rows are `row` bytes.
    pub(crate) fn rows(&self, name: &str, dtype: &str, row: usize, rows: &[usize]) -> Vec<u8> {
        let place = self.place(name, dtype);
        let mut shard = File::open(&place.file).expect("a shard");
        let mut bytes = vec![0_u8; rows.len() * row];
        for (index, &row_index) in rows.iter().enumerate() {
            shard
                .seek(SeekFrom::Start(place.start + (row_index * row) as u64))
                .expect("a seek");
            shard
                .read_exact(&mut bytes[index * row..(index + 1) * row])
                .expect("a row");
        }
        bytes
    }

    /// Every byte of `name`, a tensor of `dtype` and `width` bytes an element.
    pub(crate) fn bytes(&self, name: &str, dtype: &str, width: usize) -> Vec<u8> {
        let count: usize = self.shape(name).iter().product();
        self.rows(name, dtype, count * width, &[0])
    }

    /// `name`, an FP32 tensor, as its values.
    pub(crate) fn f32(&self, name: &str) -> Vec<f32> {
        self.bytes(name, "F32", 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    /// `name`, a BF16 tensor, widened to FP32.
    pub(crate) fn bf16(&self, name: &str) -> Vec<f32> {
        widen(&self.bytes(name, "BF16", 2))
    }

    /// Rows `rows` of `name`, a BF16 matrix, widened to FP32.
    pub(crate) fn bf16_rows(&self, name: &str, rows: &[usize]) -> Vec<f32> {
        let width = self.shape(name)[1];
        widen(&self.rows(name, "BF16", 2 * width, rows))
    }
}

/// BF16 bytes widened to FP32.
fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
        .collect()
}
