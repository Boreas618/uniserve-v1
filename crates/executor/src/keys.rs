//! The keys of derived state at the Storage Server (`specs/worker.md` §8).
//!
//! The server holds storage and never interprets it: every key is computed
//! here, by the process that derives the state, from the inputs that decide
//! what the stored bytes mean. The inputs are, per kind of derived state:
//!
//! | Derived state | Key inputs |
//! | --- | --- |
//! | retained weights | the identity of each checkpoint file the tensor is read from, the element type the load's `dtype` and `quantization` produce, the layout name the reading operator declares, and the tensor's region in the logical tensor |
//! | KV pages, recursive state checkpoints | the hash of the token prefix, the model's representation, the epoch's split shape, and the region |
//! | encoder outputs | the hash of the input media, and the model's representation |
//! | latents | the request's trajectory and its step count, and the model's representation |
//!
//! A key carries no program's backend name or revision: replacing a kernel
//! program keeps the mathematical content, so derived state keeps hitting
//! (`specs/worker.md` §7.3). A change of mathematics, precision or format is a
//! build-config change, which changes the representation and so the key.
//!
//! This replaces the structural digest of the retired loader (the versions of
//! torch, triton, flashinfer, cutlass-dsl and safetensors) and the function and
//! format numbers of `specs/online-operations.md` §3.2.

use std::ops::Range;
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;

use sha2::{Digest as _, Sha256};
use storage::Key;

use crate::tensor::DType;

/// Domain tag of retained weights, so a key of one kind never equals a key of
/// another kind with the same inputs.
const WEIGHT_TAG: &[u8] = b"uniserve-retained-weight";
const KV_TAG: &[u8] = b"uniserve-kv-page";
const ENCODER_TAG: &[u8] = b"uniserve-encoder-output";
const LATENT_TAG: &[u8] = b"uniserve-latent";

/// How long ago a checkpoint file must have last changed for its identity to
/// name its contents: longer than the kernel's timestamp resolution.
const SETTLED_NS: i128 = 1_000_000_000;

/// A domain-tagged digest of length-prefixed parts. Every part is length
/// prefixed, so no two different sequences of parts produce the same bytes.
struct Tagged(Sha256);

impl Tagged {
    fn new(tag: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update((tag.len() as u64).to_le_bytes());
        digest.update(tag);
        Self(digest)
    }

    /// Add one part, length prefixed.
    fn part(&mut self, part: &[u8]) {
        self.0.update((part.len() as u64).to_le_bytes());
        self.0.update(part);
    }

    fn text(&mut self, part: &str) {
        self.part(part.as_bytes());
    }

    fn number(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn finish(self) -> Key {
        Key(self.0.finalize().into())
    }
}

/// The identity of one checkpoint file: what its bytes are, in the sense the
/// load of §1 defines.
///
/// A file of a Hugging Face cache is a link to a blob named by a digest of its
/// contents (SHA-256 for a large file, the git blob SHA-1 otherwise), and the
/// cache writes a blob once: the blob's name is its identity. Any other file's
/// identity is its device, inode, modification time and status change time. The
/// kernel sets a file's status change time on every write, and no process can
/// set it back, so a file rewritten after it was read has another identity, and
/// a file at another path is another file. The times have the kernel clock
/// tick's resolution, though: a file written within the last second may be
/// written again under the same time, so its identity is unique to this read
/// and its weights match no later load.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileIdentity(String);

impl FileIdentity {
    /// The identity of the file at `path`, a link followed.
    pub fn of(path: &Path) -> crate::Result<Self> {
        let resolved = path.canonicalize().map_err(|error| {
            crate::Error::transport_source(
                format!("reading the identity of {}", path.display()),
                error,
            )
        })?;
        if let Some(name) = resolved.file_name().and_then(|name| name.to_str())
            && is_blob(&resolved, name)
        {
            return Ok(Self(format!("blob:{name}")));
        }
        let status = std::fs::metadata(&resolved).map_err(|error| {
            crate::Error::transport_source(
                format!("reading the identity of {}", path.display()),
                error,
            )
        })?;
        let now = now_ns();
        let changed = nanos(status.ctime(), status.ctime_nsec());
        let settled = now.saturating_sub(changed);
        if settled < SETTLED_NS {
            // The file may still be rewritten within this timestamp, so this
            // read's identity is its own and matches no later load.
            return Ok(Self(format!("unsettled:{}", uuid::Uuid::new_v4().simple())));
        }
        Ok(Self(format!(
            "{}:{}:{}:{}",
            status.dev(),
            status.ino(),
            nanos(status.mtime(), status.mtime_nsec()),
            nanos(status.ctime(), status.ctime_nsec())
        )))
    }

    /// The identity as a key input.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Whether `resolved`, named `name`, is a Hugging Face cache blob: a file in a
/// `blobs` directory whose name is the SHA-256 or git SHA-1 of its contents.
fn is_blob(resolved: &Path, name: &str) -> bool {
    let in_blobs = resolved
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|parent| parent == "blobs");
    in_blobs
        && (name.len() == 64 || name.len() == 40)
        && name
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// A file time as nanoseconds since the Unix epoch, the resolution the
/// kernel's clock keeps.
fn nanos(seconds: i64, nanos: i64) -> i128 {
    i128::from(seconds) * 1_000_000_000 + i128::from(nanos)
}

/// The current time in nanoseconds since the Unix epoch.
fn now_ns() -> i128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(since) => since.as_nanos() as i128,
        Err(_) => 0,
    }
}

/// The element type a load's `dtype` and `quantization` produce: the
/// numerical representation the stored bytes are in (`specs/worker.md` §1,
/// §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementType<'a> {
    pub dtype: DType,
    /// The quantization method the checkpoint declares or the configuration
    /// names, as SGLang spells it (`fp8`, `modelopt_fp4`, ...); `None` for a
    /// dense load.
    pub quantization: Option<&'a str>,
}

impl ElementType<'_> {
    /// The representation's spelling in a key: the element type and, when it
    /// applies, the quantization that decides how the bytes are laid out.
    fn spell(&self) -> String {
        match self.quantization {
            Some(quantization) => format!("{}:{quantization}", self.dtype.name()),
            None => self.dtype.name().to_owned(),
        }
    }
}

/// The model's representation: the part of a key that changes when the same
/// request's state would hold different bytes (`specs/worker.md` §8, §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Representation<'a> {
    pub element: ElementType<'a>,
    /// The KV layout the pages are in: their element type and page size. It
    /// is part of the build config, so a change of it is a new epoch.
    pub kv: KvLayout,
}

/// The layout of the KV pages of an epoch: the build config's `kv_cache_config`
/// as it affects the pages' bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    /// Tokens one page holds.
    pub page_tokens: u32,
    /// Element type of a page's entries.
    pub element: DType,
}

/// The epoch's split shape: how the model's tensors and sequence are divided
/// among ranks. A change of it is a new epoch and changes every derived key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// `tensor_parallel_size` of the epoch.
    pub tensor_parallel: u32,
    /// Ulysses sequence-parallel degree of the epoch.
    pub ulysses: u32,
}

/// A region of a logical tensor, in elements: which part of a checkpoint's
/// tensor a retained weight holds (a shard of a tensor-parallel weight, a field
/// of an encoded weight, one layer's slice).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Region(Vec<Range<u64>>);

impl Region {
    /// The whole tensor.
    pub fn whole() -> Self {
        Self(Vec::new())
    }

    /// A region of `axes` half-open ranges, one per logical axis.
    pub fn new(axes: impl IntoIterator<Item = Range<u64>>) -> Self {
        Self(axes.into_iter().collect())
    }

    fn tag(&self, digest: &mut Tagged) {
        digest.number(self.0.len() as u64);
        for axis in &self.0 {
            digest.number(axis.start);
            digest.number(axis.end);
        }
    }
}

/// The key of a retained weight.
///
/// `files` are the identities of the checkpoint files the tensor is read from,
/// in the order the layout names them: the same bytes read from another file
/// are another weight. `layout` is the name of the layout the reading operator
/// declares (the `nn` layer's layout), and `region` the part of the logical
/// tensor this owner holds.
pub fn weight_key(
    files: &[FileIdentity],
    element: ElementType<'_>,
    layout: &str,
    region: &Region,
) -> Key {
    let mut digest = Tagged::new(WEIGHT_TAG);
    digest.number(files.len() as u64);
    for file in files {
        digest.text(file.as_str());
    }
    digest.text(&element.spell());
    digest.text(layout);
    region.tag(&mut digest);
    digest.finish()
}

/// The hash of a token prefix, as the cache's block tables compute it over the
/// request's tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenPrefix(pub [u8; 32]);

/// The key of one KV page, or of one recursive-state checkpoint, of a token
/// prefix.
pub fn kv_page_key(
    prefix: TokenPrefix,
    representation: &Representation<'_>,
    split: Split,
    region: &Region,
) -> Key {
    let mut digest = Tagged::new(KV_TAG);
    digest.part(&prefix.0);
    digest.text(&representation.element.spell());
    digest.number(u64::from(representation.kv.page_tokens));
    digest.text(representation.kv.element.name());
    digest.number(u64::from(split.tensor_parallel));
    digest.number(u64::from(split.ulysses));
    region.tag(&mut digest);
    digest.finish()
}

/// The hash of input media (an image or audio track a block encoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MediaHash(pub [u8; 32]);

/// The key of an encoder's output for one input.
pub fn encoder_key(media: MediaHash, representation: &Representation<'_>) -> Key {
    let mut digest = Tagged::new(ENCODER_TAG);
    digest.part(&media.0);
    digest.text(&representation.element.spell());
    digest.finish()
}

/// A request's trajectory: the conditioning a latent trajectory was drawn
/// from, identified by the hash of its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trajectory(pub [u8; 32]);

/// The key of a latent trajectory's pages after `steps` denoising steps.
pub fn latent_key(trajectory: Trajectory, steps: u32, representation: &Representation<'_>) -> Key {
    let mut digest = Tagged::new(LATENT_TAG);
    digest.part(&trajectory.0);
    digest.number(u64::from(steps));
    digest.text(&representation.element.spell());
    digest.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn representation() -> Representation<'static> {
        Representation {
            element: ElementType {
                dtype: DType::BF16,
                quantization: None,
            },
            kv: KvLayout {
                page_tokens: 64,
                element: DType::BF16,
            },
        }
    }

    fn blob_hash(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    #[test]
    fn a_weight_key_names_what_the_stored_bytes_mean() {
        let file = FileIdentity("blob:abc".into());
        let key = weight_key(
            std::slice::from_ref(&file),
            representation().element,
            "row-parallel",
            &Region::new([0..128, 0..64]),
        );
        assert_eq!(
            key,
            weight_key(
                std::slice::from_ref(&file),
                representation().element,
                "row-parallel",
                &Region::new([0..128, 0..64]),
            )
        );
        // Another layout, region, element type or source file is other bytes.
        for other in [
            weight_key(
                std::slice::from_ref(&file),
                representation().element,
                "column-parallel",
                &Region::new([0..128, 0..64]),
            ),
            weight_key(
                std::slice::from_ref(&file),
                representation().element,
                "row-parallel",
                &Region::new([0..128, 0..32]),
            ),
            weight_key(
                std::slice::from_ref(&file),
                ElementType {
                    dtype: DType::F32,
                    quantization: None,
                },
                "row-parallel",
                &Region::new([0..128, 0..64]),
            ),
            weight_key(
                std::slice::from_ref(&file),
                ElementType {
                    dtype: DType::BF16,
                    quantization: Some("fp8"),
                },
                "row-parallel",
                &Region::new([0..128, 0..64]),
            ),
            weight_key(
                &[FileIdentity("blob:def".into())],
                representation().element,
                "row-parallel",
                &Region::new([0..128, 0..64]),
            ),
        ] {
            assert_ne!(key, other);
        }
    }

    #[test]
    fn a_whole_tensor_and_a_region_of_it_are_other_keys() {
        let file = FileIdentity("blob:abc".into());
        let element = representation().element;
        assert_ne!(
            weight_key(
                std::slice::from_ref(&file),
                element,
                "dense",
                &Region::whole()
            ),
            weight_key(
                &[file],
                element,
                "dense",
                &Region::new(std::iter::once(0..1))
            )
        );
    }

    #[test]
    fn derived_state_of_different_kinds_never_shares_a_key() {
        let hash = [7_u8; 32];
        let representation = representation();
        let region = Region::new(std::iter::once(0..4));
        assert_ne!(
            kv_page_key(
                TokenPrefix(hash),
                &representation,
                Split {
                    tensor_parallel: 1,
                    ulysses: 1
                },
                &region
            ),
            latent_key(Trajectory(hash), 4, &representation)
        );
        assert_ne!(
            encoder_key(MediaHash(hash), &representation),
            latent_key(Trajectory(hash), 0, &representation)
        );
    }

    #[test]
    fn a_kv_page_key_covers_the_token_prefix_representation_and_split() {
        let representation = representation();
        let split = Split {
            tensor_parallel: 8,
            ulysses: 2,
        };
        let region = Region::new(std::iter::once(0..64));
        let key = kv_page_key(TokenPrefix([1; 32]), &representation, split, &region);
        assert_ne!(
            key,
            kv_page_key(TokenPrefix([2; 32]), &representation, split, &region)
        );
        assert_ne!(
            key,
            kv_page_key(
                TokenPrefix([1; 32]),
                &Representation {
                    element: ElementType {
                        dtype: DType::BF16,
                        quantization: Some("modelopt_fp4"),
                    },
                    ..representation
                },
                split,
                &region
            )
        );
        assert_ne!(
            key,
            kv_page_key(
                TokenPrefix([1; 32]),
                &representation,
                Split {
                    tensor_parallel: 4,
                    ..split
                },
                &region
            )
        );
        // The KV layout is part of the build config: an FP8 page is not a
        // BF16 page of the same prefix.
        assert_ne!(
            key,
            kv_page_key(
                TokenPrefix([1; 32]),
                &Representation {
                    kv: KvLayout {
                        page_tokens: 64,
                        element: DType::F8E4M3,
                    },
                    ..representation
                },
                split,
                &region
            )
        );
    }

    #[test]
    fn a_hugging_face_cache_blob_is_named_by_its_contents() {
        let root = tempfile::tempdir().expect("a temporary directory");
        let blobs = root.path().join("blobs");
        std::fs::create_dir(&blobs).expect("the blobs directory");
        let name = blob_hash(0xab);
        let blob = blobs.join(&name);
        std::fs::write(&blob, b"weights").expect("the blob");
        assert_eq!(
            FileIdentity::of(&blob).expect("an identity"),
            FileIdentity(format!("blob:{name}"))
        );

        // A file outside a `blobs` directory is not a blob, whatever its
        // name.
        let named = root.path().join(&name);
        std::fs::write(&named, b"weights").expect("the named file");
        let identity = FileIdentity::of(&named).expect("an identity");
        assert!(!identity.as_str().starts_with("blob:"));
    }

    #[test]
    fn another_file_is_another_identity_and_a_rewrite_changes_it() {
        let root = tempfile::tempdir().expect("a temporary directory");
        let first = root.path().join("first.safetensors");
        let second = root.path().join("second.safetensors");
        std::fs::write(&first, b"one").expect("the first file");
        std::fs::write(&second, b"one").expect("the second file");
        // Both files are freshly written, so their identities are unique to
        // this read: two files never share one.
        let first_identity = FileIdentity::of(&first).expect("an identity");
        let second_identity = FileIdentity::of(&second).expect("an identity");
        assert_ne!(first_identity, second_identity);

        // A second read of the same settled file is the same identity.
        let mut settled = root.path().join("settled.safetensors");
        std::fs::write(&settled, b"one").expect("the file");
        settle(&settled);
        assert_eq!(
            FileIdentity::of(&settled).expect("an identity"),
            FileIdentity::of(&settled).expect("an identity")
        );

        // A rewrite changes the identity, so the old weights no longer match.
        let before = FileIdentity::of(&settled).expect("an identity");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&settled)
            .expect("the file");
        file.write_all(b"more").expect("the rewrite");
        drop(file);
        settle(&settled);
        assert_ne!(before, FileIdentity::of(&settled).expect("an identity"));
        settled.push("nothing");
    }

    /// Make `path` old enough for its identity to name its contents: the
    /// loader treats a file's timestamps as unsettled for one second.
    fn settle(path: &Path) {
        let status = std::fs::metadata(path).expect("the file");
        let earlier = std::time::Duration::from_secs(2);
        let file = std::fs::File::options()
            .write(true)
            .open(path)
            .expect("the file");
        file.set_modified(status.modified().expect("a modification time") - earlier)
            .expect("the modification time");
    }
}
