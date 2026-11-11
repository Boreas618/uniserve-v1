//! Weight adapters: a file of edits to a checkpoint's tensors, fused into the checkpoint's bytes
//! before any weight is converted or published.
//!
//! ## The format
//!
//! An adapter is one safetensors file whose header metadata states `format = fastvideo-lora-v2`,
//! FastVideo's adapter container, and the reconstruction it encodes:
//!
//! ```text
//! W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned
//! ```
//!
//! Each tensor's name is the edited module's path in the checkpoint's own namespace with one of
//! five suffixes, and the suffix names the parameter it edits and how:
//!
//! | Adapter tensor | Edits | Contribution |
//! | --- | --- | --- |
//! | `<module>.lora_A.weight` `[rank, in]`, `<module>.lora_B.weight` `[out, rank]` | `<module>.weight` | `lora_B @ lora_A`, unscaled |
//! | `<module>.diff` | `<module>.weight` | added as it is |
//! | `<module>.diff_b` | `<module>.bias` | added as it is |
//! | `<module>.set_weight` | `<module>.weight` | the parameter, which the base does not hold |
//!
//! The low-rank factors carry no `alpha`: the format adds the product directly, a scale of exactly
//! one. The metadata states how many tensors of each kind the writer emitted (`low_rank_tensors`,
//! `diff_tensors`, `set_weight_tensors`) and the factors' `rank`; an adapter that disagrees with
//! its own statement is refused, because a truncated file is otherwise indistinguishable from a
//! small one and would serve mostly base weights.
//!
//! ## Fusion
//!
//! An adapter is fused into the deployment's weights, once, at the loading boundary
//! ([`crate::Checkpoint::attach`]): every byte a weight load or a host read takes of an edited
//! tensor is the fused value, so a module, a layout conversion and every computed table read the
//! adapted parameter without knowing an adapter exists. For one element of an edited parameter
//!
//! ```text
//! fused = round((lora_B @ lora_A + diff) + base)
//! ```
//!
//! evaluated in FP32 from the stored values and rounded once, to nearest-even, into the base
//! tensor's own element type, which is the reference implementation's arithmetic (vLLM-Omni's
//! FastH3 fusion builds the FP32 delta, adds the widened base and casts back to the parameter's
//! type). The fused bytes are the checkpoint's bytes from then on: a weight the model pins at
//! another type converts them as it would any stored value.
//!
//! A tensor the adapter does not edit is read from its shard exactly as without an adapter. A
//! fused weight's retained key names both files, the base shard's and the adapter's, so it never
//! shares retained storage with the base weight or another adapter's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use uniserve_tensor::DType;

use crate::checkpoint::{Checkpoint, FileIdentity, TensorLocation};
use crate::error::{Error, Result};
use crate::weights::{dim, last, numel};

/// The container format this build reads.
pub const FASTVIDEO_FORMAT: &str = "fastvideo-lora-v2";

/// The file a directory names its adapter by, the PEFT and FastVideo convention.
const ADAPTER_FILE: &str = "adapter_model.safetensors";

/// How an adapter tensor enters the parameter it edits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    /// `lora_A`: the down projection `[rank, in]`.
    Down,
    /// `lora_B`: the up projection `[out, rank]`.
    Up,
    /// `diff`: a full-rank delta of the weight.
    Diff,
    /// `diff_b`: a delta of the bias.
    DiffBias,
    /// `set_weight`: a parameter the base does not hold.
    Assign,
}

impl Role {
    /// The suffixes of the format, each with the parameter leaf it edits.
    const SUFFIXES: [(&'static str, Self, &'static str); 5] = [
        (".lora_A.weight", Self::Down, "weight"),
        (".lora_B.weight", Self::Up, "weight"),
        (".diff_b", Self::DiffBias, "bias"),
        (".diff", Self::Diff, "weight"),
        (".set_weight", Self::Assign, "weight"),
    ];

    /// Split an adapter tensor's name into the parameter it edits and its role.
    fn of(name: &str) -> Option<(String, Self)> {
        Self::SUFFIXES.iter().find_map(|(suffix, role, leaf)| {
            name.strip_suffix(suffix)
                .map(|module| (format!("{module}.{leaf}"), *role))
        })
    }
}

/// The adapter tensors that edit one parameter, by their names in the adapter's file.
#[derive(Clone, Default, Debug)]
struct Edit {
    /// `lora_A`, `[rank, in]`.
    down: Option<String>,
    /// `lora_B`, `[out, rank]`.
    up: Option<String>,
    /// The full-rank delta, the parameter's extents.
    diff: Option<String>,
}

/// An adapter file, read and checked against its own statement but not yet fused.
pub struct Adapter {
    /// The adapter's file.
    path: PathBuf,
    /// The adapter's one shard.
    shards: Checkpoint,
    /// Each edited parameter, by its name in the checkpoint, with the tensors that edit it.
    edits: BTreeMap<String, Edit>,
    /// Each parameter the adapter assigns, by its name in the checkpoint, with the adapter tensor
    /// that holds it.
    assigned: BTreeMap<String, String>,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Adapter")
            .field("path", &self.path)
            .field("edits", &self.edits.len())
            .field("assigned", &self.assigned.len())
            .finish()
    }
}

impl Adapter {
    /// Open the adapter at `path`: a `.safetensors` file, or a directory holding
    /// `adapter_model.safetensors` or exactly one `.safetensors` file.
    ///
    /// # Errors
    ///
    /// Fails when no single adapter file is found, when the file states another format than
    /// [`FASTVIDEO_FORMAT`], names a tensor the format has no role for, pairs a low-rank factor
    /// with nothing, or carries other tensor counts or another rank than its metadata states.
    pub fn open(path: &Path) -> Result<Self> {
        let file = adapter_file(path)?;
        let root = file.parent().unwrap_or(Path::new(".")).to_path_buf();
        let shards = Checkpoint::from_shards(&root, vec![file.clone()])?;
        let refuse = |problem: String| Error::Checkpoint {
            path: file.clone(),
            problem,
        };
        let metadata = shards.files()[0].metadata().clone();
        match metadata.get("format").map(String::as_str) {
            Some(FASTVIDEO_FORMAT) => {}
            stated => {
                return Err(refuse(format!(
                    "the adapter states the format {stated:?}; this build fuses {FASTVIDEO_FORMAT} \
                     adapters, whose reconstruction it implements"
                )));
            }
        }

        let mut edits: BTreeMap<String, Edit> = BTreeMap::new();
        let mut assigned = BTreeMap::new();
        let mut low_rank = 0_u64;
        let mut diffs = 0_u64;
        for name in shards.names() {
            let (target, role) = Role::of(name).ok_or_else(|| {
                refuse(format!(
                    "tensor `{name}` has none of the format's suffixes (lora_A.weight, \
                     lora_B.weight, diff, diff_b, set_weight)"
                ))
            })?;
            let slot = match role {
                Role::Assign => {
                    assigned.insert(target, name.to_owned());
                    continue;
                }
                Role::Down => {
                    low_rank += 1;
                    &mut edits.entry(target).or_default().down
                }
                Role::Up => {
                    low_rank += 1;
                    &mut edits.entry(target).or_default().up
                }
                Role::Diff | Role::DiffBias => {
                    diffs += 1;
                    &mut edits.entry(target).or_default().diff
                }
            };
            if slot.replace(name.to_owned()).is_some() {
                return Err(refuse(format!(
                    "two tensors edit the same parameter in the same role as `{name}`"
                )));
            }
        }

        // The writer's statement of what it emitted is the one statement in the file about its
        // own completeness, so a claimed adapter has to carry each count it states.
        for (key, counted) in [
            ("low_rank_tensors", low_rank),
            ("diff_tensors", diffs),
            ("set_weight_tensors", assigned.len() as u64),
        ] {
            let stated = metadata
                .get(key)
                .ok_or_else(|| refuse(format!("the adapter does not state `{key}`")))?;
            let stated: u64 = stated
                .parse()
                .map_err(|_| refuse(format!("`{key}` is {stated:?}, not a count")))?;
            if stated != counted {
                return Err(refuse(format!(
                    "the adapter states {key} = {stated} and carries {counted}: it is incomplete"
                )));
            }
        }

        let rank = metadata
            .get("rank")
            .map(|rank| {
                rank.parse::<u64>()
                    .map_err(|_| refuse(format!("`rank` is {rank:?}, not a count")))
            })
            .transpose()?;
        for (target, edit) in &edits {
            let (down, up) = match (&edit.down, &edit.up) {
                (Some(down), Some(up)) => (down, up),
                (None, None) => continue,
                _ => {
                    return Err(refuse(format!(
                        "`{target}` has one low-rank factor without the other"
                    )));
                }
            };
            let down = shards.locate(down)?;
            let up = shards.locate(up)?;
            if down.shape.rank() != 2
                || up.shape.rank() != 2
                || dim(&down.shape, 0) != dim(&up.shape, 1)
            {
                return Err(refuse(format!(
                    "`{target}`'s factors are {:?} and {:?}, not [rank, in] and [out, rank]",
                    down.shape.dims(),
                    up.shape.dims()
                )));
            }
            if let Some(rank) = rank
                && dim(&down.shape, 0) != rank
            {
                return Err(refuse(format!(
                    "`{target}`'s factors have rank {}, and the adapter states rank {rank}",
                    dim(&down.shape, 0)
                )));
            }
        }

        Ok(Self {
            path: file,
            shards,
            edits,
            assigned,
        })
    }

    /// The adapter's file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The header metadata the adapter's writer stated: its format, rank, tensor counts and the
    /// models it was extracted from.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        self.shards.files()[0].metadata()
    }

    /// The checkpoint parameters the adapter edits, in name order.
    pub fn edited(&self) -> impl Iterator<Item = &str> {
        self.edits.keys().map(String::as_str)
    }

    /// The checkpoint parameters the adapter assigns, which the base does not hold, in name order.
    pub fn assigned(&self) -> impl Iterator<Item = &str> {
        self.assigned.keys().map(String::as_str)
    }

    /// Fuse this adapter into `checkpoint`, a flat shard set: the adapter's file joins the
    /// checkpoint's files, each assigned parameter becomes one of its tensors, and each edited one
    /// reads fused from then on.
    pub(crate) fn fuse_into(self, checkpoint: &mut Checkpoint) -> Result<Fusion> {
        let Self {
            path,
            shards,
            edits,
            assigned,
        } = self;
        let refuse = |problem: String| Error::Checkpoint {
            path: path.clone(),
            problem,
        };
        let metadata = shards.files()[0].metadata().clone();
        let identity = shards.files()[0].identity().clone();
        let (files, mut tensors) = shards.into_shards();
        let rebase = checkpoint.adopt(files);
        for location in tensors.values_mut() {
            location.file += rebase;
        }
        let take = |name: &str| -> TensorLocation { tensors[name].clone() };

        let mut fused = BTreeMap::new();
        for (target, edit) in &edits {
            let base = checkpoint.tensor(target).cloned().ok_or_else(|| {
                refuse(format!(
                    "the adapter edits `{target}`, which the checkpoint at {} does not hold",
                    checkpoint.root().display()
                ))
            })?;
            if !matches!(base.dtype, DType::BF16 | DType::F16 | DType::F32) {
                return Err(refuse(format!(
                    "the adapter edits `{target}`, which the checkpoint stores as {}; only a \
                     floating-point parameter can take a delta",
                    ops::abi::name(base.dtype)
                )));
            }
            let low_rank = match (&edit.up, &edit.down) {
                (Some(up), Some(down)) => {
                    let (up, down) = (take(up), take(down));
                    let expected = [dim(&up.shape, 0) as usize, dim(&down.shape, 1) as usize];
                    if base.shape.dims() != expected {
                        return Err(refuse(format!(
                            "`{target}` is {:?} and its factors make {expected:?}",
                            base.shape.dims()
                        )));
                    }
                    Some((up, down))
                }
                _ => None,
            };
            let diff = edit.diff.as_deref().map(take);
            if let Some(diff) = &diff
                && diff.shape.dims() != base.shape.dims()
            {
                return Err(refuse(format!(
                    "`{target}` is {:?} and its delta {:?}",
                    base.shape.dims(),
                    diff.shape.dims()
                )));
            }
            for part in low_rank
                .iter()
                .flat_map(|(up, down)| [up, down])
                .chain(&diff)
            {
                if !matches!(part.dtype, DType::BF16 | DType::F16 | DType::F32) {
                    return Err(refuse(format!(
                        "`{}` is {}, not a floating-point delta",
                        part.tensor,
                        ops::abi::name(part.dtype)
                    )));
                }
            }
            let base_identity = checkpoint.files()[base.file].identity();
            fused.insert(
                target.clone(),
                Delta {
                    identity: FileIdentity::joined(base_identity, &identity),
                    low_rank,
                    diff,
                },
            );
        }

        let mut placed = BTreeSet::new();
        for (target, name) in &assigned {
            if checkpoint.tensor(target).is_some() {
                return Err(refuse(format!(
                    "the adapter assigns `{target}`, which the checkpoint already holds: \
                     assigning it would discard the checkpoint's own parameter"
                )));
            }
            let mut location = take(name);
            location.tensor = target.clone();
            checkpoint.place(location);
            placed.insert(target.clone());
        }

        Ok(Fusion {
            path,
            metadata,
            edits: fused,
            assigned: placed,
        })
    }
}

/// The adapter file `path` names.
fn adapter_file(path: &Path) -> Result<PathBuf> {
    let refuse = |problem: &str| Error::Checkpoint {
        path: path.to_path_buf(),
        problem: problem.to_owned(),
    };
    if path.is_file() {
        return match path.extension() {
            Some(extension) if extension == "safetensors" => Ok(path.to_path_buf()),
            _ => Err(refuse("an adapter file is a .safetensors file")),
        };
    }
    if !path.is_dir() {
        return Err(refuse("no adapter file or directory is here"));
    }
    let named = path.join(ADAPTER_FILE);
    if named.is_file() {
        return Ok(named);
    }
    let entries = std::fs::read_dir(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            files.push(entry.path());
        }
    }
    match files.as_slice() {
        [file] => Ok(file.clone()),
        [] => Err(refuse("the directory holds no .safetensors adapter")),
        _ => Err(refuse(
            "the directory holds several .safetensors files and no adapter_model.safetensors; \
             name the adapter file",
        )),
    }
}

/// What one edited parameter adds to its base: locations in the checkpoint's own file list.
#[derive(Clone, Debug)]
pub(crate) struct Delta {
    /// The fused parameter's identity: the base shard's and the adapter's together.
    pub(crate) identity: FileIdentity,
    /// `(lora_B [out, rank], lora_A [rank, in])`.
    low_rank: Option<(TensorLocation, TensorLocation)>,
    /// The full-rank delta.
    diff: Option<TensorLocation>,
}

/// An adapter fused into a checkpoint: what it states and what it changed.
#[derive(Clone, Debug)]
pub struct Fusion {
    path: PathBuf,
    metadata: BTreeMap<String, String>,
    edits: BTreeMap<String, Delta>,
    assigned: BTreeSet<String>,
}

impl Fusion {
    /// The adapter's file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The header metadata the adapter's writer stated.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Whether the adapter edits the checkpoint tensor `name`.
    pub fn edits(&self, name: &str) -> bool {
        self.edits.contains_key(name)
    }

    /// Whether the adapter assigned the checkpoint tensor `name`, which the base does not hold.
    pub fn assigns(&self, name: &str) -> bool {
        self.assigned.contains(name)
    }

    /// The parameters the adapter edits, in name order.
    pub fn edited(&self) -> impl Iterator<Item = &str> {
        self.edits.keys().map(String::as_str)
    }

    /// The parameters the adapter assigned, in name order.
    pub fn assigned(&self) -> impl Iterator<Item = &str> {
        self.assigned.iter().map(String::as_str)
    }

    /// The delta of the edited tensor `name`.
    pub(crate) fn delta(&self, name: &str) -> Option<&Delta> {
        self.edits.get(name)
    }
}

/// One run of elements of a parameter: row `row`, columns `first..last`, written at element
/// `output` of the result.
#[derive(Clone, Copy)]
struct Run {
    row: u64,
    first: u64,
    last: u64,
    output: usize,
}

/// The fused bytes of `base`'s byte ranges `ranges`, concatenated in order.
///
/// `ranges` are byte offsets within the tensor's own bytes, each a whole number of elements. The
/// product is evaluated per element in FP32, in rank order, and rounded once into `base`'s type
/// (module documentation). The rows are split over the host's cores: a whole H3 modulation
/// projection is `96768 × 2688` elements of rank 64, 33 GFLOP.
pub(crate) fn fused(
    checkpoint: &Checkpoint,
    base: &TensorLocation,
    delta: &Delta,
    ranges: &[(usize, usize)],
) -> Result<Vec<u8>> {
    let dtype = base.dtype;
    let element = ops::abi::bytes(dtype).unwrap_or(4) as usize;
    let columns = if base.shape.rank() == 0 {
        1
    } else {
        last(&base.shape)
    };
    let decode = |location: &TensorLocation, from: usize, to: usize| -> Result<Vec<f32>> {
        let bytes = checkpoint.bytes(location)?;
        let width = ops::abi::bytes(location.dtype).unwrap_or(4) as usize;
        let range = bytes.get(from * width..to * width).ok_or_else(|| {
            Error::weight(
                &location.tensor,
                format!("elements {from}..{to} reach past the tensor"),
            )
        })?;
        Ok(ops::values::decode(location.dtype, range)?)
    };

    // The rows each range covers, cut at row boundaries: a range of a row-major tensor is a run
    // of whole rows between two partial ones.
    let mut runs = Vec::new();
    let mut elements = 0_usize;
    for &(from, to) in ranges {
        if from % element != 0 || to % element != 0 || to < from {
            return Err(Error::weight(
                &base.tensor,
                format!(
                    "bytes {from}..{to} are not whole {dtype} elements",
                    dtype = ops::abi::name(dtype)
                ),
            ));
        }
        let mut at = (from / element) as u64;
        let end = (to / element) as u64;
        while at < end {
            let row = at / columns;
            let first = at % columns;
            let last = columns.min(first + (end - at));
            runs.push(Run {
                row,
                first,
                last,
                output: elements,
            });
            elements += (last - first) as usize;
            at += last - first;
        }
    }

    // The factors are read whole, once per call: `lora_A` is `rank × in` and `lora_B` `out ×
    // rank`, both a small fraction of the parameter.
    let factors = match &delta.low_rank {
        Some((up, down)) => {
            let rank = dim(&down.shape, 0) as usize;
            Some((
                decode(up, 0, numel(&up.shape) as usize)?,
                decode(down, 0, numel(&down.shape) as usize)?,
                rank,
            ))
        }
        None => None,
    };

    let mut values = vec![0_f32; elements];
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(runs.len().max(1));
    let per_thread = elements.div_ceil(threads.max(1)).max(1);
    // Runs are handed out in order until a worker holds its share, so each worker writes one
    // contiguous slice of the result.
    let mut batches: Vec<(&[Run], &mut [f32])> = Vec::with_capacity(threads);
    let mut rest_runs: &[Run] = &runs;
    let mut rest_values: &mut [f32] = &mut values;
    while !rest_runs.is_empty() {
        let mut taken = 0;
        let mut count = 0;
        while count < rest_runs.len() && (taken < per_thread || count == 0) {
            taken += (rest_runs[count].last - rest_runs[count].first) as usize;
            count += 1;
        }
        let (runs_now, runs_later) = rest_runs.split_at(count);
        let (values_now, values_later) = std::mem::take(&mut rest_values).split_at_mut(taken);
        batches.push((runs_now, values_now));
        rest_runs = runs_later;
        rest_values = values_later;
    }

    let compute = |runs: &[Run], out: &mut [f32]| -> Result<()> {
        let offset = runs.first().map_or(0, |run| run.output);
        for run in runs {
            let width = (run.last - run.first) as usize;
            let start = (run.row * columns + run.first) as usize;
            let slot = &mut out[run.output - offset..][..width];
            if let Some((up, down, rank)) = &factors {
                let in_features = columns as usize;
                let up_row = &up[run.row as usize * rank..][..*rank];
                // `Σ_r B[row, r] · A[r, column]`, accumulated in rank order: one axpy of a row
                // of `lora_A` per rank, which vectorizes over the columns.
                for (r, scale) in up_row.iter().enumerate() {
                    let down_row = &down[r * in_features + run.first as usize..][..width];
                    for (value, factor) in slot.iter_mut().zip(down_row) {
                        *value += scale * factor;
                    }
                }
            }
            if let Some(diff) = &delta.diff {
                for (value, diff) in slot.iter_mut().zip(decode(diff, start, start + width)?) {
                    *value += diff;
                }
            }
            for (value, base) in slot.iter_mut().zip(decode(base, start, start + width)?) {
                *value += base;
            }
        }
        Ok(())
    };
    std::thread::scope(|scope| -> Result<()> {
        let workers = batches
            .into_iter()
            .map(|(runs, out)| scope.spawn(move || compute(runs, out)))
            .collect::<Vec<_>>();
        for worker in workers {
            worker
                .join()
                .map_err(|_| Error::weight(&base.tensor, "a fusion worker panicked"))??;
        }
        Ok(())
    })?;
    Ok(ops::values::encode(dtype, &values)?)
}
