//! Weight adapters: a file of edits to a checkpoint's tensors, fused into the checkpoint's bytes
//! before any weight is converted or published.
//!
//! ## Formats
//!
//! An adapter is one safetensors file in a container [`Format`] that its owner declares: the
//! model family that fuses it states the file an adapter directory holds it in, which
//! checkpoint parameter each tensor name edits and in which [`Role`], and how the writer's header
//! metadata states the file's completeness. This module names no container; it implements the
//! reconstruction every format here encodes:
//!
//! | Role | Extents | Contribution to the parameter it edits |
//! | --- | --- | --- |
//! | [`Role::Down`], [`Role::Up`] | `[rank, in]`, `[out, rank]` | `up @ down`, unscaled |
//! | [`Role::Diff`] | the parameter's | added as it is |
//! | [`Role::Assign`] | the parameter's | the parameter, which the base does not hold |
//!
//! The low-rank factors carry no `alpha`: the product is added directly, a scale of exactly one;
//! a format whose writer scales its factors states the scaled factors. A file is checked against
//! its writer's statement before it is fused ([`Format::check`]), because a truncated file is
//! otherwise indistinguishable from a small one and would serve mostly base weights.
//!
//! ## Fusion
//!
//! An adapter is fused into the deployment's weights, once, at the loading boundary
//! ([`crate::Checkpoint::attach`]): every byte a weight load or a host read takes of an edited
//! tensor is the fused value, so a module, a layout conversion and every computed table read the
//! adapted parameter without knowing an adapter exists. For one element of an edited parameter
//!
//! ```text
//! fused = round((up @ down + diff) + base)
//! ```
//!
//! evaluated in FP32 from the stored values and rounded once, to nearest-even, into the base
//! tensor's own element type: the FP32 delta, added to the widened base and cast back to the
//! parameter's type. The fused bytes are the checkpoint's bytes from then on: a weight the model
//! pins at another type converts them as it would any stored value.
//!
//! A tensor the adapter does not edit is read from its shard exactly as without an adapter. A
//! fused weight's retained key names both files, the base shard's and the adapter's, so it never
//! shares retained storage with the base weight or another adapter's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use tensor::DType;

use crate::checkpoint::{Checkpoint, FileIdentity, Source, TensorLocation};
use crate::error::{Error, Result};
use crate::weights::{dim, last, numel};

/// How an adapter tensor enters the parameter it edits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// The down projection of a low-rank pair, `[rank, in]`.
    Down,
    /// The up projection of a low-rank pair, `[out, rank]`.
    Up,
    /// A full-rank delta of the parameter, its extents.
    Diff,
    /// A parameter the base does not hold, assigned as it is.
    Assign,
}

/// What an adapter file carries, as a [`Format`] checks it against its writer's statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contents {
    /// Low-rank factor tensors, down and up projections together.
    pub low_rank: u64,
    /// Full-rank delta tensors.
    pub diffs: u64,
    /// Assigned parameters.
    pub assigned: u64,
    /// The rank of every low-rank pair, each once.
    pub ranks: BTreeSet<u64>,
}

/// An adapter container: the file an adapter directory holds it in, the parameter each tensor
/// name edits, and the writer's statement of what the file carries. Its owner, the model family
/// that fuses it, declares it.
pub trait Format: Sync {
    /// The file of an adapter directory that holds the adapter.
    fn file(&self) -> &'static str;

    /// The checkpoint parameter the adapter tensor `name` edits and its role, or `None` for a
    /// name the format gives no role.
    fn role(&self, name: &str) -> Option<(String, Role)>;

    /// Checks the writer's header `metadata` against what the file carries.
    ///
    /// # Errors
    ///
    /// The problem, when the metadata states another container or other contents.
    fn check(
        &self,
        metadata: &BTreeMap<String, String>,
        contents: &Contents,
    ) -> std::result::Result<(), String>;
}

/// The adapter tensors that edit one parameter, by their names in the adapter's file.
#[derive(Clone, Default, Debug)]
struct Edit {
    /// The down projection, `[rank, in]`.
    down: Option<String>,
    /// The up projection, `[out, rank]`.
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
    /// Open the adapter the directory `directory` holds in `format`: the format's file there.
    ///
    /// # Errors
    ///
    /// Fails when the directory does not hold the format's file, when the file names a tensor the
    /// format has no role for, edits one parameter twice in one role, pairs a low-rank factor
    /// with nothing or with a factor of other extents, or carries what its metadata does not
    /// state ([`Format::check`]).
    pub fn open(directory: &Source, format: &dyn Format) -> Result<Self> {
        let file = directory.directory().join(format.file());
        if !file.is_file() {
            return Err(Error::Checkpoint {
                path: directory.directory().to_path_buf(),
                problem: format!(
                    "an adapter directory holds its adapter as {}",
                    format.file()
                ),
            });
        }
        let shards = Checkpoint::from_shards(directory, vec![file.clone()])?;
        let refuse = |problem: String| Error::Checkpoint {
            path: file.clone(),
            problem,
        };

        let mut edits: BTreeMap<String, Edit> = BTreeMap::new();
        let mut assigned = BTreeMap::new();
        let mut low_rank = 0_u64;
        let mut diffs = 0_u64;
        for name in shards.names() {
            let (target, role) = format.role(name).ok_or_else(|| {
                refuse(format!(
                    "tensor `{name}` has no role in the adapter's format"
                ))
            })?;
            let slot = match role {
                Role::Assign => {
                    if assigned.insert(target, name.to_owned()).is_some() {
                        return Err(refuse(format!(
                            "two tensors assign the same parameter as `{name}`"
                        )));
                    }
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
                Role::Diff => {
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

        let mut ranks = BTreeSet::new();
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
            ranks.insert(dim(&down.shape, 0));
        }

        // The writer's statement of what it emitted is the one statement in the file about its
        // own completeness, so a claimed adapter has to carry what it states.
        let contents = Contents {
            low_rank,
            diffs,
            assigned: assigned.len() as u64,
            ranks,
        };
        format
            .check(shards.files()[0].metadata(), &contents)
            .map_err(refuse)?;

        Ok(Self {
            path: file,
            shards,
            edits,
            assigned,
        })
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
            edits: fused,
            assigned: placed,
        })
    }
}

/// What one edited parameter adds to its base: locations in the checkpoint's own file list.
#[derive(Clone, Debug)]
pub(crate) struct Delta {
    /// The fused parameter's identity: the base shard's and the adapter's together.
    pub(crate) identity: FileIdentity,
    /// `(up [out, rank], down [rank, in])`.
    low_rank: Option<(TensorLocation, TensorLocation)>,
    /// The full-rank delta.
    diff: Option<TensorLocation>,
}

/// An adapter fused into a checkpoint: what it changed.
#[derive(Clone, Debug)]
pub struct Fusion {
    edits: BTreeMap<String, Delta>,
    assigned: BTreeSet<String>,
}

impl Fusion {
    /// Whether the adapter edits the checkpoint tensor `name`.
    pub fn edits(&self, name: &str) -> bool {
        self.edits.contains_key(name)
    }

    /// Whether the adapter assigned the checkpoint tensor `name`, which the base does not hold.
    pub fn assigns(&self, name: &str) -> bool {
        self.assigned.contains(name)
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
/// (module documentation). The rows are split over the host's cores: a low-rank product costs
/// `2 · rank` operations an element, which for a projection of hundreds of millions of elements
/// is tens of GFLOP.
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

    // The factors are read whole, once per call: the down projection is `rank × in` and the up
    // projection `out × rank`, both a small fraction of the parameter.
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
                // `Σ_r up[row, r] · down[r, column]`, accumulated in rank order: one axpy of a row
                // of the down projection per rank, which vectorizes over the columns.
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
