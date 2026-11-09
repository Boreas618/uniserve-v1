//! The recorded walk: a runner's forwards run on a recording [`nn::Forward`] over a counting arena,
//! so the programs they launch and the arena they reserve are observations of the forwards
//! themselves rather than a second statement of them.
//!
//! A runner walks, for every bucket it captures, the same model calls its capture walk makes and
//! the forwards it launches eagerly on the serving path (a request's opening, a clip's splice, a
//! model's constants). Each call reports its `(operator, key)` to the capture sink installed here
//! and launches nothing; each reservation lands in an unbounded [`nn::Arena::counting`] whose
//! [`nn::Arena::peak`] is the extent the same forward reaches in the stream's real region, because
//! reservations follow shapes and element types alone. The worker claims each stream's region from
//! the peak and the engine installs exactly the recorded keys.
//!
//! Nothing a recorded walk does reaches the device: no copy, no launch, no collective. Views of
//! memory the walk does not reserve in the stream's arena (state tables, staged regions) are
//! [`Walk::placeholder`]s at synthetic addresses that nothing reads.

use std::collections::{BTreeMap, BTreeSet};

use nn::capability::StateTable;
use ops::{DType, Shape, TensorView};

use crate::plan::Bucket;
use crate::{Error, Result};

/// What a runner's recorded walk observed: every program its forwards launch and the extent they
/// reserve in the stream's arena region.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Every descriptor key the walked forwards launched, by operator name (`ops::Op::NAME`).
    pub keys: BTreeMap<String, BTreeSet<String>>,
    /// The furthest byte any walked forward reserved in the stream's arena region; zero when no
    /// forward reserved anything.
    pub peak: u64,
    /// The bucket whose walk reached [`Self::peak`], or `None` when a forward that belongs to no
    /// bucket (a model's constants, filled once per load) reached it or nothing was reserved.
    pub widest: Option<Bucket>,
}

impl Footprint {
    /// Walks `forwards` under a capture sink and folds what they launched and reserved into this
    /// footprint, attributing the arena's peak to `bucket`.
    ///
    /// `forwards` receives a [`Walk`] whose [`Walk::arena`] stands for the stream's region: every
    /// forward that starts at the region's first address (a fresh `nn::Arena` over the stream's
    /// base) starts with [`nn::Arena::reset`] here, so the walk's peak is the furthest offset any
    /// of them reached. Its forwards must be recording ones ([`Walk::forward`]): a launching
    /// forward would launch.
    ///
    /// # Errors
    ///
    /// What `forwards` answers, or a refusal when a capture sink is already installed on this
    /// thread: a recorded walk inside a capture would report its keys to the capture's recorder.
    pub fn walk(
        &mut self,
        bucket: Option<Bucket>,
        forwards: impl FnOnce(&Walk) -> Result<()>,
    ) -> Result<()> {
        if ops::capture::installed() {
            return Err(Error::Invariant(
                "a recorded walk began while a capture sink is installed on this thread".into(),
            ));
        }
        let walk = Walk {
            arena: nn::Arena::counting(),
            placeholders: nn::Arena::counting(),
        };
        let mut launches = Launches::default();
        let answer = {
            // The sink lives in `launches`, which outlives the guard: the guard removes it on every
            // exit from this scope, an error or a panic included.
            ops::capture::set_sink(Some(&mut launches));
            let _installed = Installed;
            forwards(&walk)
        };
        answer?;
        for (operator, keys) in launches.0 {
            self.keys.entry(operator).or_default().extend(keys);
        }
        let peak = walk.arena.peak();
        if peak > self.peak {
            self.peak = peak;
            self.widest = bucket;
        }
        Ok(())
    }

    /// Adds `other`'s keys to this footprint's: the set a device's runners launch together.
    pub fn union(&mut self, other: &Self) {
        for (operator, keys) in &other.keys {
            self.keys
                .entry(operator.clone())
                .or_default()
                .extend(keys.iter().cloned());
        }
    }
}

/// The resources of one recorded walk.
pub struct Walk {
    /// The stream's arena region, counted: unbounded, at synthetic addresses.
    arena: nn::Arena,
    /// Where the walk's placeholders live: counted apart from [`Self::arena`], because the memory
    /// they stand for is not the stream's arena region.
    placeholders: nn::Arena,
}

impl Walk {
    /// The counting arena that stands for the stream's region.
    pub fn arena(&self) -> &nn::Arena {
        &self.arena
    }

    /// A recording forward over the stream's region, split as `group`'s rank when the runner's
    /// forwards carry one: every layer answers the shapes that rank launches with, and every
    /// collective checks its views and issues nothing.
    ///
    /// # Errors
    ///
    /// When the group's rank is not a member of it.
    pub fn forward(&self, group: Option<&nn::Group>) -> Result<nn::Forward<'_>> {
        let forward = nn::Forward::recording(&self.arena);
        let Some(group) = group else {
            return Ok(forward);
        };
        // NCCL reports a member's rank as a non-negative `int`, as `nn::Forward::split` reads it.
        let rank = u64::try_from(group.rank()).unwrap_or(0);
        Ok(forward.with_split(rank, group.ranks())?)
    }

    /// A view of `shape` elements of `dtype` that stands for memory outside the stream's arena: a
    /// state table, a staged column, a product region. Nothing reads or writes it.
    ///
    /// # Errors
    ///
    /// When `dtype` has no whole-byte size or the extent overflows.
    pub fn placeholder(&self, dtype: DType, shape: Shape) -> Result<TensorView> {
        Ok(self.placeholders.tensor(dtype, shape)?)
    }

    /// One placeholder per table, in the tables' order: what `Tables::views` answers once the
    /// tables are filled.
    ///
    /// # Errors
    ///
    /// As [`Self::placeholder`].
    pub fn tables(&self, tables: &[StateTable]) -> Result<Vec<TensorView>> {
        tables
            .iter()
            .map(|table| self.placeholder(table.dtype, table.shape))
            .collect()
    }
}

/// The capture sink of a recorded walk: every launch's operator and key.
#[derive(Default)]
struct Launches(BTreeMap<String, BTreeSet<String>>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0
            .entry(operator.to_owned())
            .or_default()
            .insert(key.to_owned());
    }
}

/// Removes this thread's capture sink when dropped.
struct Installed;

impl Drop for Installed {
    fn drop(&mut self) {
        ops::capture::set_sink(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ops::Op;
    use ops::logits::{GatherArgs, GatherDescriptor, LogitsGather};

    /// A gather of `columns` i32 columns: an operator whose descriptor a test can state in full.
    fn gather(columns: u32) -> GatherDescriptor {
        GatherDescriptor {
            dtype: DType::I32,
            columns,
            block: 128,
            pdl: false,
        }
    }

    /// One forward that reserves `bytes` and launches a gather of `columns`.
    fn forward(f: &nn::Forward<'_>, bytes: u64, columns: u32) -> Result<()> {
        let rows = f.arena().tensor(DType::I32, Shape::new(&[bytes / 4]))?;
        let mark = f.arena().mark();
        let scratch = f
            .arena()
            .tensor(DType::I32, Shape::new(&[8, u64::from(columns)]))?;
        f.operate::<LogitsGather>(
            &gather(columns),
            GatherArgs {
                hidden: scratch,
                // The gather reads the first eight row indices of the region's leading buffer, one
                // per scratch row.
                rows: nn::TensorView::new(rows.ptr, DType::I32, Shape::new(&[8])),
                out: scratch,
            },
        )?;
        // Scratch handed back still counts: the peak is what the region must map.
        f.arena().release(mark);
        Ok(())
    }

    fn key(columns: u32) -> String {
        ops::descriptor::key(&gather(columns))
    }

    /// A walk reports every program its forwards launched, by operator, and the furthest byte any
    /// forward reserved - including scratch a forward handed back - attributed to its bucket.
    #[test]
    fn a_walk_reports_its_programs_and_its_peak() {
        let bucket = Bucket::decode(4, crate::plan::Variant::DecodeGreedy);
        let mut footprint = Footprint::default();
        footprint
            .walk(Some(bucket), |walk| {
                let f = walk.forward(None)?;
                forward(&f, 1024, 16)?;
                // A second forward from the region's first address: an eager forward beside the
                // captured one, which shares the region and not the offsets.
                walk.arena().reset();
                forward(&f, 4096, 32)
            })
            .expect("a recorded walk launches nothing");

        let expected: BTreeSet<String> = [key(16), key(32)].into_iter().collect();
        assert_eq!(
            footprint.keys,
            BTreeMap::from([(LogitsGather::NAME.to_owned(), expected)])
        );
        // The wider forward: 4096 bytes of rows, then 8 rows of 32 i32 at the next 256-byte line.
        assert_eq!(footprint.peak, 4096 + 8 * 32 * 4);
        assert_eq!(footprint.widest, Some(bucket));
    }

    /// The peak of a device's stream is the widest walk's, and the bucket named is the one that set
    /// it; a narrower walk after it adds its keys and moves neither.
    #[test]
    fn the_widest_walk_sets_the_peak() {
        let wide = Bucket::prefill(1, 64, crate::plan::Variant::PrefillWrite);
        let narrow = Bucket::decode(1, crate::plan::Variant::DecodeGreedy);
        let mut footprint = Footprint::default();
        footprint
            .walk(Some(wide), |walk| forward(&walk.forward(None)?, 8192, 8))
            .unwrap();
        footprint
            .walk(Some(narrow), |walk| forward(&walk.forward(None)?, 256, 4))
            .unwrap();

        assert_eq!(footprint.widest, Some(wide));
        assert_eq!(footprint.peak, 8192 + 8 * 8 * 4);
        assert_eq!(footprint.keys[LogitsGather::NAME].len(), 2);
    }

    /// Placeholders stand for memory outside the stream's region, so they never move its peak.
    #[test]
    fn placeholders_are_not_the_streams_arena() {
        let mut footprint = Footprint::default();
        footprint
            .walk(None, |walk| {
                walk.placeholder(DType::F32, Shape::new(&[1 << 20]))?;
                walk.tables(&[StateTable {
                    name: "positions",
                    dtype: DType::I32,
                    shape: Shape::new(&[64, 3]),
                }])?;
                Ok(())
            })
            .unwrap();

        assert_eq!(footprint.peak, 0);
        assert!(footprint.keys.is_empty());
    }

    /// A walk that fails leaves the thread without a sink, so the next capture or walk on it starts
    /// clean, and its partial keys are not taken.
    #[test]
    fn a_failed_walk_removes_its_sink_and_keeps_nothing() {
        let mut footprint = Footprint::default();
        let refused = footprint.walk(None, |walk| {
            forward(&walk.forward(None)?, 512, 2)?;
            Err(Error::Invalid("the walk refused its bucket".into()))
        });

        assert!(refused.is_err());
        assert!(!ops::capture::installed());
        assert_eq!(footprint, Footprint::default());
    }
}
