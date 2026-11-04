//! One generation of loaded kernels.
//!
//! A runner holds the generation it is serving and, during a program-layer update, the one it is
//! switching to (specs/worker.md 4.2, 4.5, 7.3). A generation is immutable: a switchover builds the
//! next one from the current one plus the kernels of the operators whose program changed, so every
//! operator whose program did not change keeps its `Arc<Kernel>` — the same loaded module, not a
//! second copy — and the modules of a replaced generation unload when the last `Arc` to them drops.
//!
//! A generation is keyed by (operator name, descriptor key). The operator name is the crate-level
//! name a kernel program fills, the descriptor key is the canonical text of the values one
//! compilation covers, and together they are the module's path under the program server's store
//! (specs/worker.md 5.3).

use std::collections::HashMap;
use std::sync::Arc;

use crate::descriptor;
use crate::kernel::Kernel;
use crate::{Error, Op, Result};

/// One kernel of a generation: which operator and descriptor it was compiled for.
#[derive(Clone)]
pub struct KernelEntry {
    /// The operator's name, the same text the operator's `Op::NAME` gives.
    pub operator: &'static str,
    /// The descriptor key the module was compiled for.
    pub descriptor: String,
    /// The loaded kernel.
    pub kernel: Arc<Kernel>,
}

impl KernelEntry {
    /// The kernel of operator `operator` for `descriptor`'s key.
    pub fn new<O: Op>(descriptor: &O::Descriptor, kernel: Arc<Kernel>) -> Self {
        Self {
            operator: O::NAME,
            descriptor: descriptor::key(descriptor),
            kernel,
        }
    }

    /// The kernel of `operator` for an already-formatted key.
    pub fn keyed(operator: &'static str, key: impl Into<String>, kernel: Arc<Kernel>) -> Self {
        Self {
            operator,
            descriptor: key.into(),
            kernel,
        }
    }
}

/// The kernels one runner may launch, indexed by (operator, descriptor key).
///
/// Built once and then read-only; a program-layer update builds the successor with [`Self::replace`]
/// and hands it to the runner at a batch boundary (specs/worker.md 4.5, 7.3).
pub struct Generation {
    number: u64,
    kernels: HashMap<(&'static str, String), Arc<Kernel>>,
}

impl Generation {
    /// The generation numbered `number` holding `kernels`.
    ///
    /// A later entry for the same (operator, descriptor) replaces an earlier one, so a package that
    /// resolves the same descriptor twice cannot leave a generation with two kernels for it.
    pub fn new(number: u64, kernels: impl IntoIterator<Item = KernelEntry>) -> Self {
        let mut map: HashMap<(&'static str, String), Arc<Kernel>> = HashMap::new();
        for entry in kernels {
            map.insert((entry.operator, entry.descriptor), entry.kernel);
        }
        Self {
            number,
            kernels: map,
        }
    }

    /// The generation's number, which graph capture and purge carry (specs/worker.md 4.5, 7.3).
    pub const fn number(&self) -> u64 {
        self.number
    }

    /// The kernel of `operator` for `key`.
    pub fn kernel(&self, operator: &'static str, key: &str) -> Result<&Arc<Kernel>> {
        self.kernels
            .get(&(operator, key.to_owned()))
            .ok_or_else(|| Error::MissingKernel {
                operator,
                descriptor: key.to_owned(),
                generation: self.number,
            })
    }

    /// The kernel of operator `O` for `descriptor`.
    ///
    /// This is the call every operator's launch makes, and the only way a launch reaches a kernel:
    /// which program fills the operator is the epoch record's decision, not the launch code's
    /// (specs/worker.md 4.2).
    pub fn op<O: Op>(&self, descriptor: &O::Descriptor) -> Result<&Arc<Kernel>> {
        let key = descriptor::key(descriptor);
        self.kernel(O::NAME, &key)
    }

    /// Whether a kernel for `operator` and `key` is present.
    pub fn contains(&self, operator: &'static str, key: &str) -> bool {
        self.kernels.contains_key(&(operator, key.to_owned()))
    }

    /// How many kernels the generation holds.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the generation holds no kernel at all.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// The generation numbered `number` with the kernels of every operator named in `replacements`
    /// taken from `replacements` and everything else copied from this one.
    ///
    /// An operator whose program changed is named once and its whole set of descriptors is given:
    /// descriptors it no longer has are gone, so a generation never holds a stale kernel for an
    /// operator that was re-resolved. Every other operator keeps its `Arc<Kernel>`, so its module
    /// stays loaded across the update (specs/worker.md 4.2).
    pub fn replace(
        &self,
        number: u64,
        replacements: impl IntoIterator<Item = KernelEntry>,
    ) -> Generation {
        let replacements: Vec<KernelEntry> = replacements.into_iter().collect();
        let replaced: Vec<&'static str> = replacements.iter().map(|entry| entry.operator).collect();
        let mut kernels: HashMap<(&'static str, String), Arc<Kernel>> = self
            .kernels
            .iter()
            .filter(|((operator, _), _)| !replaced.contains(operator))
            .map(|((operator, descriptor), kernel)| {
                ((*operator, descriptor.clone()), Arc::clone(kernel))
            })
            .collect();
        for entry in replacements {
            kernels.insert((entry.operator, entry.descriptor), entry.kernel);
        }
        Generation { number, kernels }
    }

    /// Every kernel of the generation, as (operator, descriptor key, kernel).
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str, &Arc<Kernel>)> {
        self.kernels
            .iter()
            .map(|((operator, descriptor), kernel)| (*operator, descriptor.as_str(), kernel))
    }
}

impl std::fmt::Debug for Generation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Generation")
            .field("number", &self.number)
            .field("kernels", &self.len())
            .finish()
    }
}
