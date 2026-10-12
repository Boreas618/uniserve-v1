//! Kernels selected by numerical layers, logged once at startup and again
//! when serving selects new ones.
//!
//! Each runner reports one record per prepared operator as the text of a
//! JSON object. The rank does not read the records: it keeps them by runner,
//! notices when a runner's report changes, and hands them back to the
//! library, which renders the kernel table (`interface.kernels.table`). The
//! table is logged as the line `uniserve-kernel-table {json}`.

use indexmap::IndexMap;
use tvm_ffi::Any;

use crate::ffi::{Function, integer_list, strings};
use crate::model::interface::Interface;
use crate::{Error, Result};

const TAG: &str = "uniserve-kernel-table";

/// The log target the kernel table is written under.
pub const TARGET: &str = "uniserve_worker.execution.model_executor";

/// The library function rendering the kernel table.
const TABLE: &str = "uniserve_models.interface.kernels.table";

/// One operator's kernel record, as a runner's `kernels` method reports it:
/// a JSON object text the rank passes on unread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRecord(String);

impl KernelRecord {
    /// Take the record `value` that `owner` reported.
    pub fn new(owner: impl Into<String>, value: Any) -> Result<Self> {
        tvm_ffi::String::try_from(value)
            .map(|text| Self(text.as_str().to_owned()))
            .map_err(|error| Error::numerical(&owner.into(), &error))
    }
}

/// The records of each runner label, with the runner's device name. Runners
/// sharing a label report under it together.
#[derive(Clone)]
pub struct KernelRecords {
    table: Function,
    runners: IndexMap<String, (String, Vec<KernelRecord>)>,
}

impl KernelRecords {
    /// An empty report that renders through the library's table function.
    pub fn new(interface: &Interface) -> Result<Self> {
        Ok(Self {
            table: interface.import(TABLE)?,
            runners: IndexMap::new(),
        })
    }

    /// An empty report rendering like this one.
    pub fn empty(&self) -> Self {
        Self {
            table: self.table.clone(),
            runners: IndexMap::new(),
        }
    }

    /// Add one runner's records under `label` (its component and module
    /// path).
    pub fn add(&mut self, label: &str, device: &str, records: &[KernelRecord]) {
        self.runners
            .entry(label.to_owned())
            .or_insert_with(|| (device.to_owned(), Vec::new()))
            .1
            .extend_from_slice(records);
    }

    /// Merge a newer report; returns whether any label's records changed.
    /// A label the newer report has no records for keeps its last ones, as
    /// a retired runner reports none.
    pub fn merge(&mut self, current: Self) -> bool {
        let mut changed = false;
        for (label, (device, records)) in current.runners {
            if records.is_empty() {
                continue;
            }
            match self.runners.get_mut(&label) {
                Some((_, stored)) if *stored == records => {}
                Some((_, stored)) => {
                    *stored = records;
                    changed = true;
                }
                None => {
                    self.runners.insert(label, (device, records));
                    changed = true;
                }
            }
        }
        changed
    }

    /// The table for `stage` (`startup` or `serving`), as the library
    /// renders it from the records.
    pub fn table(&self, stage: &str, rank: usize, device: &str) -> Result<String> {
        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        // Each runner's records cross back to back, with their counts.
        let rendered = self.table.call_values(&[
            text(stage),
            Any::from(rank as i64),
            text(device),
            Any::from(strings(self.runners.keys())),
            Any::from(strings(self.runners.values().map(|(device, _)| device))),
            Any::from(integer_list(
                self.runners
                    .values()
                    .map(|(_, records)| records.len() as i64),
            )),
            Any::from(strings(
                self.runners
                    .values()
                    .flat_map(|(_, records)| records)
                    .map(|record| &record.0),
            )),
        ])?;
        tvm_ffi::String::try_from(rendered)
            .map(|table| table.as_str().to_owned())
            .map_err(|error| Error::numerical(TABLE, &error))
    }

    /// Log the table at INFO.
    pub fn log(&self, stage: &str, rank: usize, device: &str) -> Result<()> {
        tracing::info!(target: TARGET, "{TAG} {}", self.table(stage, rank, device)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(text: &str) -> KernelRecord {
        KernelRecord(text.to_owned())
    }

    /// A report over a table function that is never called.
    fn report(runners: &[(&str, &[&str])]) -> KernelRecords {
        let table = tvm_ffi::Function::from_packed(|_| Ok(Any::default()));
        let mut records = KernelRecords {
            table: Function::new(TABLE, table),
            runners: IndexMap::new(),
        };
        for (label, texts) in runners {
            let texts: Vec<KernelRecord> = texts.iter().map(|text| record(text)).collect();
            records.add(label, "cuda:0", &texts);
        }
        records
    }

    #[test]
    fn a_changed_report_replaces_its_runners_records() {
        let mut stored = report(&[("model", &["a", "b"]), ("vision", &["c"])]);

        // The same records are no change.
        assert!(!stored.merge(report(&[("model", &["a", "b"])])));
        // A runner selecting another kernel reports different records.
        assert!(stored.merge(report(&[("model", &["a", "d"])])));
        assert_eq!(stored.runners["model"].1, [record("a"), record("d")]);
        // A new runner is a change.
        assert!(stored.merge(report(&[("decoder", &["e"])])));
    }

    #[test]
    fn a_retired_runner_keeps_its_last_records() {
        let mut stored = report(&[("model", &["a"])]);
        assert!(!stored.merge(report(&[("model", &[])])));
        assert_eq!(stored.runners["model"].1, [record("a")]);
    }
}
