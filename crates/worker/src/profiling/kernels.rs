//! Kernels selected by numerical layers, logged once at startup and again
//! when serving selects new ones.
//!
//! Each runner reports one record per prepared operator as the text of a
//! JSON object. The rank does not read the records: it keeps them by runner,
//! notices when a runner's report changes, and hands them back to the
//! library, which renders the kernel table (`uniserve.profiling.kernel_table`). The
//! table is logged as the line `uniserve-kernel-table {json}`.
//!
//! Startup lists and renders with calls it waits for. While serving, the
//! control thread waits for none of it ([`KernelWatch`]): the library's
//! count of kernel choices, the runners' listings and the rendering are
//! calls submitted to the Python thread, each taken up at its completion.

use std::collections::HashMap;

use indexmap::IndexMap;
use tvm_ffi::Any;

use crate::ffi::{Function, Interpreter};
use crate::lane::{Completion, Job, JobId, JobQueues};
use crate::runner::methods::{Argument, MethodCall};
use crate::{Error, Result};

const TAG: &str = "uniserve-kernel-table";

/// The library function rendering the kernel table.
const TABLE: &str = "uniserve.profiling.kernel_table";

/// One operator's kernel record, as a runner's `kernels` method reports it:
/// a JSON object text the rank passes on unread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelRecord(String);

impl KernelRecord {
    /// Take the record `value` that `owner` reported.
    pub(crate) fn new(owner: impl Into<String>, value: Any) -> Result<Self> {
        tvm_ffi::String::try_from(value)
            .map(|text| Self(text.as_str().to_owned()))
            .map_err(|error| Error::numerical(&owner.into(), &error))
    }

    /// The record whose JSON object text is `text`.
    pub(crate) fn from_text(text: String) -> Self {
        Self(text)
    }
}

/// The listing of one runner's kernels: the runner's kernel-table label,
/// its device name and the calls of its Python runners' `kernels` methods,
/// each of which returns one record text per prepared operator.
pub(crate) type KernelCalls = (String, String, Vec<MethodCall>);

/// The records of each runner label, with the runner's device name. Runners
/// sharing a label report under it together.
#[derive(Clone)]
pub(crate) struct KernelRecords {
    table: Function,
    runners: IndexMap<String, (String, Vec<KernelRecord>)>,
}

impl KernelRecords {
    /// An empty report that renders through the library's table function.
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            table: Interpreter::import(TABLE)?,
            runners: IndexMap::new(),
        })
    }

    /// An empty report rendering like this one.
    pub(crate) fn empty(&self) -> Self {
        Self {
            table: self.table.clone(),
            runners: IndexMap::new(),
        }
    }

    /// Add one runner's records under `label` (its component and module
    /// path).
    pub(crate) fn add(&mut self, label: &str, device: &str, records: &[KernelRecord]) {
        self.runners
            .entry(label.to_owned())
            .or_insert_with(|| (device.to_owned(), Vec::new()))
            .1
            .extend_from_slice(records);
    }

    /// Merge a newer report; returns whether any label's records changed.
    /// A label the newer report has no records for keeps its last ones, as
    /// a retired runner reports none.
    pub(crate) fn merge(&mut self, current: Self) -> bool {
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

    /// The library call that renders the table for `stage` (`startup` or
    /// `serving`) from the records; it returns the table's text.
    fn table_call(&self, stage: &str, rank: usize, device: &str) -> MethodCall {
        let devices = self.runners.values().map(|(device, _)| device.clone());
        let counts = self
            .runners
            .values()
            .map(|(_, records)| records.len() as i64);
        let records = self
            .runners
            .values()
            .flat_map(|(_, records)| records)
            .map(|record| record.0.clone());

        // Each runner's records cross back to back, with their counts.
        MethodCall::new(
            &self.table,
            vec![
                Argument::Text(stage.to_owned()),
                Argument::Integer(rank as i64),
                Argument::Text(device.to_owned()),
                Argument::Texts(self.runners.keys().cloned().collect()),
                Argument::Texts(devices.collect()),
                Argument::Integers(counts.collect()),
                Argument::Texts(records.collect()),
            ],
        )
    }

    /// The table for `stage`, as the library renders it from the records:
    /// a call made now, which its caller waits for (startup).
    pub(crate) fn table(&self, stage: &str, rank: usize, device: &str) -> Result<String> {
        let rendered = self.table_call(stage, rank, device).invoke()?;
        tvm_ffi::String::try_from(rendered)
            .map(|table| table.as_str().to_owned())
            .map_err(|error| Error::numerical(TABLE, &error))
    }

    /// Log the table at INFO.
    pub(crate) fn log(&self, stage: &str, rank: usize, device: &str) -> Result<()> {
        tracing::info!("{TAG} {}", self.table(stage, rank, device)?);
        Ok(())
    }
}

/// What a [`KernelWatch`] submitted to the Python thread and has not heard
/// of yet.
enum Pending {
    /// The library's count of kernel choices.
    Count,
    /// The runners' kernel listings: each listed runner's label, device
    /// and number of calls, in the order of the job's calls.
    Listing(Vec<(String, String, usize)>),
    /// The rendering of the table.
    Table,
}

/// Serving-time kernel reporting. After each batch's forward the library's
/// count of kernel choices is asked for; when it has moved, the runners'
/// kernels are listed, and when a runner's records changed, the table is
/// rendered and logged. Each step is a call submitted to the Python thread
/// and taken up at its completion: none is waited for, and a step that
/// fails is logged and fails no batch.
pub(crate) struct KernelWatch {
    choices: Function,
    /// The count the report in `records` was made at.
    last: i64,
    records: KernelRecords,
    rank: usize,
    device: String,
    pending: HashMap<JobId, Pending>,
    /// Kernels were chosen while a listing was in flight: the runners are
    /// listed again at its completion.
    stale: bool,
}

impl KernelWatch {
    /// A watch that reports changes from `records`, the startup report,
    /// made when `choices` returned `last`.
    pub(crate) fn new(
        choices: Function,
        last: i64,
        records: KernelRecords,
        rank: usize,
        device: String,
    ) -> Self {
        Self {
            choices,
            last,
            records,
            rank,
            device,
            pending: HashMap::new(),
            stale: false,
        }
    }

    /// Ask for the count of kernel choices, after a batch's forward. The
    /// count needs no wake of its own: it is taken up with the next
    /// completion the control thread takes.
    pub(crate) fn count(&mut self, jobs: &JobQueues) {
        let calls = Job::new();
        calls.call_returning(MethodCall::new(&self.choices, Vec::new()));
        self.pending.insert(jobs.call(calls, false), Pending::Count);
    }

    /// Whether `job` is a call this watch submitted.
    pub(crate) fn owns(&self, job: JobId) -> bool {
        self.pending.contains_key(&job)
    }

    /// Apply the completion of `job`, a call this watch submitted, and
    /// submit what follows it. `runners` lists the rank's runners when
    /// their kernels are to be listed.
    pub(crate) fn complete(
        &mut self,
        job: JobId,
        completion: Completion,
        jobs: &JobQueues,
        runners: impl FnOnce() -> Vec<KernelCalls>,
    ) {
        let Some(pending) = self.pending.remove(&job) else {
            return;
        };
        let completed = completion.check().map_err(|failure| failure.error);
        let applied = completed.and_then(|mut completion| match pending {
            Pending::Count => {
                let count = completion.returned()?;
                if count.first().is_some_and(|&count| count != self.last) {
                    self.last = count[0];
                    self.list(jobs, runners);
                }
                Ok(())
            }
            Pending::Listing(listed) => {
                let mut current = self.records.empty();
                for (label, device, calls) in &listed {
                    let mut records = Vec::new();
                    for _ in 0..*calls {
                        let texts = completion.texts()?;
                        records.extend(texts.into_iter().map(KernelRecord::from_text));
                    }
                    current.add(label, device, &records);
                }
                if self.records.merge(current) {
                    let calls = Job::new();
                    calls.call_texts(self.records.table_call("serving", self.rank, &self.device));
                    self.pending.insert(jobs.call(calls, true), Pending::Table);
                }
                if std::mem::take(&mut self.stale) {
                    self.list(jobs, runners);
                }
                Ok(())
            }
            Pending::Table => {
                let table = completion.texts()?;
                tracing::info!("{TAG} {}", table.concat());
                Ok(())
            }
        });
        if let Err(error) = applied {
            tracing::warn!("kernel report failed: {error}");
        }
    }

    /// List the runners' kernels, unless a listing is in flight: that one
    /// may predate the newest choice, so the runners are listed again when
    /// it completed.
    fn list(&mut self, jobs: &JobQueues, runners: impl FnOnce() -> Vec<KernelCalls>) {
        let listing = |pending: &Pending| matches!(pending, Pending::Listing(_));
        if self.pending.values().any(listing) {
            self.stale = true;
            return;
        }

        let calls = Job::new();
        let mut listed = Vec::new();
        for (label, device, listing) in runners() {
            listed.push((label, device, listing.len()));
            for call in listing {
                calls.call_texts(call);
            }
        }
        self.pending
            .insert(jobs.call(calls, true), Pending::Listing(listed));
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
    fn a_moved_choice_count_lists_the_runners_and_renders_what_changed() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::{Arc, Mutex};

        use tvm_ffi::AnyView;

        // The library's count, one runner's listing and the renderer, which
        // keeps the stage and the records it was given.
        let count = Arc::new(AtomicI64::new(3));
        let choices = {
            let count = Arc::clone(&count);
            tvm_ffi::Function::from_packed(move |_| Ok(Any::from(count.load(Ordering::Relaxed))))
        };
        let kernels =
            tvm_ffi::Function::from_packed(|_| Ok(Any::from(crate::ffi::strings(["a", "d"]))));
        let rendered = Arc::new(Mutex::new(Vec::new()));
        let table = {
            let rendered = Arc::clone(&rendered);
            tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
                let stage = tvm_ffi::String::try_from(args[0])?.as_str().to_owned();
                let records = crate::ffi::texts(Any::from(args[6]))?;
                rendered.lock().unwrap().push((stage, records));
                Ok(Any::from(tvm_ffi::String::from("{}")))
            })
        };

        let mut startup = report(&[("model", &["a", "b"])]);
        startup.table = Function::new(TABLE, table);
        let mut watch = KernelWatch::new(
            Function::new("test.choices", choices),
            3,
            startup,
            0,
            "cuda:0".to_owned(),
        );
        let kernels = Function::new("test.kernels", kernels);
        let runners = || {
            let call = MethodCall::new(&kernels, Vec::new());
            vec![("model".to_owned(), "cuda:0".to_owned(), vec![call])]
        };

        // A rank without lanes and without a started interpreter performs
        // each call where it is submitted; its completion is taken here as
        // the control thread takes it. Returns whether one was taken.
        let jobs = JobQueues::default();
        let take = |watch: &mut KernelWatch| {
            let Some((job, completion)) = jobs.take() else {
                return false;
            };
            assert!(watch.owns(job));
            watch.complete(job, completion, &jobs, runners);
            true
        };

        // The count has not moved: nothing is listed.
        watch.count(&jobs);
        assert!(take(&mut watch));
        assert!(!take(&mut watch));

        // It moved: the runners are listed, and their changed records are
        // rendered for the serving stage.
        count.store(4, Ordering::Relaxed);
        watch.count(&jobs);
        assert!(take(&mut watch), "the count");
        assert!(take(&mut watch), "the listing");
        assert!(take(&mut watch), "the table");
        assert!(!take(&mut watch));
        let expected = ("serving".to_owned(), vec!["a".to_owned(), "d".to_owned()]);
        assert_eq!(*rendered.lock().unwrap(), [expected]);

        // The same count again lists nothing more.
        watch.count(&jobs);
        assert!(take(&mut watch));
        assert!(!take(&mut watch));
        assert_eq!(rendered.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_retired_runner_keeps_its_last_records() {
        let mut stored = report(&[("model", &["a"])]);
        assert!(!stored.merge(report(&[("model", &[])])));
        assert_eq!(stored.runners["model"].1, [record("a")]);
    }
}
