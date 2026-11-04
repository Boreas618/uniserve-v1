//! The main process's allocator as the deployment exports it (`specs/worker.md` 4.1 and 8).
//!
//! Two questions, one family each: how much physical memory a device's budget allows this epoch
//! (`uniserve:memory_budget`, labelled by device) and how much the device's allocator pools have
//! mapped now (`uniserve:memory_mapped`, labelled by device and pool). The invariant the design
//! states - a device never maps more than its budget allows - is then a comparison of two series the
//! deployment publishes, rather than a claim only the main process can see.
//!
//! The main process reports both once a second in `Metrics.memory`, per device, listing every pool
//! its allocator has. A report sets the series it names; a series no report names keeps its last
//! value, which is what a drained epoch leaves behind until the next epoch on that device reports.
//!
//! **A device index identifies a series only because the deployment executes one main process.** A
//! device index is chosen by the process that owns the device, so two hosts' processes both report
//! `device = 0` and the later report wins: on a multi-host deployment these two families would
//! describe no host's allocator, and the reader that compares mapped bytes against a budget would
//! compare across machines that never shared one. The epoch path is one node and one process today
//! (`specs/worker.md` 2.4; a replica count is refused by name), so the labels stay `device` alone -
//! and when several hosts report, the labels gain the host **together with every reader's key**,
//! because a label added on one side only turns a real comparison into a vacuous one.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::metrics::U64Gauge;

/// The device a budget belongs to.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceLabels {
    /// The device's index, as the main process numbers it.
    pub device: u32,
}

/// The device and pool a mapping belongs to.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PoolLabels {
    /// The device's index, as the main process numbers it.
    pub device: u32,
    /// The allocator's pool: `graph`, `scratch` or `resident`.
    pub pool: String,
}

/// The allocator families: one budget per device, one mapping per device and pool.
pub struct MemoryMetrics {
    /// `uniserve:memory_budget{device}` in bytes.
    pub budget: Family<DeviceLabels, U64Gauge>,
    /// `uniserve:memory_mapped{device, pool}` in bytes.
    pub mapped: Family<PoolLabels, U64Gauge>,
}

impl MemoryMetrics {
    /// Registers both families under their exported names and help, and returns the handles; the
    /// registry keeps a clone of each, whose state it shares.
    pub fn register(registry: &mut Registry) -> Self {
        let budget: Family<DeviceLabels, U64Gauge> = Default::default();
        registry.register(
            "uniserve:memory_budget",
            "Bytes of physical memory the device's budget allows this epoch",
            budget.clone(),
        );
        let mapped: Family<PoolLabels, U64Gauge> = Default::default();
        registry.register(
            "uniserve:memory_mapped",
            "Bytes of physical memory the device's allocator pool has mapped now",
            mapped.clone(),
        );
        Self { budget, mapped }
    }

    /// Writes one `Metrics.memory` report: every device's budget, and every pool it lists.
    pub fn record(&self, devices: &[proto::request::wire::DeviceMemory]) {
        for device in devices {
            self.budget
                .get_or_create(&DeviceLabels {
                    device: device.device,
                })
                .set(device.budget);
            for pool in &device.pools {
                self.mapped
                    .get_or_create(&PoolLabels {
                        device: device.device,
                        pool: pool.pool.clone(),
                    })
                    .set(pool.mapped);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use prometheus_client::encoding::text::encode;

    use super::MemoryMetrics;
    use proto::request::wire::{DeviceMemory, PoolMemory};

    fn rendered(devices: &[DeviceMemory]) -> String {
        let mut registry = Default::default();
        let metrics = MemoryMetrics::register(&mut registry);
        metrics.record(devices);
        let mut text = String::new();
        encode(&mut text, &registry).expect("the registry encodes as text");
        text
    }

    /// The two families a deployment's mapping budget is read from, under the names the design
    /// exports, each device's pools summed by the reader rather than by the writer.
    #[test]
    fn a_report_publishes_the_budget_and_each_pools_mapping() {
        let text = rendered(&[DeviceMemory {
            device: 1,
            budget: 8_388_608,
            pools: vec![
                PoolMemory {
                    pool: "graph".to_owned(),
                    mapped: 2_097_152,
                },
                PoolMemory {
                    pool: "resident".to_owned(),
                    mapped: 4_194_304,
                },
            ],
        }]);

        assert!(
            text.contains("uniserve:memory_budget{device=\"1\"} 8388608"),
            "the device's budget is a series of its own: {text}"
        );
        assert!(
            text.contains("uniserve:memory_mapped{device=\"1\",pool=\"graph\"} 2097152"),
            "each pool's mapping is exported under its own pool: {text}"
        );
        assert!(
            text.contains("uniserve:memory_mapped{device=\"1\",pool=\"resident\"} 4194304"),
            "every pool the report lists is exported: {text}"
        );
    }

    /// A later report replaces the values it names, so the series follows the allocator instead of
    /// accumulating across the reports a serving epoch sends once a second.
    #[test]
    fn a_later_report_replaces_the_values_it_names() {
        let mut registry = Default::default();
        let metrics = MemoryMetrics::register(&mut registry);
        metrics.record(&[DeviceMemory {
            device: 0,
            budget: 8_388_608,
            pools: vec![PoolMemory {
                pool: "scratch".to_owned(),
                mapped: 1_048_576,
            }],
        }]);
        metrics.record(&[DeviceMemory {
            device: 0,
            budget: 8_388_608,
            pools: vec![PoolMemory {
                pool: "scratch".to_owned(),
                mapped: 3_145_728,
            }],
        }]);

        let mut text = String::new();
        encode(&mut text, &registry).expect("the registry encodes as text");
        assert!(
            text.contains("uniserve:memory_mapped{device=\"0\",pool=\"scratch\"} 3145728"),
            "the second report's value is the series: {text}"
        );
        assert!(
            !text.contains("1048576"),
            "the first report is not added to the second: {text}"
        );
    }
}
