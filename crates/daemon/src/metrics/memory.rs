//! The main processes' allocators as the deployment exports them.
//!
//! Two questions, one family each: how much physical memory a device's budget allows this epoch
//! (`uniserve:memory_budget`) and how much the device's allocator pools and its KV regions'
//! planes have mapped now (`uniserve:memory_mapped`, by pool). The invariant the design states -
//! a device never maps more than its budget allows - is then a comparison of two series the
//! deployment publishes, rather than a claim only the main process can see.
//!
//! Each replica's main process reports both at its report period
//! (`EngineInfo::report_period`), per device, listing every pool its allocator has. A device is
//! numbered by the process that owns it, so a series is labelled by the replica whose process
//! reported it as well as by the device: two replicas' processes on different hosts both have a
//! device 0, and their series stay apart, each comparable with its own budget. A report sets the
//! series it names; a series no report names keeps its last value, which is what a drained
//! epoch leaves behind until the next epoch on that device reports.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::metrics::U64Gauge;

/// The device a budget belongs to.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceLabels {
    /// The replica whose main process reports the device.
    pub replica: u32,
    /// The device's index, as that process numbers it.
    pub device: u32,
}

/// The device and pool a mapping belongs to.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PoolLabels {
    /// The replica whose main process reports the device.
    pub replica: u32,
    /// The device's index, as that process numbers it.
    pub device: u32,
    /// The pool: an allocator's `graph`, `scratch` or `resident`, or `kv`, the planes of the
    /// device's KV regions.
    pub pool: String,
}

/// The allocator families: one budget per device, one mapping per device and pool.
pub struct MemoryMetrics {
    /// `uniserve:memory_budget{replica, device}` in bytes.
    pub budget: Family<DeviceLabels, U64Gauge>,
    /// `uniserve:memory_mapped{replica, device, pool}` in bytes.
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

    /// Writes one report of `replica`'s main process: every device's budget, and every pool it
    /// lists.
    pub fn record(&self, replica: u32, devices: &[proto::request::wire::DeviceMemory]) {
        for device in devices {
            self.budget
                .get_or_create(&DeviceLabels {
                    replica,
                    device: device.device,
                })
                .set(device.budget);
            for pool in &device.pools {
                self.mapped
                    .get_or_create(&PoolLabels {
                        replica,
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
    use super::{DeviceLabels, MemoryMetrics, PoolLabels};
    use proto::request::wire::{DeviceMemory, PoolMemory};

    /// The series of `pool` on device `device` of replica `replica`.
    fn pool(replica: u32, device: u32, pool: &str) -> PoolLabels {
        PoolLabels {
            replica,
            device,
            pool: pool.to_owned(),
        }
    }

    /// A report publishes the device's budget and each pool's mapping, each a series of its own,
    /// summed by the reader rather than by the writer.
    #[test]
    fn a_report_publishes_the_budget_and_each_pools_mapping() {
        let metrics = MemoryMetrics::register(&mut Default::default());
        metrics.record(
            0,
            &[DeviceMemory {
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
            }],
        );

        let device = DeviceLabels {
            replica: 0,
            device: 1,
        };
        assert_eq!(metrics.budget.get_or_create(&device).get(), 8_388_608);
        assert_eq!(
            metrics.mapped.get_or_create(&pool(0, 1, "graph")).get(),
            2_097_152
        );
        assert_eq!(
            metrics.mapped.get_or_create(&pool(0, 1, "resident")).get(),
            4_194_304
        );
    }

    /// A later report replaces the values it names, so the series follows the allocator instead of
    /// accumulating across reports; another replica's device of the same number is its own series.
    #[test]
    fn a_later_report_replaces_the_values_it_names() {
        let metrics = MemoryMetrics::register(&mut Default::default());
        let report = |mapped| {
            [DeviceMemory {
                device: 0,
                budget: 8_388_608,
                pools: vec![PoolMemory {
                    pool: "scratch".to_owned(),
                    mapped,
                }],
            }]
        };
        metrics.record(0, &report(1_048_576));
        metrics.record(0, &report(3_145_728));
        metrics.record(1, &report(524_288));

        assert_eq!(
            metrics.mapped.get_or_create(&pool(0, 0, "scratch")).get(),
            3_145_728,
            "the second report's value is the series"
        );
        assert_eq!(
            metrics.mapped.get_or_create(&pool(1, 0, "scratch")).get(),
            524_288,
            "another replica's device 0 is a series of its own"
        );
    }
}
