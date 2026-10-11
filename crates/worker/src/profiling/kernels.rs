//! Kernels selected by numerical layers, logged once at startup and again
//! when serving selects new ones.
//!
//! Each binding reports one record per prepared operator (`op`,
//! `provider`, optional per-input providers) with the layer `path`. The log
//! line `uniserve-kernel-table {json}` groups equal records into call sites
//! with compressed layer path ranges, and lists the CUDA call sites served
//! by portable torch code.

use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::ffi::Record;
use crate::{Error, Result};

const TAG: &str = "uniserve-kernel-table";

/// The log target the kernel table is written under.
pub const TARGET: &str = "uniserve_worker.execution.model_executor";

type LayerRecords = IndexMap<String, Vec<Map<String, Value>>>;

/// Selections by runner label. A retired binding keeps its last report.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct KernelRecords {
    runners: IndexMap<String, (String, LayerRecords)>,
}

impl KernelRecords {
    /// Add one binding's records under `label` (component, root path and
    /// sorted call kinds).
    pub fn add(&mut self, label: &str, device: &str, records: &[Record]) -> Result<()> {
        let records = records
            .iter()
            .map(Record::to_json)
            .collect::<Result<Vec<_>>>()?;
        self.insert(label, device, records)
    }

    /// Add a runner's kernel listings under `label`: each the JSON array of
    /// records its Python half's `kernels` method returned.
    pub fn add_listings(&mut self, label: &str, device: &str, listings: &[String]) -> Result<()> {
        for listing in listings {
            let records: Vec<Map<String, Value>> = serde_json::from_str(listing)
                .map_err(|error| Error::Interface(format!("kernel listing of {label}: {error}")))?;
            self.insert(label, device, records)?;
        }
        Ok(())
    }

    fn insert(
        &mut self,
        label: &str,
        device: &str,
        records: Vec<Map<String, Value>>,
    ) -> Result<()> {
        let (_, paths) = self
            .runners
            .entry(label.to_owned())
            .or_insert_with(|| (device.to_owned(), IndexMap::new()));
        for mut record in records {
            let path = record
                .remove("path")
                .and_then(|path| path.as_str().map(str::to_owned))
                .ok_or_else(|| Error::Interface("kernel records require a layer path".into()))?;
            let values = paths.entry(path).or_default();
            if !values.contains(&record) {
                values.push(record);
            }
        }
        Ok(())
    }

    /// Merge a newer report; returns whether any layer's selection changed.
    pub fn merge(&mut self, current: Self) -> bool {
        let mut changed = false;
        for (label, (device, paths)) in current.runners {
            let (_, stored) = self
                .runners
                .entry(label)
                .or_insert_with(|| (device, IndexMap::new()));
            for (path, records) in paths {
                if stored.get(&path) != Some(&records) {
                    stored.insert(path, records);
                    changed = true;
                }
            }
        }
        changed
    }

    /// The table for `stage` (`startup` or `serving`).
    pub fn table(&self, stage: &str, rank: usize, device: &str) -> Value {
        let mut runners = Vec::new();
        let mut portable = Vec::new();
        for (label, (device, paths)) in &self.runners {
            let mut sites: Vec<(&Map<String, Value>, Vec<&str>)> = Vec::new();
            for (path, records) in paths {
                // An attention layer with a prepared provider reports its
                // unprepared default too; the prepared one is the call site.
                let prepared = records
                    .iter()
                    .any(|record| record["op"] == "attention" && !record["provider"].is_null());
                for record in records {
                    if prepared && record["op"] == "attention" && record["provider"].is_null() {
                        continue;
                    }
                    match sites.iter_mut().find(|(value, _)| *value == record) {
                        Some((_, layers)) => layers.push(path),
                        None => sites.push((record, vec![path])),
                    }
                }
            }
            let mut calls = Vec::new();
            for (record, paths) in sites {
                let layers = layer_paths(paths);
                let mut call = record.clone();
                call.insert("layers".to_owned(), json!(layers));
                calls.push(call);
                let served: Vec<&str> = if record["provider"] == "torch" {
                    vec!["every input"]
                } else if record["provider"].is_null() {
                    Vec::new()
                } else {
                    record
                        .get("inputs")
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(|inputs| {
                            inputs
                                .iter()
                                .filter(|(_, provider)| **provider == "torch")
                                .map(|(name, _)| name.as_str())
                        })
                        .collect()
                };
                if !served.is_empty() && (device == "cuda" || device.starts_with("cuda:")) {
                    portable.push(json!({ "runner": label, "op": record["op"], "layers": layers, "inputs": served }));
                }
            }
            runners.push(json!({ "runner": label, "device": device, "call_sites": calls }));
        }
        json!({ "stage": stage, "rank": rank, "device": device, "runners": runners, "portable": portable })
    }

    /// Log the table at INFO.
    pub fn log(&self, stage: &str, rank: usize, device: &str) {
        tracing::info!(target: TARGET, "{TAG} {}", self.table(stage, rank, device));
    }
}

/// Compress numbered layer paths into ranges: `layers.{0-3,5}.mlp`.
fn layer_paths(paths: Vec<&str>) -> Vec<String> {
    let mut templates: IndexMap<Vec<&str>, Vec<Vec<&str>>> = IndexMap::new();
    for path in paths {
        let parts: Vec<_> = path.split('.').collect();
        let key = parts
            .iter()
            .map(|&part| {
                if part.parse::<u64>().is_ok() {
                    "*"
                } else {
                    part
                }
            })
            .collect();
        templates.entry(key).or_default().push(parts);
    }
    let mut result = Vec::new();
    for (key, members) in templates {
        let varying: Vec<_> = key
            .iter()
            .enumerate()
            .filter(|&(position, &part)| {
                part == "*"
                    && members
                        .iter()
                        .any(|member| member[position] != members[0][position])
            })
            .map(|(position, _)| position)
            .collect();
        if varying.len() > 1 {
            result.extend(members.iter().map(|parts| parts.join(".")));
            continue;
        }
        let mut parts: Vec<_> = members[0].iter().map(|&part| part.to_owned()).collect();
        if let Some(&position) = varying.first() {
            let numbers: BTreeSet<_> = members
                .iter()
                .filter_map(|member| member[position].parse::<u64>().ok())
                .collect();
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for number in numbers {
                match runs.last_mut() {
                    Some((_, stop)) if stop.checked_add(1) == Some(number) => *stop = number,
                    _ => runs.push((number, number)),
                }
            }
            let ranges: Vec<_> = runs
                .into_iter()
                .map(|(start, stop)| {
                    if start == stop {
                        start.to_string()
                    } else {
                        format!("{start}-{stop}")
                    }
                })
                .collect();
            parts[position] = format!("{{{}}}", ranges.join(","));
        }
        result.push(parts.join("."));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbered_layers_compress_into_ranges() {
        let paths = vec![
            "layers.0.mlp",
            "layers.1.mlp",
            "layers.2.mlp",
            "layers.5.mlp",
            "head",
        ];
        assert_eq!(
            layer_paths(paths),
            vec!["layers.{0-2,5}.mlp".to_owned(), "head".to_owned()]
        );
        // Two varying positions cannot share one template.
        assert_eq!(
            layer_paths(vec!["a.0.b.1", "a.1.b.0"]),
            vec!["a.0.b.1".to_owned(), "a.1.b.0".to_owned()]
        );
    }

    #[test]
    fn equal_records_share_a_call_site_and_portable_sites_are_listed() {
        let site = |provider: &str| {
            let mut record = Map::new();
            record.insert("op".into(), json!("matmul"));
            record.insert("provider".into(), json!(provider));
            record
        };
        let mut records = KernelRecords::default();
        let (_, paths) = records
            .runners
            .entry("model[prefill]".into())
            .or_insert_with(|| ("cuda:0".into(), IndexMap::new()));
        paths.insert("layers.0.proj".into(), vec![site("cublas")]);
        paths.insert("layers.1.proj".into(), vec![site("cublas")]);
        paths.insert("head".into(), vec![site("torch")]);
        let table = records.table("startup", 0, "cuda:0");
        let calls = table["runners"][0]["call_sites"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["layers"], json!(["layers.{0-1}.proj"]));
        assert_eq!(table["portable"][0]["layers"], json!(["head"]));

        let mut current = records.clone();
        assert!(!records.merge(current.clone()));
        current.runners["model[prefill]"]
            .1
            .insert("head".into(), vec![site("cutlass")]);
        assert!(records.merge(current));
    }
}
