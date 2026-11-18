//! The process interface's wire form of the module vocabulary (`process.proto`'s `ModuleSet`,
//! `KernelPackage`, `PackageRevision` and `Value`), and the one conversion between it and the
//! Rust values the daemon, the engine and the Build Server work with.
//!
//! A package's parameters keep the types their configuration fields have: each TOML value becomes
//! the `Value` of its kind, a datetime as its RFC 3339 text in a kind of its own, so a parameter
//! reads back as exactly the value it was.

use std::collections::BTreeMap;

use crate::process as wire;

use super::{ModuleSet, ModuleSlot, PackageRevision};

/// Why a wire form is no module vocabulary value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// The module set leaves out the package of an interface it always fills.
    #[error("the module set names no {0} package")]
    Missing(&'static str),
    /// A kernel slot names no package.
    #[error("the kernel slot of {0} names no package")]
    KernelWithoutPackage(String),
    /// A parameter value of no kind.
    #[error("the parameter {0} states no value")]
    EmptyValue(String),
    /// A datetime parameter whose text is no TOML datetime.
    #[error("the parameter {0} is no datetime: {1}")]
    Datetime(String, String),
}

impl From<&PackageRevision> for wire::PackageRevision {
    fn from(package: &PackageRevision) -> Self {
        Self {
            backend: package.backend.clone(),
            revision: package.revision,
            arguments: package
                .arguments
                .iter()
                .map(|(name, value)| (name.clone(), value_to_wire(value)))
                .collect(),
        }
    }
}

impl TryFrom<wire::PackageRevision> for PackageRevision {
    type Error = WireError;

    fn try_from(package: wire::PackageRevision) -> Result<Self, WireError> {
        let mut arguments = toml::Table::new();
        for (name, value) in package.arguments {
            let value = value_of(&name, value)?;
            arguments.insert(name, value);
        }
        Ok(Self {
            backend: package.backend,
            revision: package.revision,
            arguments,
        })
    }
}

/// One slot's package as the wire states it: the `Prepare` command's and a module set's kernel.
pub fn kernel_to_wire(slot: &ModuleSlot, package: &PackageRevision) -> wire::KernelPackage {
    wire::KernelPackage {
        op: slot.op.clone(),
        key: slot.key.clone(),
        package: Some(package.into()),
        part: slot.part.clone(),
    }
}

/// The slot and the package a wire kernel names.
///
/// # Errors
///
/// A kernel without a package, or an unreadable parameter.
pub fn kernel_of(kernel: wire::KernelPackage) -> Result<(ModuleSlot, PackageRevision), WireError> {
    let package = kernel
        .package
        .ok_or_else(|| WireError::KernelWithoutPackage(kernel.op.clone()))?;
    let slot = ModuleSlot {
        op: kernel.op,
        key: kernel.key,
        part: kernel.part,
    };
    Ok((slot, package.try_into()?))
}

impl From<&ModuleSet> for wire::ModuleSet {
    fn from(modules: &ModuleSet) -> Self {
        Self {
            engine: Some((&modules.engine).into()),
            policy: Some((&modules.policy).into()),
            kernels: modules
                .kernels
                .iter()
                .map(|(slot, package)| kernel_to_wire(slot, package))
                .collect(),
        }
    }
}

impl TryFrom<wire::ModuleSet> for ModuleSet {
    type Error = WireError;

    fn try_from(modules: wire::ModuleSet) -> Result<Self, WireError> {
        let engine = modules.engine.ok_or(WireError::Missing("engine"))?;
        let policy = modules.policy.ok_or(WireError::Missing("policy"))?;
        let kernels = modules
            .kernels
            .into_iter()
            .map(kernel_of)
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Self {
            engine: engine.try_into()?,
            policy: policy.try_into()?,
            kernels,
            programs: BTreeMap::new(),
        })
    }
}

/// A TOML value as the wire's `Value` of its kind.
fn value_to_wire(value: &toml::Value) -> wire::Value {
    use wire::value::Kind;

    let kind = match value {
        toml::Value::Boolean(value) => Kind::Boolean(*value),
        toml::Value::Integer(value) => Kind::Integer(*value),
        toml::Value::Float(value) => Kind::Float(*value),
        toml::Value::String(value) => Kind::Text(value.clone()),
        toml::Value::Datetime(value) => Kind::Datetime(value.to_string()),
        toml::Value::Array(items) => Kind::List(wire::Values {
            items: items.iter().map(value_to_wire).collect(),
        }),
        toml::Value::Table(entries) => Kind::Table(wire::Table {
            entries: entries
                .iter()
                .map(|(name, value)| (name.clone(), value_to_wire(value)))
                .collect(),
        }),
    };
    wire::Value { kind: Some(kind) }
}

/// The TOML value a wire `Value` states for the parameter `name`.
fn value_of(name: &str, value: wire::Value) -> Result<toml::Value, WireError> {
    use wire::value::Kind;

    Ok(match value.kind {
        None => return Err(WireError::EmptyValue(name.to_owned())),
        Some(Kind::Boolean(value)) => toml::Value::Boolean(value),
        Some(Kind::Integer(value)) => toml::Value::Integer(value),
        Some(Kind::Float(value)) => toml::Value::Float(value),
        Some(Kind::Text(value)) => toml::Value::String(value),
        Some(Kind::Datetime(text)) => toml::Value::Datetime(text.parse().map_err(
            |error: toml::value::DatetimeParseError| {
                WireError::Datetime(name.to_owned(), error.to_string())
            },
        )?),
        Some(Kind::List(list)) => toml::Value::Array(
            list.items
                .into_iter()
                .map(|item| value_of(name, item))
                .collect::<Result<_, _>>()?,
        ),
        Some(Kind::Table(table)) => toml::Value::Table(
            table
                .entries
                .into_iter()
                .map(|(entry, value)| Ok((entry.clone(), value_of(&entry, value)?)))
                .collect::<Result<_, WireError>>()?,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A module set crosses the wire and reads back as itself, every parameter with the type its
    /// configuration gave it, a pinned key's slot and a part slot included.
    #[test]
    fn a_module_set_reads_back_as_itself() {
        let parameters: toml::Table = toml::from_str(
            r#"
            chunked_prefill_size = 4096
            ratio = 0.5
            enabled = true
            name = "fcfs"
            sizes = [1, 2, 3]
            at = 1979-05-27T07:32:00Z
            nested = { depth = 2, labels = ["a", "b"] }
            "#,
        )
        .unwrap();
        let package = |backend: &str, revision| PackageRevision {
            backend: backend.to_owned(),
            revision,
            arguments: parameters.clone(),
        };
        let modules = ModuleSet {
            engine: package("uniserve", 3),
            policy: package("fcfs", 1),
            kernels: BTreeMap::from([
                (ModuleSlot::all("gemm"), package("cublas", 2)),
                (
                    ModuleSlot::key("paged_attention", "phase_decode_head_128"),
                    package("trtllm", 4),
                ),
                (
                    ModuleSlot {
                        op: "rms_norm".to_owned(),
                        key: None,
                        part: Some("triton".to_owned()),
                    },
                    package("triton", 1),
                ),
            ]),
            programs: Default::default(),
        };

        let wire = wire::ModuleSet::from(&modules);
        let encoded = prost::Message::encode_to_vec(&wire);
        let decoded: wire::ModuleSet = prost::Message::decode(encoded.as_slice()).unwrap();
        assert_eq!(ModuleSet::try_from(decoded).unwrap(), modules);
    }

    /// A set that leaves out a package it always fills, or a kernel slot that names no package, is
    /// refused rather than read with a default.
    #[test]
    fn an_incomplete_module_set_is_refused() {
        let empty = wire::ModuleSet::default();
        assert_eq!(
            ModuleSet::try_from(empty),
            Err(WireError::Missing("engine"))
        );

        let kernel = wire::KernelPackage {
            op: "gemm".to_owned(),
            key: None,
            package: None,
            part: None,
        };
        assert_eq!(
            kernel_of(kernel),
            Err(WireError::KernelWithoutPackage("gemm".to_owned()))
        );
    }
}
