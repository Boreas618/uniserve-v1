//! Weight chunking and copy manifests (`storage::manifest`), exported to
//! Python.
//!
//! A planner of a world change computes the retained storages a rank of a
//! parallel configuration holds and the manifest that redistributes them
//! between two worlds. Loading the library starts nothing, so any process
//! may call these by name (`uniserve.worker.library()`): a test or a
//! harness that loads a model under two configurations on the host checks
//! with them that a manifest turns one into the other. Each function takes
//! and returns JSON text; a description it cannot read raises
//! `ValueError`.

use crate::memory::weights;

type Result<T> = tvm_ffi::Result<T>;

fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

/// `weight_storages(description) -> storages`: the retained storages, with
/// their chunks and keys, of a rank whose loaded layout `description`
/// describes ([`weights::plan_storages`]).
fn weight_storages(description: tvm_ffi::String) -> Result<tvm_ffi::String> {
    weights::plan_storages(description.as_str())
        .map(|storages| tvm_ffi::String::from(storages.as_str()))
        .map_err(|error| value_error(&error.to_string()))
}

/// `copy_manifest(request) -> manifest`: the manifest of a
/// `storage::manifest::Request`, both as JSON.
fn copy_manifest(request: tvm_ffi::String) -> Result<tvm_ffi::String> {
    let request: storage::manifest::Request = serde_json::from_str(request.as_str())
        .map_err(|error| value_error(&format!("a copy manifest request: {error}")))?;
    let manifest = storage::manifest::Manifest::plan(&request)
        .map_err(|error| value_error(&error.to_string()))?;
    serde_json::to_string(&manifest)
        .map(|manifest| tvm_ffi::String::from(manifest.as_str()))
        .map_err(|error| value_error(&format!("encoding a copy manifest: {error}")))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(weight_storages, weight_storages);
tvm_ffi::tvm_ffi_dll_export_typed_func!(copy_manifest, copy_manifest);
