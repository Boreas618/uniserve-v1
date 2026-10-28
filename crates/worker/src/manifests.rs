//! Weight chunking and copy manifests (`::storage::manifest`), exported to
//! Python.
//!
//! A planner of a world change computes the retained storages a rank of a
//! parallel configuration holds and the manifest that redistributes them
//! between two worlds; a denoising rank computes the copies that lay a
//! stored trajectory's pages out in its own. Loading the library starts
//! nothing, so any process may call these by name
//! (`uniserve.worker.library()`): a test or a harness that describes a
//! model's state under two configurations on the host checks with them
//! that a manifest turns one into the other. Each function takes and
//! returns JSON text; a description it cannot read raises `ValueError`.

use crate::storage::latent::restore as latent;
use crate::storage::weights;

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
/// `::storage::manifest::Request`, both as JSON.
fn copy_manifest(request: tvm_ffi::String) -> Result<tvm_ffi::String> {
    let request: ::storage::manifest::Request = serde_json::from_str(request.as_str())
        .map_err(|error| value_error(&format!("a copy manifest request: {error}")))?;
    let manifest = ::storage::manifest::Manifest::plan(&request)
        .map_err(|error| value_error(&error.to_string()))?;
    serde_json::to_string(&manifest)
        .map(|manifest| tvm_ffi::String::from(manifest.as_str()))
        .map_err(|error| value_error(&format!("encoding a copy manifest: {error}")))
}

/// A restore of stored latent pages into a rank's request slot pages, as
/// `latent_restore_copies` reads it.
#[derive(serde::Deserialize)]
struct LatentRestore {
    /// The stored ranks' slot pages.
    sources: Vec<LatentPages>,
    /// The restoring rank's slot pages.
    target: LatentPages,
    /// What the stored values are (`ipc::LatentRestore::content`).
    content: [u8; 32],
}

#[derive(serde::Deserialize)]
struct LatentPages {
    layout: ::storage::manifest::StorageLayout,
    page_bytes: u64,
}

impl From<LatentPages> for latent::Stored {
    fn from(pages: LatentPages) -> Self {
        Self {
            layout: pages.layout,
            page_bytes: pages.page_bytes,
        }
    }
}

/// `latent_restore_copies(restore) -> copies`: the copies a denoising rank
/// makes to lay stored latent pages out in its own request slot pages
/// ([`latent::copies`]), for a JSON restore `{"sources": [{"layout",
/// "page_bytes"}], "target": {"layout", "page_bytes"}, "content": [32
/// bytes]}` of executor-declared layouts; each copy is `[source, page,
/// offset, target, bytes]`.
fn latent_restore_copies(restore: tvm_ffi::String) -> Result<tvm_ffi::String> {
    let restore: LatentRestore = serde_json::from_str(restore.as_str())
        .map_err(|error| value_error(&format!("a latent restore: {error}")))?;
    let sources: Vec<latent::Stored> = restore.sources.into_iter().map(Into::into).collect();
    // The copies name no GPU; any one plans them alike.
    let copies = latent::copies(&sources, &restore.target.into(), restore.content, [0; 16])
        .map_err(|error| value_error(&error.to_string()))?;
    let copies: Vec<[u64; 5]> = copies
        .iter()
        .map(|copy| {
            [
                copy.source as u64,
                copy.page,
                copy.offset,
                copy.target,
                copy.bytes,
            ]
        })
        .collect();
    serde_json::to_string(&copies)
        .map(|copies| tvm_ffi::String::from(copies.as_str()))
        .map_err(|error| value_error(&format!("encoding latent copies: {error}")))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(weight_storages, weight_storages);
tvm_ffi::tvm_ffi_dll_export_typed_func!(copy_manifest, copy_manifest);
tvm_ffi::tvm_ffi_dll_export_typed_func!(latent_restore_copies, latent_restore_copies);
