//! `cover` (design section 6.3): which installed package is selected for each
//! of an operator's descriptor keys. Every fixture package declares its own
//! descriptors in `[kernel.descriptors]`, so the response is read from the
//! manifests alone and nothing is built.

use std::path::Path;

use build_server::protocol::{CoverRequest, Package, Preference};
use build_server::server::BuildServer;
use build_server::target::Target;
use proto::module::Phase;

use crate::support::{self, arch, other_arch};

/// The operator every fixture package implements.
const OP: &str = "dense_attention";

/// A BF16 key, an FP32 key and an FP16 key of the operator.
const BF16: &str = "dtype_bf16_head_dim_128";
const F32: &str = "dtype_f32_head_dim_128";
const F16: &str = "dtype_f16_head_dim_128";

/// The manifest of a package of [`OP`] under `backend` at `priority`,
/// covering the element types `dtypes` at head width 128 and building for
/// `architecture`.
fn manifest(backend: &str, priority: u32, dtypes: &[&str], architecture: &Target) -> String {
    let dtypes: Vec<String> = dtypes.iter().map(|dtype| format!("\"{dtype}\"")).collect();
    support::vendor_manifest(
        OP,
        backend,
        priority,
        &format!(
            "{{ dtype = [{}], head_dim = [\"128\"] }}",
            dtypes.join(", ")
        ),
        architecture,
    )
}

/// Installs the package `manifest` on `server` from a directory under
/// `sources`, and returns the revision it was installed as.
fn install(server: &BuildServer, sources: &Path, manifest: &str) -> u32 {
    let directory = support::vendor_package(&sources.join("package"), manifest, &[]);
    server.install(&directory).unwrap().revision
}

/// A `cover` of `keys` for [`arch`].
fn request(keys: &[&str]) -> CoverRequest {
    CoverRequest {
        op: OP.to_owned(),
        targets: vec![arch()],
        keys: keys.iter().map(|key| (*key).to_owned()).collect(),
        preferred: Vec::new(),
        fixed: Vec::new(),
    }
}

/// `backend` named for `phase`, at its latest revision.
fn named(phase: Option<Phase>, backend: &str) -> Preference {
    Preference {
        phase,
        backend: backend.to_owned(),
        revision: None,
    }
}

/// Revision `revision` of `backend`'s package of [`OP`].
fn package(backend: &str, revision: u32) -> Package {
    Package {
        op: OP.to_owned(),
        backend: backend.to_owned(),
        revision,
    }
}

/// The entry of a `cover` response that selects `backend`'s `revision` for
/// `keys`.
fn covered(backend: &str, revision: u32, keys: &[&str]) -> (Package, Vec<String>) {
    (
        package(backend, revision),
        keys.iter().map(|key| (*key).to_owned()).collect(),
    )
}

/// Two packages declaring disjoint descriptors each get their own keys, and a
/// key neither declares is named as uncovered: the response a client reads
/// over the socket after installing both.
#[test]
fn each_key_goes_to_the_package_that_declares_it_over_the_socket() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let serving = support::serve(server, &temporary.path().join("build-server.sock"));
    let mut client = serving.client();

    let f32 = support::vendor_package(
        &temporary.path().join("f32"),
        &manifest("f32", 0, &["f32"], &arch()),
        &[],
    );
    let fa4 = support::vendor_package(
        &temporary.path().join("fa4"),
        &manifest("fa4", 0, &["bf16"], &arch()),
        &[],
    );
    client.install(&f32).unwrap();
    client.install(&fa4).unwrap();

    let coverage = client.cover(&request(&[BF16, F32, F16, BF16])).unwrap();
    // `f32` sorts before `fa4` at the same priority, so it is listed first,
    // and a key asked for twice is selected once.
    assert_eq!(
        coverage.covered,
        vec![covered("f32", 1, &[F32]), covered("fa4", 1, &[BF16])]
    );
    assert_eq!(coverage.uncovered, vec![F16.to_owned()]);
}

/// A key several packages declare goes to the one of highest priority, and
/// among equal priorities to the backend whose name sorts first: a later
/// revision of another backend does not outrank them.
#[test]
fn a_key_several_packages_cover_goes_to_the_highest_priority() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let sources = temporary.path();
    install(
        &server,
        sources,
        &manifest("beta", 2, &["bf16", "f16"], &arch()),
    );
    install(
        &server,
        sources,
        &manifest("alpha", 0, &["bf16", "f32"], &arch()),
    );
    for _ in 0..3 {
        install(&server, sources, &manifest("gamma", 0, &["f32"], &arch()));
    }

    let coverage = server.cover(&request(&[BF16, F16, F32])).unwrap();
    assert_eq!(
        coverage.covered,
        vec![
            covered("beta", 1, &[BF16, F16]),
            covered("alpha", 1, &[F32]),
        ]
    );
    assert!(coverage.uncovered.is_empty());
}

/// A named backend takes every key it covers before any priority, and a key
/// it does not cover goes on to the packages that do.
#[test]
fn a_named_backend_takes_the_keys_it_covers() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let sources = temporary.path();
    install(
        &server,
        sources,
        &manifest("alpha", 5, &["bf16", "f32"], &arch()),
    );
    install(&server, sources, &manifest("beta", 0, &["bf16"], &arch()));

    let mut asked = request(&[BF16, F32]);
    asked.preferred = vec![named(None, "beta")];
    let coverage = server.cover(&asked).unwrap();
    assert_eq!(
        coverage.covered,
        vec![covered("alpha", 1, &[F32]), covered("beta", 1, &[BF16])]
    );
}

/// A backend named for one phase is preferred for that phase's keys alone.
#[test]
fn a_backend_named_for_a_phase_takes_that_phase() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let phased = |backend: &str| {
        support::vendor_manifest(
            OP,
            backend,
            0,
            r#"{ phase = ["prefill", "decode"], dtype = ["bf16"], head_dim = ["128"] }"#,
            &arch(),
        )
    };
    install(&server, temporary.path(), &phased("alpha"));
    install(&server, temporary.path(), &phased("beta"));

    let prefill = "phase_prefill_dtype_bf16_head_dim_128";
    let decode = "phase_decode_dtype_bf16_head_dim_128";
    let mut asked = request(&[prefill, decode]);
    asked.preferred = vec![named(Some(Phase::Decode), "beta")];
    let coverage = server.cover(&asked).unwrap();
    assert_eq!(
        coverage.covered,
        vec![
            covered("alpha", 1, &[prefill]),
            covered("beta", 1, &[decode])
        ]
    );
}

/// Only a backend's latest revision is a candidate, as `prepare` selects it:
/// a key an earlier revision declared and the latest does not is uncovered.
#[test]
fn only_the_latest_revision_of_a_backend_covers() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let sources = temporary.path();
    install(
        &server,
        sources,
        &manifest("fa4", 0, &["bf16", "f16"], &arch()),
    );
    install(&server, sources, &manifest("fa4", 0, &["bf16"], &arch()));

    let coverage = server.cover(&request(&[BF16, F16])).unwrap();
    assert_eq!(coverage.covered, vec![covered("fa4", 2, &[BF16])]);
    assert_eq!(coverage.uncovered, vec![F16.to_owned()]);
}

/// The response lists every candidate in selection order, which is what a
/// module set fixes; and a request that states a module set's revisions
/// selects among exactly those: a revision or a package installed since is no
/// candidate, and a fixed revision the server lacks is refused.
#[test]
fn a_module_sets_revisions_are_the_only_candidates() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    let sources = temporary.path();
    install(&server, sources, &manifest("fa4", 0, &["bf16"], &arch()));

    // What the daemon asks when a configuration is applied: no key, every
    // candidate.
    let applied = server.cover(&request(&[])).unwrap();
    assert_eq!(applied.candidates, vec![package("fa4", 1)]);

    // Installed after the set was fixed: a later revision of `fa4`, and
    // another package.
    install(
        &server,
        sources,
        &manifest("fa4", 0, &["bf16", "f16"], &arch()),
    );
    install(&server, sources, &manifest("f32", 0, &["f32"], &arch()));
    let latest = server.cover(&request(&[BF16, F16, F32])).unwrap();
    assert_eq!(
        latest.candidates,
        vec![package("f32", 1), package("fa4", 2)]
    );
    assert!(latest.uncovered.is_empty());

    let mut fixed = request(&[BF16, F16, F32]);
    fixed.fixed = vec![("fa4".to_owned(), 1)];
    let coverage = server.cover(&fixed).unwrap();
    assert_eq!(coverage.candidates, vec![package("fa4", 1)]);
    assert_eq!(coverage.covered, vec![covered("fa4", 1, &[BF16])]);
    assert_eq!(coverage.uncovered, vec![F16.to_owned(), F32.to_owned()]);

    fixed.fixed = vec![("fa4".to_owned(), 3)];
    let refused = server.cover(&fixed).unwrap_err();
    assert!(refused.to_string().contains("revision 3"), "{refused}");
}

/// A package that does not build for the requested target covers nothing
/// there, whatever its descriptors declare.
#[test]
fn a_package_for_another_target_covers_nothing() {
    let _exclusive = support::exclusive();
    let temporary = tempfile::tempdir().unwrap();
    let server = BuildServer::open(&support::config(temporary.path())).unwrap();
    install(
        &server,
        temporary.path(),
        &manifest("fa4", 0, &["bf16"], &other_arch()),
    );

    let coverage = server.cover(&request(&[BF16])).unwrap();
    assert!(coverage.covered.is_empty());
    assert!(coverage.candidates.is_empty());
    assert_eq!(coverage.uncovered, vec![BF16.to_owned()]);
}
