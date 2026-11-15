//! Released checkpoints a test reads, named by their Hugging Face repository and commit.
//!
//! A checkpoint is resolved into the standard Hugging Face cache (`HF_HOME`) the serving path and
//! the evaluation share: a host fetches what its cache lacks from the hub (`HF_ENDPOINT`, and the
//! token the cache holds or `HF_TOKEN`), and an offline host (`HF_HUB_OFFLINE`) provides its
//! cache beforehand and fetches nothing. Tests that read one are ignored by default: the
//! repository holds none of them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};

/// A released model checkpoint at one commit of its Hugging Face repository.
#[derive(Clone, Copy, Debug)]
pub struct Release {
    /// The repository, `owner/name`.
    pub repo: &'static str,
    /// The commit the test reads, in full.
    pub revision: &'static str,
}

/// The FastH3 export of four denoising steps with sparse video attention.
pub const FAST_H3_4_STEP: Release = Release {
    repo: "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
    revision: "5ea076f35b84da4c3c82217112fa733d8eea2ae1",
};

/// The FastH3 export of eight denoising steps.
pub const FAST_H3_8_STEP: Release = Release {
    repo: "FastVideo/FastVideo-FastH3-8-Step-V2",
    revision: "3da2ddfe1954d9cda4c05b643dc0f26007a655c5",
};

/// The base MiniMax-H3 release, whose dense denoiser any Ulysses degree splits.
pub const MINIMAX_H3: Release = Release {
    repo: "MiniMaxAI/MiniMax-H3",
    revision: "9bfb6693f2cf6de171db46d1aa586f67d773a1da",
};

/// Qwen3-30B-A3B, a text mixture-of-experts model.
pub const QWEN3_30B_A3B: Release = Release {
    repo: "Qwen/Qwen3-30B-A3B",
    revision: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
};

/// Qwen3.6-35B-A3B, a hybrid mixture-of-experts model with a vision tower.
pub const QWEN3_6_35B_A3B: Release = Release {
    repo: "Qwen/Qwen3.6-35B-A3B",
    revision: "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
};

impl Release {
    /// The checkpoint's directory: its snapshot in the Hugging Face cache, holding every file of
    /// the commit. A host that is not offline fetches the files its cache lacks first, once per
    /// test process.
    ///
    /// # Panics
    ///
    /// When an offline host's cache holds no snapshot of the commit, or the hub refuses or fails
    /// a listing or a file.
    #[expect(
        clippy::unwrap_used,
        reason = "test support fails the test that reads it"
    )]
    pub fn snapshot(&self) -> PathBuf {
        // The snapshots this process resolved, by repository and commit: a test that names a
        // checkpoint several times lists the hub once.
        static RESOLVED: Mutex<BTreeMap<(&str, &str), PathBuf>> = Mutex::new(BTreeMap::new());
        let mut resolved = RESOLVED.lock().unwrap();
        if let Some(snapshot) = resolved.get(&(self.repo, self.revision)) {
            return snapshot.clone();
        }

        let repo = Repo::with_revision(
            self.repo.to_owned(),
            RepoType::Model,
            self.revision.to_owned(),
        );
        let cache = Cache::from_env();
        let snapshot = cache.repo(repo.clone()).pointer_path(self.revision);

        if !offline() {
            let api = ApiBuilder::from_env()
                .with_progress(false)
                .build()
                .unwrap()
                .repo(repo);
            for file in api.info().unwrap().siblings {
                api.get(&file.rfilename).unwrap();
            }
        }

        assert!(
            snapshot.is_dir(),
            "the Hugging Face cache holds no snapshot of {} at {} ({})",
            self.repo,
            self.revision,
            snapshot.display()
        );
        resolved.insert((self.repo, self.revision), snapshot.clone());
        snapshot
    }
}

/// Whether the host reads the hub's files from its cache alone, as `huggingface_hub` reads
/// `HF_HUB_OFFLINE`: `1`, `ON`, `YES` or `TRUE`, in any case.
fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        ["1", "ON", "YES", "TRUE"].contains(&value.to_ascii_uppercase().as_str())
    })
}
