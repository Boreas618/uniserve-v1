//! Where a model's files are: a directory, or the snapshot of a Hugging Face repository in the local
//! Hub cache, which is how `uniserve serve` finds its model too.
//!
//! The reference engines name a test or example model by its repository id (vLLM's model registry,
//! SGLang's `DEFAULT_MODEL_NAME_FOR_TEST`) and read it through the Hub cache that `HF_HOME` locates.
//! Nothing here downloads: a repository the cache does not hold is an error that says how to fetch
//! it.

use std::path::{Path, PathBuf};

use hf_hub::{Cache, Repo, RepoType};

/// The directory of `model`: the directory itself when one exists at that path; otherwise the
/// snapshot the local Hub cache holds of the repository `model` names, at `main` or at the
/// revision (a branch, a tag or a commit) after an `@`.
pub fn model_dir(model: &str) -> Result<PathBuf, String> {
    let path = Path::new(model);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let (repo_id, revision) = model.split_once('@').unwrap_or((model, "main"));
    if repo_id.split('/').count() != 2 || repo_id.starts_with('/') {
        return Err(format!(
            "{model} is neither a directory nor a Hugging Face repository id"
        ));
    }
    let cache = Cache::from_env();
    let repo = cache.repo(Repo::with_revision(
        repo_id.to_owned(),
        RepoType::Model,
        revision.to_owned(),
    ));
    // A commit names its snapshot directly; a branch or a tag names it through its ref.
    let commit = revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit());
    if commit {
        let snapshot = repo.pointer_path(revision);
        if snapshot.is_dir() {
            return Ok(snapshot);
        }
    } else {
        for marker in ["config.json", "model_index.json"] {
            if let Some(dir) = repo
                .get(marker)
                .and_then(|file| file.parent().map(Path::to_path_buf))
            {
                return Ok(dir);
            }
        }
    }
    Err(format!(
        "the Hugging Face cache at {} holds no snapshot of {repo_id} at {revision}: \
         `huggingface-cli download {repo_id} --revision {revision}` fetches it",
        cache.path().display()
    ))
}
