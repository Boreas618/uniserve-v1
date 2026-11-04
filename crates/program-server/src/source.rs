//! A package's source: how its files become a module (design sections 5.2,
//! 5.3 and 6.2).
//!
//! The Program Server drives six kinds of source:
//!
//! - [`CudaSource`]: CUDA C++ through nvcc, to a cubin (with the launch
//!   metadata the package writes in Triton's fields) or, when the package
//!   brings a C ABI layer, to a native library.
//! - [`TritonSource`]: Triton through `triton.compile`, which writes a cubin
//!   and its metadata JSON. Triton's own AOT tool cannot express clusters,
//!   PDL or scratch, so it is not used (design section 5.3).
//! - [`CuteDslSource`]: CuTe DSL through its Python compiler and
//!   `export_to_c`, linked with `libcuda_dialect_runtime_static.a` into a
//!   native library.
//! - [`CubinSource`]: a vendor cubin the package carries, such as
//!   trtllm-gen's launch code and cubins, stored as it is.
//! - [`WasmSource`]: a scheduling policy built for wasm32 and precompiled for
//!   this host's CPU.
//! - [`ExecutableSource`]: an executable program, stored as it is.
//!
//! A source builds one descriptor into a [`Workspace`] and returns the names
//! of the files it wrote; the store publishes them (design section 5.3). The
//! descriptor key is passed through untouched: the package's compile script
//! reads the parameter values out of it and picks the kernel.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::manifest::{Manifest, ModuleForm, SourceKind, ToolchainKind};
use crate::toolchain::Toolchain;
use crate::{Error, Result};

/// The files a build wrote, relative to its [`Workspace`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built {
    pub files: Vec<String>,
}

/// The directory one build writes into. It is private to the build and is
/// deleted when the build ends, after the store has published the module.
#[derive(Debug)]
pub struct Workspace {
    directory: tempfile::TempDir,
}

impl Workspace {
    pub fn new() -> crate::Result<Workspace> {
        Ok(Workspace {
            directory: tempfile::tempdir()?,
        })
    }

    pub fn path(&self) -> &Path {
        self.directory.path()
    }
}

/// A program package's source: it builds one descriptor into a module.
pub trait Source: Send + Sync {
    /// The toolchain this source compiles with.
    fn toolchain(&self) -> ToolchainKind;

    /// Builds the module of one descriptor.
    ///
    /// `op` and `key` are the operator name and the descriptor key of the
    /// module's path (design section 5.3); `dir` is the workspace to write
    /// into. The returned file names, and the module's form from the
    /// manifest, are what the store publishes.
    fn build(
        &self,
        op: &str,
        key: &str,
        toolchain: &dyn Toolchain,
        dir: &Workspace,
    ) -> Result<Built>;
}

/// Makes the source a package's manifest names.
pub fn open(manifest: &Manifest, root: &Path) -> Result<Box<dyn Source>> {
    let package = root.to_path_buf();
    Ok(match manifest.source.kind {
        SourceKind::Cuda => Box::new(CudaSource::new(package, manifest.clone())),
        SourceKind::Triton => Box::new(TritonSource::new(package, manifest.clone())),
        SourceKind::CuteDsl => Box::new(CuteDslSource::new(package, manifest.clone())),
        SourceKind::Cubin => Box::new(CubinSource::new(package, manifest.clone())),
        SourceKind::Wasm => Box::new(WasmSource::new(package, manifest.clone())),
        SourceKind::Executable => Box::new(ExecutableSource::new(package, manifest.clone())),
    })
}

/// What every source shares: where the package is and what it declares.
///
/// A source is built for exactly one package revision, so the manifest it
/// carries is fixed for its whole life.
#[derive(Clone, Debug)]
pub struct PackageSource {
    pub root: PathBuf,
    pub manifest: Manifest,
}

impl PackageSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self { root, manifest }
    }

    /// The module form the package's manifest names.
    pub fn form(&self) -> ModuleForm {
        self.manifest.module.form
    }

    /// A file of the package, by the manifest's relative path.
    pub fn file(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    /// The compile script the package names, or fails.
    pub fn script(&self) -> Result<PathBuf> {
        let script = self
            .manifest
            .source
            .script
            .as_deref()
            .ok_or_else(|| Error::Build {
                op: self.manifest.op.clone(),
                backend: self.manifest.backend.clone(),
                revision: 0,
                target: String::new(),
                descriptor: String::new(),
                message: format!(
                    "a {:?} package needs a compile script",
                    self.manifest.source.kind
                ),
            })?;
        Ok(self.file(script))
    }

    /// The environment the Program Server passes to a compile script: the
    /// package root, the descriptor key, the output directory and the
    /// toolchain's own variables.
    pub fn environment(&self, key: &str, dir: &Workspace) -> Vec<(OsString, OsString)> {
        vec![
            ("UNISERVE_OP".into(), self.manifest.op.clone().into()),
            (
                "UNISERVE_BACKEND".into(),
                self.manifest.backend.clone().into(),
            ),
            ("UNISERVE_DESCRIPTOR".into(), key.into()),
            (
                "UNISERVE_PACKAGE".into(),
                self.root.clone().into_os_string(),
            ),
            ("UNISERVE_OUT".into(), dir.path().as_os_str().to_owned()),
        ]
    }
}

/// CUDA C++ through nvcc (design section 5.3).
#[derive(Clone, Debug)]
pub struct CudaSource {
    package: PackageSource,
}

impl CudaSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CudaSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the nvcc source: compile the package's CUDA C++ into a cubin or a native library")
    }
}

/// Triton through `triton.compile` (design section 5.3).
#[derive(Clone, Debug)]
pub struct TritonSource {
    package: PackageSource,
}

impl TritonSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for TritonSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Python
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the Triton source: drive triton.compile through the python environment")
    }
}

/// CuTe DSL through `export_to_c`, linked into a native library (design
/// sections 5.3 and 9).
#[derive(Clone, Debug)]
pub struct CuteDslSource {
    package: PackageSource,
}

impl CuteDslSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CuteDslSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Python
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the CuTe DSL source: export_to_c, then link libcuda_dialect_runtime_static.a")
    }
}

/// A vendor cubin the package carries (design section 5.3).
#[derive(Clone, Debug)]
pub struct CubinSource {
    package: PackageSource,
}

impl CubinSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for CubinSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the cubin source: copy the package's cubin and metadata into the workspace")
    }
}

/// A scheduling policy: wasm32, then precompiled for this host's CPU
/// (design section 5.3).
#[derive(Clone, Debug)]
pub struct WasmSource {
    package: PackageSource,
}

impl WasmSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for WasmSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Wasm
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the wasm source: build wasm32, then precompile it with wasmtime")
    }
}

/// An executable program, stored as it is (design section 5.1).
#[derive(Clone, Debug)]
pub struct ExecutableSource {
    package: PackageSource,
}

impl ExecutableSource {
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self {
            package: PackageSource::new(root, manifest),
        }
    }
}

impl Source for ExecutableSource {
    fn toolchain(&self) -> ToolchainKind {
        ToolchainKind::Nvcc
    }

    fn build(
        &self,
        _op: &str,
        _key: &str,
        _toolchain: &dyn Toolchain,
        _dir: &Workspace,
    ) -> Result<Built> {
        todo!("the executable source: store the package's binary as it is")
    }
}
