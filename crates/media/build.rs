//! The media crate's native dependency: the FFmpeg libraries the C shim (`src/ffi/shim.c`) is
//! compiled against and links.
//!
//! The shim is written for one FFmpeg release line, 8.1, and the build requires exactly the
//! library versions of that line ([`LIBRARIES`]). They are found through pkg-config, the standard
//! description of an installed C library: a distribution package, or an installation whose
//! `lib/pkgconfig` the host registers with pkg-config. With the `vendored` feature the build
//! first builds the pinned FFmpeg 8.1.2 and libx264 from their release sources into its output
//! directory ([`vendored`]) and finds that installation through its own pkg-config files, so a
//! host whose system lacks FFmpeg 8.1 still builds.
//!
//! The package declares `links = "ffmpeg"` and states two values to the build scripts of the
//! packages that depend on it directly:
//!
//! - `DEP_FFMPEG_LIB`: the directories that hold the libraries, in the platform's path-list
//!   syntax, when pkg-config names any. pkg-config names none for libraries on the default loader
//!   path, which every executable finds without a run path.
//! - `DEP_FFMPEG_VERSION`: the version of `libavcodec` the build links.
//!
//! An executable's package adds `DEP_FFMPEG_LIB` to its run path, because Cargo passes a
//! dependency's search path and library names on to its dependents but not its link arguments.
//! The run path also lets the linker resolve the libraries the FFmpeg libraries themselves need.

use std::path::PathBuf;

/// A build step's failure, which Cargo reports as the build script's error.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The pkg-config modules whose headers the shim includes, with each one's version in FFmpeg 8.1,
/// in link order: every library before the libraries it calls.
const LIBRARIES: [(&str, &str); 6] = [
    ("libavformat", "62.12.102"),
    ("libavfilter", "11.14.102"),
    ("libavcodec", "62.28.102"),
    ("libswscale", "9.5.102"),
    ("libswresample", "6.3.102"),
    ("libavutil", "60.26.102"),
];

/// The variables pkg-config reads to find a module; a change of any of them can find another
/// installation.
const PKG_CONFIG_VARIABLES: [&str; 3] = [
    "PKG_CONFIG_PATH",
    "PKG_CONFIG_LIBDIR",
    "PKG_CONFIG_SYSROOT_DIR",
];

/// What pkg-config reports of the FFmpeg installation the build links.
struct Installation {
    /// Directories of the headers, in the order pkg-config states them.
    include_paths: Vec<PathBuf>,
    /// Directories of the libraries, outside the default loader path.
    link_paths: Vec<PathBuf>,
    /// The libraries to link, in link order.
    libraries: Vec<String>,
    /// The version of `libavcodec`.
    version: String,
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/ffi/shim.c");
    for variable in PKG_CONFIG_VARIABLES {
        println!("cargo::rerun-if-env-changed={variable}");
    }

    if std::env::var_os("CARGO_FEATURE_VENDORED").is_some() {
        let out = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("Cargo did not set OUT_DIR")?);
        let prefix = vendored::install(&out)?;
        // The installation's own pkg-config files describe it, and the search is limited to
        // them, so an FFmpeg of the system cannot stand in for the one just built.
        //
        // SAFETY: the build script runs on one thread and starts none.
        unsafe {
            std::env::set_var("PKG_CONFIG_LIBDIR", prefix.join("lib").join("pkgconfig"));
            std::env::remove_var("PKG_CONFIG_PATH");
        }
    }

    let installation = probe()?;

    // The shim is compiled before the libraries are named, so that the linker sees the shim's
    // references to them first: libraries are linked as needed, and one named before anything
    // refers to it would be dropped.
    cc::Build::new()
        .file("src/ffi/shim.c")
        .includes(&installation.include_paths)
        .warnings(true)
        .compile("uniserve_media_shim");

    for path in &installation.link_paths {
        println!("cargo::rustc-link-search=native={}", path.display());
    }
    for library in &installation.libraries {
        println!("cargo::rustc-link-lib=dylib={library}");
    }

    // This package's own tests run against the libraries too; Cargo applies these arguments to
    // the package's own targets only, and dependents read the same directories from the metadata.
    for path in &installation.link_paths {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
    if !installation.link_paths.is_empty() {
        let joined = std::env::join_paths(&installation.link_paths)
            .map_err(|error| format!("joining the FFmpeg library directories: {error}"))?;
        println!("cargo::metadata=lib={}", joined.to_string_lossy());
    }
    println!("cargo::metadata=version={}", installation.version);
    Ok(())
}

/// Finds every module of [`LIBRARIES`] at its exact version.
///
/// # Errors
///
/// When a module is missing or has another version, naming the provisioning the build needs.
fn probe() -> Result<Installation> {
    let mut installation = Installation {
        include_paths: Vec::new(),
        link_paths: Vec::new(),
        libraries: Vec::new(),
        version: String::new(),
    };
    for (module, version) in LIBRARIES {
        let library = pkg_config::Config::new()
            .exactly_version(version)
            .cargo_metadata(false)
            .env_metadata(true)
            .probe(module)
            .map_err(|error| {
                format!(
                    "the media crate links FFmpeg 8.1 ({module} {version}), which pkg-config \
                     does not provide: {error}\n\
                     Install FFmpeg 8.1 with libx264 as shared libraries where pkg-config finds \
                     them, or build with the `vendored` feature, which builds the pinned FFmpeg \
                     8.1.2 and libx264 from source (crates/media/README.md)."
                )
            })?;
        for path in library.include_paths {
            if !installation.include_paths.contains(&path) {
                installation.include_paths.push(path);
            }
        }
        for path in library.link_paths {
            if !installation.link_paths.contains(&path) {
                installation.link_paths.push(path);
            }
        }
        for name in library.libs {
            if !installation.libraries.contains(&name) {
                installation.libraries.push(name);
            }
        }
        if module == "libavcodec" {
            installation.version = library.version;
        }
    }
    Ok(installation)
}

/// The pinned FFmpeg 8.1.2 and libx264, built from their release sources into the build's output
/// directory (the `vendored` feature).
///
/// The recipe is one value, [`Recipe`]: the source archives, each pinned by its SHA-256, the
/// compiler and every configure argument. Its digest names the installation's prefix, and the
/// prefix is used only once it holds the completion record with that digest, written after the
/// last install step: an interrupted build, or an installation of another recipe, is never taken
/// for this one.
///
/// Each archive is fetched once into the output directory with `curl` and verified before it is
/// unpacked. Both projects are configured and built with their own build systems (`configure`,
/// `make`), which take their parallelism from Cargo's jobserver (`CARGO_MAKEFLAGS`), so the build
/// uses no more jobs than Cargo was given. The host tools this needs are `curl`, `tar` with `xz`,
/// `make` and a C compiler, and `nasm` on x86-64, where both projects assemble their kernels with
/// it.
///
/// Both are built as shared libraries, as the crate links any FFmpeg. Every FFmpeg library is
/// linked with `$ORIGIN` as its run path, so it resolves `libx264` and the other FFmpeg libraries
/// beside itself wherever the prefix is: the prefix needs no loader configuration, and an
/// executable needs only the prefix's `lib` in its own run path. The run path reaches the linker
/// through a response file ([`RUN_PATH`]): FFmpeg's configure re-evaluates its arguments in the
/// shell and make expands `$` again, so a `$ORIGIN` on the command line would not survive.
mod vendored {
    use std::fs;
    use std::os::fd::AsFd as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    use sha2::{Digest as _, Sha256};

    use super::Result;

    /// A source archive the recipe builds from.
    struct Source {
        /// The archive's URL.
        url: &'static str,
        /// The archive's SHA-256, in hexadecimal.
        sha256: &'static str,
        /// The file the archive is stored as.
        file: &'static str,
        /// The directory the archive unpacks to.
        directory: &'static str,
    }

    /// libx264 at upstream commit `0480cb05`, the source of the encoder's build 165 (`libx264.so.165`),
    /// as VideoLAN's repository archives that commit.
    const X264: Source = Source {
        url: "https://code.videolan.org/videolan/x264/-/archive/0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee/x264-0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee.tar.gz",
        sha256: "d0967a1348c85dfde363bb52610403be898171493100561efa0dd05d5fd1ae50",
        file: "x264-0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee.tar.gz",
        directory: "x264-0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee",
    };

    /// The FFmpeg 8.1.2 release.
    const FFMPEG: Source = Source {
        url: "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz",
        sha256: "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c",
        file: "ffmpeg-8.1.2.tar.xz",
        directory: "ffmpeg-8.1.2",
    };

    /// The name of the completion record in a finished prefix.
    const RECORD: &str = "recipe.sha256";

    /// The linker options every FFmpeg library is linked with, read by the linker from a response
    /// file (`@file`): the library's own directory as its run path.
    const RUN_PATH: &str = "-rpath=$ORIGIN\n";

    /// Everything the installation depends on.
    struct Recipe {
        /// The C compiler both projects build with.
        compiler: PathBuf,
        /// libx264's configure arguments, the prefix excluded.
        x264: Vec<String>,
        /// FFmpeg's configure arguments, the prefix and the compiler excluded.
        ffmpeg: Vec<String>,
    }

    impl Recipe {
        /// The recipe built with `compiler`, whose FFmpeg libraries read [`RUN_PATH`] from the
        /// response file `run_path`.
        fn new(compiler: PathBuf, run_path: &Path) -> Self {
            let arguments =
                |list: &[&str]| list.iter().map(|argument| (*argument).to_owned()).collect();
            Self {
                compiler,
                x264: arguments(&["--enable-shared", "--enable-pic", "--disable-cli"]),
                ffmpeg: arguments(&[
                    "--enable-shared",
                    "--disable-static",
                    // libx264 is GPL, so an FFmpeg that calls it is configured as one.
                    "--enable-gpl",
                    "--enable-version3",
                    "--enable-libx264",
                    // The crate calls the libraries; the command-line programs and capture devices
                    // are not built.
                    "--disable-programs",
                    "--disable-avdevice",
                    "--disable-doc",
                    "--disable-debug",
                    // Only the external libraries named here are linked, whatever the host has.
                    "--disable-autodetect",
                    "--enable-zlib",
                ])
                .into_iter()
                .chain([format!("--extra-ldsoflags=-Wl,@{}", run_path.display())])
                .collect(),
            }
        }

        /// The recipe's digest in hexadecimal, which names the prefix.
        fn digest(&self) -> String {
            let mut hasher = Sha256::new();
            for source in [&X264, &FFMPEG] {
                hasher.update(source.sha256.as_bytes());
                hasher.update([0]);
            }
            hasher.update(self.compiler.as_os_str().as_encoded_bytes());
            hasher.update([0]);
            hasher.update(RUN_PATH.as_bytes());
            hasher.update([0]);
            for argument in self.x264.iter().chain([&String::new()]).chain(&self.ffmpeg) {
                hasher.update(argument.as_bytes());
                hasher.update([0]);
            }
            hex(&hasher.finalize())
        }
    }

    /// Installs the recipe's FFmpeg under `out` unless a finished installation of it is there, and
    /// returns its prefix.
    ///
    /// # Errors
    ///
    /// When the build targets another machine than the host, a host tool is missing, an archive
    /// cannot be fetched or does not match its pin, or a configure, build or install step fails.
    pub(crate) fn install(out: &Path) -> Result<PathBuf> {
        let target = std::env::var("TARGET").map_err(|_| "Cargo did not set TARGET")?;
        let host = std::env::var("HOST").map_err(|_| "Cargo did not set HOST")?;
        if target != host {
            return Err(format!(
                "the vendored FFmpeg is built for the host ({host}) only; a build for {target} \
                 needs an FFmpeg 8.1 for that target through pkg-config"
            )
            .into());
        }

        let compiler = cc::Build::new().get_compiler().path().to_owned();
        let run_path = out.join("run-path.rsp");
        let recipe = Recipe::new(compiler, &run_path);
        let digest = recipe.digest();
        let prefix = out.join(format!("ffmpeg-{}", &digest[..16]));
        if fs::read_to_string(prefix.join(RECORD)).is_ok_and(|record| record == digest) {
            return Ok(prefix);
        }
        if prefix.exists() {
            fs::remove_dir_all(&prefix)
                .map_err(|error| format!("removing an unfinished FFmpeg installation: {error}"))?;
        }

        let downloads = out.join("downloads");
        let sources = out.join("sources");
        fs::create_dir_all(&downloads)
            .map_err(|error| format!("creating the source archive directory: {error}"))?;
        if sources.exists() {
            fs::remove_dir_all(&sources)
                .map_err(|error| format!("removing the previous source trees: {error}"))?;
        }
        fs::create_dir_all(&sources)
            .map_err(|error| format!("creating the source tree directory: {error}"))?;
        for source in [&X264, &FFMPEG] {
            let archive = fetch(source, &downloads)?;
            run(
                Command::new("tar")
                    .arg("--extract")
                    .arg("--file")
                    .arg(&archive)
                    .arg("--directory")
                    .arg(&sources),
                source.file,
            )?;
        }

        fs::write(&run_path, RUN_PATH)
            .map_err(|error| format!("writing the FFmpeg libraries' linker options: {error}"))?;
        let prefix_argument = format!("--prefix={}", prefix.display());
        let x264 = sources.join(X264.directory);
        run(
            Command::new(x264.join("configure"))
                .current_dir(&x264)
                .arg(&prefix_argument)
                .args(&recipe.x264)
                .env("CC", &recipe.compiler),
            "configuring libx264",
        )?;
        make(&x264, "building libx264")?;

        // FFmpeg's configure finds libx264 through the pkg-config file libx264 just installed.
        let ffmpeg = sources.join(FFMPEG.directory);
        run(
            Command::new(ffmpeg.join("configure"))
                .current_dir(&ffmpeg)
                .arg(&prefix_argument)
                .arg(format!("--cc={}", recipe.compiler.display()))
                .args(&recipe.ffmpeg)
                .env("PKG_CONFIG_PATH", prefix.join("lib").join("pkgconfig"))
                .env_remove("PKG_CONFIG_LIBDIR"),
            "configuring FFmpeg",
        )?;
        make(&ffmpeg, "building FFmpeg")?;

        fs::remove_dir_all(&sources)
            .map_err(|error| format!("removing the built source trees: {error}"))?;
        fs::write(prefix.join(RECORD), &digest)
            .map_err(|error| format!("recording the finished FFmpeg installation: {error}"))?;
        Ok(prefix)
    }

    /// The verified archive of `source` in `downloads`, fetched unless a verified copy is there.
    fn fetch(source: &Source, downloads: &Path) -> Result<PathBuf> {
        let archive = downloads.join(source.file);
        if archive.exists() && sha256(&archive)? == source.sha256 {
            return Ok(archive);
        }
        let partial = downloads.join(format!("{}.partial", source.file));
        run(
            Command::new("curl")
                .args([
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--output",
                ])
                .arg(&partial)
                .arg(source.url),
            source.url,
        )?;
        let found = sha256(&partial)?;
        if found != source.sha256 {
            return Err(format!(
                "{} has SHA-256 {found}, not the pinned {}",
                source.url, source.sha256
            )
            .into());
        }
        fs::rename(&partial, &archive)
            .map_err(|error| format!("placing a verified source archive: {error}"))?;
        Ok(archive)
    }

    /// Builds and installs the project configured in `directory`.
    fn make(directory: &Path, what: &str) -> Result<()> {
        for target in [None, Some("install")] {
            let mut command = Command::new("make");
            command.current_dir(directory).args(target);
            // Cargo's jobserver: make runs as many jobs as Cargo grants the build.
            if let Some(flags) = std::env::var_os("CARGO_MAKEFLAGS") {
                command.env("MAKEFLAGS", flags);
            }
            run(&mut command, what)?;
        }
        Ok(())
    }

    /// Runs `command`, its output on the build's diagnostic stream, and fails naming `what` when
    /// it cannot start or fails.
    fn run(command: &mut Command, what: &str) -> Result<()> {
        // Cargo reads the build script's standard output as instructions; the tools write to its
        // standard error, which Cargo keeps as the build's log.
        let diagnostic = std::io::stderr()
            .as_fd()
            .try_clone_to_owned()
            .map(Stdio::from)
            .map_err(|error| format!("duplicating the build's standard error: {error}"))?;
        let status = command
            .stdout(diagnostic)
            .status()
            .map_err(|error| format!("{what}: cannot run {:?}: {error}", command.get_program()))?;
        if !status.success() {
            return Err(format!("{what}: {command:?} failed with {status}").into());
        }
        Ok(())
    }

    /// The SHA-256 of `file` in hexadecimal.
    fn sha256(file: &Path) -> Result<String> {
        let bytes =
            fs::read(file).map_err(|error| format!("reading {}: {error}", file.display()))?;
        Ok(hex(&Sha256::digest(bytes)))
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
