//! Locate `libtvm_ffi` in the environment this build runs in.
//!
//! `tvm-ffi-sys` records the link search path once and is not rebuilt when
//! `PATH` selects another installation, so a package build in a fresh
//! isolated environment would link against a directory that no longer
//! exists. This script reruns on `PATH` changes and names the current
//! directory for the link.
//!
//! Executables built here also need a runtime path. When Python loads the
//! worker library, `libtvm_ffi` is already loaded by `tvm_ffi` and resolved
//! by soname, so the runtime path only matters for standalone executables.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PATH");
    let Ok(output) = Command::new("tvm-ffi-config").arg("--libdir").output() else {
        return;
    };
    let libdir = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && !libdir.is_empty() {
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}
