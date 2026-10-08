//! Locate `libtvm_ffi` for Rust test and bench executables.
//!
//! `tvm-ffi-sys` adds the link search path; executables built here also need
//! a runtime path. When Python loads the worker library, `libtvm_ffi` is
//! already loaded by `tvm_ffi` and resolved by soname, so the runtime path
//! only matters for standalone executables.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PATH");
    let Ok(output) = Command::new("tvm-ffi-config").arg("--libdir").output() else {
        return;
    };
    let libdir = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && !libdir.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}
