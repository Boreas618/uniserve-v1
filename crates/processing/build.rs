//! Sizes the stack of the processing program's wasm32 module.
//!
//! A chat template recurses as it renders (macros, loops over nested JSON), up to minijinja's
//! own recursion bound, and the module's stack is the region of its linear memory below its
//! data: Rust's default of one mebibyte holds too few of minijinja's frames. The module's stack
//! is therefore 16 MiB, within its memory budget; a native build keeps its threads' stacks.

/// The bytes of the module's stack.
const STACK_BYTES: u32 = 16 << 20;

fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "wasm32" {
        println!("cargo:rustc-link-arg-cdylib=-zstack-size={STACK_BYTES}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
