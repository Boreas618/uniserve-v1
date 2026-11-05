//! The package headers' mirrors checked against the layout the operator passes.
//!
//! A native operator's call structure is a `#[repr(C)]` struct in this crate and a C struct in each
//! package that serves it, and the two are written by hand. The layout is therefore stated twice -
//! and, where an operator declares a `CallLayout`, three times, because the offsets are also written
//! as literals next to the struct. Nothing in the compiler holds the C side to the Rust side: a header
//! left behind at a struct change compiles cleanly, loads cleanly under `dlopen(RTLD_NOW)` because the
//! symbol is there, and reads every extent from the wrong offset at run time. That is a wrong number
//! and not a refusal, which is why the mirror is checked here rather than left to a run to reveal.
//!
//! The check is on the header's own `static_assert`s: those are the statements the package's build
//! enforces, so agreeing with them is agreeing with the module that will be loaded.

use std::path::{Path, PathBuf};

/// What a package's header says about one call structure.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Mirror {
    /// The offset each `offsetof(<struct>, <field>)` assert states.
    pub fields: std::collections::BTreeMap<String, usize>,
    /// The size its `sizeof(<struct>)` assert states.
    pub size: Option<usize>,
}

/// Reads the asserts a header states about one struct.
///
/// Both the plain `static_assert(...)` form and the `UNISERVE_STATIC_ASSERT(...)` macro several
/// packages use are read; a multi-line assert's message continuation is not.
pub fn mirror_of(text: &str, structure: &str) -> Mirror {
    let mut mirror = Mirror::default();
    for line in text.lines() {
        let line = line.trim();
        let rest = line
            .strip_prefix("static_assert(")
            .or_else(|| line.strip_prefix("UNISERVE_STATIC_ASSERT("));
        let Some(rest) = rest else {
            continue;
        };
        let Some((subject, value)) = rest.split_once("==") else {
            continue;
        };
        let Some(number) = value
            .trim_start()
            .split(|character: char| !character.is_ascii_digit())
            .next()
            .filter(|digits| !digits.is_empty())
        else {
            continue;
        };
        let Ok(number) = number.parse::<usize>() else {
            continue;
        };
        let subject = subject.trim_end();
        if let Some(field) = subject
            .strip_prefix(&format!("offsetof({structure}, "))
            .and_then(|field| field.strip_suffix(')'))
        {
            mirror.fields.insert(field.trim().to_owned(), number);
        } else if subject == format!("sizeof({structure})") {
            mirror.size = Some(number);
        }
    }
    mirror
}

/// Every header of a package that states something about `structure`.
pub fn headers_of(package: &str, structure: &str) -> Vec<(PathBuf, Mirror)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("kernels")
        .join(package);
    let mut found = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
        for entry in entries {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if !matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("h") | Some("cuh")
            ) {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            if !text.contains(&format!("struct {structure}")) {
                continue;
            }
            let mirror = mirror_of(&text, structure);
            if mirror.size.is_some() || !mirror.fields.is_empty() {
                found.push((path, mirror));
            }
        }
    }
    found
}

/// Asserts that every package header of `package` states `fields` and `size` for `structure`.
///
/// Panics naming the header, the field, and both numbers, because the failure this catches is a
/// disagreement between two files and the reader needs to know which one to look at.
pub fn check(package: &str, structure: &str, fields: &[(&str, usize)], size: usize) {
    let headers = headers_of(package, structure);
    assert!(
        !headers.is_empty(),
        "no header under kernels/{package} states anything about `{structure}`: the package's mirror \
         of the operator's call structure is missing, so nothing ties the module to this crate"
    );
    for (path, mirror) in headers {
        let named = path.display();
        for (field, offset) in fields {
            assert_eq!(
                mirror.fields.get(*field).copied(),
                Some(*offset),
                "{named}: `{structure}.{field}` is passed at offset {offset}, and the header states \
                 {:?}",
                mirror.fields.get(*field)
            );
        }
        let stated: Vec<&str> = mirror
            .fields
            .keys()
            .map(String::as_str)
            .filter(|field| !fields.iter().any(|(name, _)| name == field))
            .collect();
        assert!(
            stated.is_empty(),
            "{named}: the header asserts {stated:?} of `{structure}`, which the operator's layout \
             does not have"
        );
        assert_eq!(
            mirror.size,
            Some(size),
            "{named}: `{structure}` must be {size} bytes, the operator's layout's size"
        );
    }
}
