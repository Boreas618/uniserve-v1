//! The C headers this crate's hand-written numbers come from, read by the
//! unit tests so that every constant is compared with the header that defines
//! it.
//!
//! A wrong number in a binding is not a compile error: it is a wrong
//! computation, a silently ignored attribute or a refused call, and the cost
//! is a hardware round trip to find it (a real one: cuBLASLt's workspace
//! preference was 0, which is its *search mode* attribute, and every
//! heuristic call was refused). The tests that use this module therefore read
//! the header the number came from and compare.
//!
//! The headers are provisioned on the machines that build device code; where
//! they are absent (a sandbox with no CUDA toolkit) the comparison cannot run,
//! and [`assert_constants`] says so on stderr instead of passing silently.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The directories a header is looked for in, in order, after the ones named
/// by `UNISERVE_CUDA_HEADERS` (a colon-separated list).
const DIRECTORIES: [&str; 6] = [
    "/usr/local/cuda/include",
    "/usr/include/aarch64-linux-gnu",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
    "/opt/nvidia/cuda/include",
    "/usr/local/cuda-13.0/include",
];

/// The value of the enumerator `name` in `header`, when the header is found
/// and defines it: from the header's enumerations (either explicitly, with
/// `= value`, or implicitly, by position, as CUDA's headers mostly are) or
/// from a `#define`.
pub(crate) fn value(header: &str, name: &str) -> Option<i64> {
    if let Some(value) = enumerators(header).get(name) {
        return Some(*value);
    }
    define(header, name)
}

/// Every enumerator of every `enum` in `header`, by name.
fn enumerators(header: &str) -> &'static HashMap<String, i64> {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static HashMap<String, i64>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(cached) = cache.get(header) {
        return cached;
    }
    let table = match text(header) {
        Some(text) => parse_enumerators(text),
        None => HashMap::new(),
    };
    let table: &'static HashMap<String, i64> = Box::leak(Box::new(table));
    cache.insert(header.to_owned(), table);
    table
}

/// Walks every `enum … { … }` body of `text`, numbering its entries the way C
/// does: an entry with `= value` takes that value, and one without takes the
/// previous entry's plus one.
fn parse_enumerators(text: &str) -> HashMap<String, i64> {
    let mut table = HashMap::new();
    let mut from = 0;
    while let Some(at) = text[from..].find('{') {
        let open = from + at;
        if !opens_an_enum(&text[from..open]) {
            from = open + 1;
            continue;
        }
        let mut depth = 0usize;
        let mut close = None;
        for (index, character) in text[open..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + index);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        // The comments come out before the split: CUDA's headers put prose,
        // commas and all, between the entries.
        let body = strip_comments(&text[open + 1..close]);
        let mut previous: Option<i64> = None;
        for entry in body.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let name: String = entry
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            let explicit = entry[name.len()..]
                .split_once('=')
                .map(|(_, value)| value.trim().to_owned());
            let value = match explicit {
                Some(expression) => match number(&expression) {
                    Some(value) => Some(value),
                    // An entry defined as another enumerator's name.
                    None => table.get(&expression).copied(),
                },
                // An entry without `=` continues the sequence, and one after
                // an entry this scan could not number has no number either:
                // counting from zero again would invent values.
                None => match previous {
                    Some(previous) => Some(previous + 1),
                    None if table.is_empty() => Some(0),
                    None => None,
                },
            };
            if let Some(value) = value {
                table.insert(name, value);
                previous = Some(value);
            } else {
                previous = None;
            }
        }
        from = close + 1;
    }
    table
}

/// Whether the text just before a `{` names an enumeration.
fn opens_an_enum(prefix: &str) -> bool {
    let recent = match prefix.len().checked_sub(200) {
        Some(start) => &prefix[start..],
        None => prefix,
    };
    let Some(at) = recent.rfind("enum") else {
        return false;
    };
    let between = &recent[at + 4..];
    // A statement boundary or a closing brace since `enum` means this brace
    // belongs to something else.
    !between.contains(';') && !between.contains('}') && !between.contains('(')
}

/// A C integer literal, in the forms the headers use: decimal, hexadecimal,
/// with an optional `U`/`L` suffix and parentheses around it.
fn number(expression: &str) -> Option<i64> {
    let trimmed = expression
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    let cleaned: String = trimmed
        .trim_end_matches(|character: char| character.is_ascii_alphabetic())
        .to_owned();
    if let Some(hex) = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
    {
        return i64::from_str_radix(hex, 16).ok();
    }
    cleaned.parse().ok()
}

/// `text` with every `/* … */` and `// …` comment removed.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start..].find("*/") {
            Some(end) => rest = &rest[start + end + 2..],
            None => return out,
        }
    }
    let mut rest = rest;
    while let Some(start) = rest.find("//") {
        out.push_str(&rest[..start]);
        match rest[start..].find('\n') {
            Some(end) => rest = &rest[start + end + 1..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Whether `header` could be read at all.
pub(crate) fn found(header: &str) -> bool {
    text(header).is_some()
}

/// Compares every `(enumerator, our value)` pair with the header.
///
/// Returns without asserting when the header is not provisioned on this
/// machine, saying so on stderr: a comparison that did not run must not look
/// like one that passed. When the header *is* there, an enumerator the header
/// does not define fails the test, because a table naming an enumerator that
/// does not exist cannot be checking anything.
pub(crate) fn assert_constants(header: &str, constants: &[(&str, i64)]) {
    if !found(header) {
        eprintln!(
            "skipped: {header} is not provisioned on this machine, so its {} constants were not compared",
            constants.len()
        );
        return;
    }
    let mut wrong = Vec::new();
    let mut missing = Vec::new();
    for (name, ours) in constants {
        match value(header, name) {
            Some(header_value) if header_value == *ours => {}
            Some(header_value) => wrong.push(format!("{name}: header {header_value}, ours {ours}")),
            None => missing.push(*name),
        }
    }
    assert!(
        missing.is_empty(),
        "{header} does not define {missing:?}; the table names an enumerator that is not there"
    );
    assert!(
        wrong.is_empty(),
        "{header} disagrees with the crate: {wrong:#?}"
    );
}

/// The nearest enclosing struct's size in a header, in bytes, computed from
/// the C field types the header lists.
///
/// Only the types this crate's own structures mirror are understood
/// (`unsigned int`, `size_t`, pointers, fixed arrays of those, nested structs
/// by name); the first field it cannot size makes it answer `None`, which the
/// caller reports as a skipped comparison rather than a failure.
pub(crate) fn struct_size(header: &str, name: &str) -> Option<usize> {
    let text = text(header)?;
    let body = struct_body(&text, name)?;
    let mut offset = 0usize;
    let mut alignment = 1usize;
    for field in fields(&body) {
        let (size, align) = field_size(header, &field)?;
        offset = offset.next_multiple_of(align) + size;
        alignment = alignment.max(align);
    }
    Some(offset.next_multiple_of(alignment))
}

/// The text between the braces of `struct name { … }`, or of the anonymous
/// struct a `typedef` closes with `} name;`.
fn struct_body(text: &str, name: &str) -> Option<String> {
    let start = match text.find(&format!("struct {name} {{")) {
        Some(start) => start,
        None => {
            // `typedef struct { … } Name;`: the body ends at the closing brace
            // that the name follows, and begins at the nearest `struct` before
            // it.
            let close = text.find(&format!("}} {name};"))?;
            let before = &text[..close];
            before.rfind("struct")?
        }
    };
    let open = start + text[start..].find('{')?;
    let mut depth = 0usize;
    for (index, character) in text[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[open + 1..open + index].to_owned());
                }
            }
            _ => {}
        }
    }
    None
}

/// The declarations of one struct body, each with its comments, its
/// preprocessor lines and its alignment specifiers removed: what is left is a
/// `;`-separated list of C declarations.
fn fields(body: &str) -> Vec<String> {
    let body = body
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .map(|line| {
            let mut line = line.to_owned();
            for marker in ["alignas(", "_Alignas("] {
                while let Some(at) = line.find(marker) {
                    match line[at..].find(')') {
                        Some(end) => line.replace_range(at..at + end + 1, ""),
                        None => break,
                    }
                }
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");
    body.lines()
        .map(|line| match line.find("/*") {
            Some(at) => line[..at].to_owned(),
            None => line.to_owned(),
        })
        .filter(|line| !line.contains("//"))
        .collect::<Vec<_>>()
        .join(" ")
        .split(';')
        .map(|declaration| declaration.trim().to_owned())
        .filter(|declaration| !declaration.is_empty())
        .collect()
}

/// The size and alignment of one C declaration, from its type and any array
/// extent or bit field on the name.
fn field_size(header: &str, declaration: &str) -> Option<(usize, usize)> {
    let mut words = declaration.split_whitespace();
    let mut kind = String::new();
    while let Some(word) = words.next() {
        // A type ends at the identifier that starts the declarator: the first
        // word the primitive types below do not claim.
        let primitive = matches!(
            word,
            "unsigned"
                | "signed"
                | "int"
                | "char"
                | "short"
                | "long"
                | "float"
                | "double"
                | "void"
                | "size_t"
                | "const"
                | "*"
                | "struct"
                | "CU"
        ) || word.starts_with("CU")
            || word.starts_with("unsigned")
            || word.starts_with("cuda")
            || word.ends_with("_t");
        if !primitive {
            break;
        }
        kind.push_str(word);
        kind.push(' ');
    }
    let kind = kind.trim();
    let (size, align) = match kind {
        "unsigned int" | "int" | "unsigned" | "signed int" | "float" => (4, 4),
        "char" | "signed char" | "unsigned char" | "bool" => (1, 1),
        "short" | "unsigned short" => (2, 2),
        "unsigned long long" | "long long" => (8, 8),
        "unsigned long" | "long" => (8, 8),
        "size_t" | "unsigned __int64" | "CUdeviceptr" => (8, 8),
        "cuuint64_t" | "int64_t" | "uint64_t" => (8, 8),
        "cuuint32_t" | "int32_t" | "uint32_t" => (4, 4),
        "cuuint16_t" | "int16_t" | "uint16_t" => (2, 2),
        "cuuint8_t" | "int8_t" | "uint8_t" => (1, 1),
        "double" => (8, 8),
        _ if kind.ends_with('*') => (8, 8),
        _ => {
            // A nested struct this crate mirrors, by name.
            let name = kind.strip_prefix("struct ").unwrap_or(kind);
            let nested = struct_size(header, name)?;
            (nested, 8)
        }
    };
    // A fixed array multiplies the size: `unsigned int reserved[3]`, whose
    // extent may be a `#define`d constant.
    let name = declaration.rsplit(' ').next().unwrap_or(declaration);
    let count = match name
        .split_once('[')
        .and_then(|(_, rest)| rest.split(']').next())
    {
        Some(count) => match count.trim().parse::<usize>() {
            Ok(count) => count,
            Err(_) => define(header, count.trim())? as usize,
        },
        None => 1,
    };
    Some((size * count, align))
}

/// The value of `#define name value` in `header`.
fn define(header: &str, name: &str) -> Option<i64> {
    let text = text(header)?;
    for line in text.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("#define") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(after) = rest.strip_prefix(name) else {
            continue;
        };
        if !after.starts_with(char::is_whitespace) {
            continue;
        }
        let value: String = after
            .trim_start()
            .chars()
            .take_while(|character| character.is_ascii_digit() || *character == '-')
            .collect();
        if let Ok(value) = value.parse() {
            return Some(value);
        }
    }
    None
}

/// The contents of `header`, read once per process.
fn text(header: &str) -> Option<&'static str> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<&'static str>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(cached) = cache.get(header) {
        return *cached;
    }
    let mut directories: Vec<PathBuf> = std::env::var("UNISERVE_CUDA_HEADERS")
        .map(|paths| paths.split(':').map(PathBuf::from).collect())
        .unwrap_or_default();
    directories.extend(DIRECTORIES.iter().map(PathBuf::from));
    let found = directories
        .iter()
        .map(|directory| directory.join(header))
        .find(|path| path.is_file())
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| &*Box::leak(text.into_boxed_str()));
    cache.insert(header.to_owned(), found);
    found
}

/// Where `name` starts in `line`, as a whole identifier: not part of a longer
/// name (`CUBLASLT_X` is not `CUBLASLT_X_Y`), and not inside a comment.
fn find_identifier(line: &str, name: &str) -> Option<usize> {
    let comment = line.find("/*").unwrap_or(line.len());
    let line = &line[..comment];
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before_ok = start == 0 || !is_identifier_byte(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_identifier_byte(bytes[end]);
        if before_ok && after_ok {
            return Some(start);
        }
        from = end;
    }
    None
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
mod tests {
    use super::{find_identifier, struct_size, value};

    /// The scan reads enumerators, not comments or longer names.
    #[test]
    fn an_enumerator_is_found_by_its_whole_name() {
        let line = "    CUBLASLT_MATMUL_DESC_TRANSA = 3, /* 3 */";
        assert_eq!(
            find_identifier(line, "CUBLASLT_MATMUL_DESC_TRANSA"),
            Some(4)
        );
        assert_eq!(find_identifier(line, "CUBLASLT_MATMUL_DESC_TRANS"), None);
        assert_eq!(
            find_identifier(line, "CUBLASLT_MATMUL_DESC_TRANSA_EXTRA"),
            None
        );
        // A definition inside a comment is not a definition.
        assert_eq!(find_identifier("/* CUBLASLT_X = 4, */", "CUBLASLT_X"), None);
    }

    /// A real header this sandbox has: `CUtensorMap` is 128 bytes and 64-byte
    /// aligned in `cuda.h`, which is what the crate's `#[repr(C, align(64))]`
    /// claims.
    #[test]
    fn a_cuda_struct_is_sized_from_its_header() {
        if !super::found("cuda.h") {
            eprintln!("skipped: cuda.h is not provisioned on this machine");
            return;
        }
        assert_eq!(struct_size("cuda.h", "CUtensorMap"), Some(128));
        assert_eq!(
            value("cuda.h", "CU_TENSOR_MAP_DATA_TYPE_FLOAT32"),
            Some(7),
            "the tensor map's FP32 code"
        );
        assert_eq!(value("cuda.h", "CUDA_SUCCESS"), Some(0));
    }
}
