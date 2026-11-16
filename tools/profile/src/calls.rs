//! Repeated call edges between Python and selected native functions, from a
//! cProfile capture.
//!
//! A capture is `pstats`' file: a `marshal` dump of a dictionary from each
//! function, `(filename, line, name)` with `~` as the filename of a builtin, to
//! `(primitive calls, calls, own time, inclusive time, callers)`, its callers
//! being a dictionary from each caller to `(calls, primitive calls, own time,
//! inclusive time)` of that edge. A native function is a builtin whose label
//! holds one of the given names (substrings of labels, not of source paths).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

/// A value of a `marshal` dump, as far as a capture holds them.
#[derive(Debug, Clone, PartialEq)]
enum Marshal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Tuple(Vec<Marshal>),
    Dict(Vec<(Marshal, Marshal)>),
}

/// A `marshal` reader: a type byte (its high bit asking for the value to be kept
/// for later references), then the value.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    refs: Vec<Marshal>,
}

impl Reader<'_> {
    fn bytes(&mut self, count: usize) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.data.len())
            .context("a truncated capture")?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn value(&mut self) -> Result<Marshal> {
        let code = self.bytes(1)?[0];
        let (kind, keep) = (code & 0x7f, code & 0x80 != 0);
        // A kept container takes its reference before its items are read.
        let slot = keep.then(|| {
            self.refs.push(Marshal::None);
            self.refs.len() - 1
        });
        let value = match kind {
            b'N' => Marshal::None,
            b'T' => Marshal::Bool(true),
            b'F' => Marshal::Bool(false),
            b'i' => Marshal::Int(i64::from(self.u32()? as i32)),
            b'l' => {
                // Base-2^15 digits, least significant first; the count's sign is
                // the number's.
                let count = self.u32()? as i32;
                let mut value: i128 = 0;
                for index in 0..count.unsigned_abs() {
                    let digits = self.bytes(2)?;
                    let digit = i128::from(u16::from_le_bytes([digits[0], digits[1]]));
                    value += digit << (15 * index);
                }
                Marshal::Int(i64::try_from(if count < 0 { -value } else { value })?)
            }
            b'g' => {
                let bytes = self.bytes(8)?;
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                Marshal::Float(f64::from_le_bytes(array))
            }
            b'u' | b't' | b'a' | b'A' => {
                let length = self.u32()? as usize;
                Marshal::Text(String::from_utf8_lossy(self.bytes(length)?).into_owned())
            }
            b'z' | b'Z' => {
                let length = usize::from(self.bytes(1)?[0]);
                Marshal::Text(String::from_utf8_lossy(self.bytes(length)?).into_owned())
            }
            b'(' | b')' => {
                let count = if kind == b')' {
                    usize::from(self.bytes(1)?[0])
                } else {
                    self.u32()? as usize
                };
                Marshal::Tuple((0..count).map(|_| self.value()).collect::<Result<_>>()?)
            }
            b'{' => {
                let mut entries = Vec::new();
                loop {
                    if self.data.get(self.at) == Some(&b'0') {
                        self.at += 1;
                        break;
                    }
                    let key = self.value()?;
                    let value = self.value()?;
                    entries.push((key, value));
                }
                Marshal::Dict(entries)
            }
            b'r' => {
                let index = self.u32()? as usize;
                self.refs
                    .get(index)
                    .cloned()
                    .context("a reference to an unread value")?
            }
            other => bail!("a capture value of marshal type {:?}", other as char),
        };
        if let Some(slot) = slot {
            self.refs[slot] = value.clone();
        }
        Ok(value)
    }
}

/// A function's `(filename, line, name)`.
type Function = (String, i64, String);

fn function(value: &Marshal) -> Result<Function> {
    match value {
        Marshal::Tuple(items) if items.len() == 3 => match (&items[0], &items[1], &items[2]) {
            (Marshal::Text(file), Marshal::Int(line), Marshal::Text(name)) => {
                Ok((file.clone(), *line, name.clone()))
            }
            _ => bail!("a capture function is not (filename, line, name)"),
        },
        _ => bail!("a capture function is not a 3-tuple"),
    }
}

fn number(value: &Marshal) -> f64 {
    match value {
        Marshal::Int(number) => *number as f64,
        Marshal::Float(number) => *number,
        _ => 0.0,
    }
}

/// A function's label: a builtin's name, else `file:line(name)`.
fn label((file, line, name): &Function) -> String {
    if file == "~" {
        name.clone()
    } else {
        format!("{file}:{line}({name})")
    }
}

#[derive(Debug, Serialize)]
pub struct CallHint {
    pub caller: String,
    pub callee: String,
    pub direction: String,
    pub calls: i64,
    pub total_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct CallReport {
    pub hints: Vec<CallHint>,
    pub native_functions: Vec<String>,
    pub notes: Vec<String>,
}

/// The edges called twice or more between a selected native function and a
/// Python one, by inclusive time, then by calls.
///
/// # Errors
///
/// When the capture is not a `pstats` file.
pub fn analyze(path: &Path, native_names: &[String]) -> Result<CallReport> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut reader = Reader {
        data: &data,
        at: 0,
        refs: Vec::new(),
    };
    let Marshal::Dict(stats) = reader.value()? else {
        bail!("{} is not a cProfile capture", path.display());
    };
    let mut entries = Vec::new();
    for (key, value) in &stats {
        let callee = function(key)?;
        let Marshal::Tuple(fields) = value else {
            bail!("a capture entry is not a tuple")
        };
        ensure!(
            fields.len() == 5,
            "a capture entry is not (cc, nc, tt, ct, callers)"
        );
        let Marshal::Dict(callers) = &fields[4] else {
            bail!("a capture entry's callers are not a dictionary")
        };
        let callers: Vec<(Function, i64, f64)> = callers
            .iter()
            .map(|(caller, edge)| {
                let Marshal::Tuple(edge) = edge else {
                    bail!("a caller edge is not a tuple")
                };
                ensure!(edge.len() == 4, "a caller edge is not (nc, cc, tt, ct)");
                Ok((function(caller)?, number(&edge[0]) as i64, number(&edge[3])))
            })
            .collect::<Result<_>>()?;
        entries.push((callee, callers));
    }
    let native: BTreeSet<Function> = entries
        .iter()
        .map(|(function, _)| function)
        .filter(|(file, _, name)| {
            file == "~"
                && native_names
                    .iter()
                    .any(|native| name.contains(native.as_str()))
        })
        .cloned()
        .collect();
    let mut hints = Vec::new();
    for (callee, callers) in &entries {
        for (caller, calls, total) in callers {
            let (from_native, to_native) = (native.contains(caller), native.contains(callee));
            if *calls < 2 || from_native == to_native {
                continue;
            }
            let peer = if from_native { callee } else { caller };
            if peer.0 == "~" {
                continue;
            }
            hints.push(CallHint {
                caller: label(caller),
                callee: label(callee),
                direction: if from_native {
                    "native-to-python"
                } else {
                    "python-to-native"
                }
                .into(),
                calls: *calls,
                total_seconds: *total,
            });
        }
    }
    hints.sort_by(|a, b| {
        b.total_seconds
            .total_cmp(&a.total_seconds)
            .then(b.calls.cmp(&a.calls))
    });
    let mut notes = vec![
        "Repeated edges are inspection candidates: batch numerical calls are expected, while per-row metadata \
         conversions may be avoidable."
            .to_owned(),
        "cProfile has no timeline or batch attribution. Its thread coverage depends on capture; foreign threads and \
         many C API accesses are absent."
            .to_owned(),
        "Inclusive edge times overlap. Do not sum them or treat them as FFI overhead; they include work performed by \
         the callee."
            .to_owned(),
    ];
    if native.is_empty() {
        notes.push("No selected native functions recorded; coverage is unknown.".into());
    }
    let mut native_functions: Vec<String> = native.iter().map(label).collect();
    native_functions.sort();
    Ok(CallReport {
        hints,
        native_functions,
        notes,
    })
}
