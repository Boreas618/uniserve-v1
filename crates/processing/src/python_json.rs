//! Python's `json.dumps`, as the reference encoders and output parsers a family ports from Python
//! write JSON into prompts and tool-call arguments.
//!
//! [`dumps`] writes what `json.dumps(value, ensure_ascii=False)` writes with its default
//! separators: `", "` between items, `": "` after a key, text other than ASCII as it is, and each
//! float as Python's `repr` spells it (`1e+16`, `1e-05`, `100.0`), which differs from the shortest
//! spelling `serde_json` writes (`1e16`, `1e-5`). Strings are escaped as both escape them: `"`,
//! `\` and the control characters below U+0020, `\b`, `\f`, `\n`, `\r` and `\t` by their short
//! escapes and the others as `\u00XX`.
//!
//! An integer is written as its value. `serde_json` holds an integer beyond 64 bits as a float, so
//! such an integer is written as Python writes that float rather than as Python writes the
//! integer.

use std::io;

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::Formatter;

/// The text `json.dumps(value, ensure_ascii=False)` writes for `value`.
#[expect(
    clippy::expect_used,
    reason = "a JSON value always serializes, writing into a vector cannot fail, and serde_json \
              writes UTF-8"
)]
pub(crate) fn dumps(value: &Value) -> String {
    let mut text = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut text, PythonFormatter);
    value
        .serialize(&mut serializer)
        .expect("a JSON value serializes into memory");
    String::from_utf8(text).expect("serde_json writes UTF-8")
}

/// The text of `text` as a JSON string, quotes included, as `json.dumps(text,
/// ensure_ascii=False)` writes it.
#[expect(clippy::expect_used, reason = "a string always serializes into memory")]
pub(crate) fn string(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes into memory")
}

/// The JSON text of a string's content, without its quotes: what [`string`] writes between them.
/// The escape of a text is the concatenation of the escapes of its parts, so a value streamed in
/// pieces is escaped piece by piece.
pub(crate) fn escaped(text: &str) -> String {
    let quoted = string(text);
    quoted[1..quoted.len() - 1].to_owned()
}

/// `serde_json`'s formatter with Python's default separators and float spelling.
struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        writer.write_all(float_repr(f64::from(value)).as_bytes())
    }
}

/// Python's `repr` of `value`, which `json.dumps` writes for a float.
///
/// The digits are the shortest that round-trip, as Rust's `{:e}` writes them. Python writes them
/// in positional notation, with at least one digit after the point, when the decimal point falls
/// within 16 digits of the first digit and no more than 4 zeros before it (`0.0001`), and in
/// scientific notation otherwise, its exponent signed and of at least two digits (`1e-05`,
/// `1.5e+16`).
#[expect(
    clippy::expect_used,
    reason = "`{:e}` writes a finite float as its digits, `e` and a decimal exponent"
)]
fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }

    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes a decimal exponent");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    // The decimal point stands after the first `point` digits (before them when negative).
    let point = exponent + 1;

    let body = if point <= -4 || point > 16 {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{first}{fraction}e{exponent_sign}{:02}", exponent.abs())
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
    } else {
        let point = point as usize;
        if point >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            format!("{}.{}", &digits[..point], &digits[point..])
        }
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Floats are spelled as Python's `repr` spells them, each expected text written by
    /// Python 3.12's `json.dumps`.
    #[test]
    fn floats_are_spelled_as_python_spells_them() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (100.0, "100.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (9007199254740993.0, "9007199254740992.0"),
            (123456789012345.6, "123456789012345.6"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (2.5e300, "2.5e+300"),
            (-3.25e-10, "-3.25e-10"),
            (1e22, "1e+22"),
            (0.30000000000000004, "0.30000000000000004"),
        ];
        for (value, expected) in cases {
            assert_eq!(dumps(&json!(value)), expected, "{value:e}");
        }
    }

    /// Items and keys are separated as Python separates them by default, and strings keep text
    /// other than ASCII, escaping quotes, backslashes and control characters alone.
    #[test]
    fn values_are_written_with_python_separators_and_escapes() {
        let value = json!({"a": [1, 2, {"b": null}], "c": "x\"\\\n\t\u{1}\u{7f}中", "d": true});
        assert_eq!(
            dumps(&value),
            "{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\\\"\\\\\\n\\t\\u0001\u{7f}中\", \"d\": true}"
        );
        assert_eq!(escaped("a\"b\n"), "a\\\"b\\n");
    }
}
