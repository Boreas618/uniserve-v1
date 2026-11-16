//! GSM8K's strict extracted-answer accuracy.
//!
//! A response is correct only when its final answer's value equals the value
//! after the last marker of the row's reference answer. A model states its
//! final answer after a `#### ` marker or inside LaTeX's `\boxed{...}`; the form
//! that starts last decides, being the last final answer the model wrote. The
//! marker includes its trailing space, so `####20` carries no answer, and a
//! deciding form whose text does not reduce to one number is an incorrect
//! answer, never a request failure: every measured row stays in the denominator.

use regex::Regex;
use serde_json::Value;

use crate::record::Record;

/// The marker that introduces a final answer, its space included.
pub const ANSWER_MARKER: &str = "#### ";

/// The form that decided one response's final answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Marker,
    Boxed,
    None,
}

/// A final answer's value, normalized so that equal numbers compare equal
/// (`18`, `18.0` and `+18` alike): its sign, its integer digits without leading
/// zeros and its fraction digits without trailing zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number {
    negative: bool,
    integer: String,
    fraction: String,
}

impl Number {
    fn parse(text: &str) -> Self {
        let (negative, unsigned) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let unsigned = unsigned.replace(',', "");
        let (integer, fraction) = unsigned.split_once('.').unwrap_or((&unsigned, ""));
        let integer = integer.trim_start_matches('0').to_owned();
        let fraction = fraction.trim_end_matches('0').to_owned();
        // Zero has no sign.
        let negative = negative && !(integer.is_empty() && fraction.is_empty());
        Self {
            negative,
            integer,
            fraction,
        }
    }
}

/// One response's final answer and the form that decided it: `value` is `None`
/// when the deciding form's text is not one number, and when there is no form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub value: Option<Number>,
    pub form: Form,
}

#[expect(
    clippy::expect_used,
    reason = "the pattern is a constant that compiles"
)]
fn boxed() -> Regex {
    Regex::new(r"\\boxed\{(?P<payload>[^{}]*)\}").expect("a valid pattern")
}

/// A value may sit between whitespace and punctuation (a `$`, parentheses, a
/// full stop after it); a letter or digit next to it means the text is not the
/// answer alone, so `#### The answer is 18` and `#### 18 apples` hold no value.
/// The value keeps its sign, thousands commas and decimal fraction.
#[expect(
    clippy::expect_used,
    reason = "the pattern is a constant that compiles"
)]
fn answer_value() -> Regex {
    Regex::new(r"^[^\w+\-]*(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)[^\w]*$")
        .expect("a valid pattern")
}

fn parse_value(text: &str) -> Option<Number> {
    answer_value()
        .captures(text)
        .map(|captures| Number::parse(&captures["number"]))
}

/// The final answer of `text` and the form that decided it. Both forms are read
/// and the one that starts later decides; only its text is parsed, so a
/// malformed last form gives no answer rather than an earlier candidate's.
pub fn extract_form(text: &str) -> Answer {
    let marker = text.rfind(ANSWER_MARKER);
    let last_boxed = boxed().captures_iter(text).last();
    if let Some(captures) = &last_boxed
        && let Some(whole) = captures.get(0)
        && marker.is_none_or(|marker| whole.start() > marker)
    {
        return Answer {
            value: parse_value(&captures["payload"]),
            form: Form::Boxed,
        };
    }
    match marker {
        Some(marker) => Answer {
            value: parse_value(&text[marker + ANSWER_MARKER.len()..]),
            form: Form::Marker,
        },
        None => Answer {
            value: None,
            form: Form::None,
        },
    }
}

/// The number the text's last final-answer form introduces.
pub fn extract(text: &str) -> Option<Number> {
    extract_form(text).value
}

/// The gold value of the row a request was built from: its
/// `metadata.answer`, when that parses.
pub fn reference_answer(record: &Record) -> Option<Number> {
    record
        .example
        .as_ref()
        .and_then(|example| example.get("metadata"))
        .and_then(|metadata| metadata.get("answer"))
        .and_then(Value::as_str)
        .and_then(extract)
}

/// The outcome counts of one run over its measured rows. Each count of a form
/// is over successful requests; a failed request is counted in `total` alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Score {
    pub correct: usize,
    pub total: usize,
    pub unparseable: usize,
    pub by_marker: usize,
    pub by_boxed: usize,
    pub without_form: usize,
}

impl Score {
    /// Correct over total; zero when no row was measured.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

/// Scores every measured record against its gold answer. A failed request is
/// incorrect even when its partial text holds the right value.
pub fn score(records: &[&Record]) -> Score {
    let mut score = Score {
        total: records.len(),
        ..Score::default()
    };
    for record in records.iter().filter(|record| record.success) {
        let answer = extract_form(&record.generated_text);
        match answer.form {
            Form::Marker => score.by_marker += 1,
            Form::Boxed => score.by_boxed += 1,
            Form::None => score.without_form += 1,
        }
        let Some(value) = answer.value else {
            score.unparseable += 1;
            continue;
        };
        if reference_answer(record).is_some_and(|gold| gold == value) {
            score.correct += 1;
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The later form decides, and its text alone is parsed.
    #[test]
    fn the_last_final_answer_form_decides() {
        assert_eq!(extract("so #### 18"), Some(Number::parse("18")));
        assert_eq!(extract("#### 3\n\\boxed{18}"), Some(Number::parse("18")));
        assert_eq!(extract("\\boxed{3}\n#### 18."), Some(Number::parse("18")));
        assert_eq!(extract("#### 1,234.50"), Some(Number::parse("1234.5")));
        assert_eq!(extract("#### $18"), Some(Number::parse("18.0")));
        assert_eq!(extract_form("####20").form, Form::None);
        assert_eq!(extract("#### 18 apples"), None);
        assert_eq!(extract("#### The answer is 18"), None);
        assert_eq!(extract("#### 3\n\\boxed{\\frac{1}{2}}"), None);
    }
}
