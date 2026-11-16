//! Compares two GSM8K bundles under the quality confirmation rule.
//!
//! The candidate is accepted when the two accuracies' difference lies within
//! the confidence interval of the difference of two binomial proportions at the
//! runs' sample size, and it failed no check. The premises come first: both
//! bundles are completed GSM8K runs over the same selected rows (the protocol's
//! 1319), with the same sampling settings, served model, benchmark cell (the
//! point's provenance) and accuracy direction, each having scored every row
//! against a gold answer, the reference a valid run; a pair that fails one is
//! not comparable. A request the candidate failed is an incorrect answer, as
//! every GSM8K run scores it. The report states the change in the declared
//! direction: `regression` or `improvement` beyond the interval, `none` within.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};

/// The rows of GSM8K's test split, the protocol's sample set.
const GSM8K_ROWS: u64 = 1319;

/// A bundle's summary and lifecycle record.
struct Bundle {
    directory: String,
    summary: Map<String, Value>,
    run: Map<String, Value>,
}

fn read(directory: &Path) -> Result<Bundle> {
    let object = |name: &str| -> Result<Map<String, Value>> {
        let path = directory.join(name);
        let value: Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )?;
        match value {
            Value::Object(object) => Ok(object),
            _ => bail!("{} does not hold a JSON object", path.display()),
        }
    };
    Ok(Bundle {
        directory: directory.display().to_string(),
        summary: object("summary.json")?,
        run: object("run.json")?,
    })
}

impl Bundle {
    fn section(&self, name: &str) -> Result<&Map<String, Value>> {
        self.summary
            .get(name)
            .and_then(Value::as_object)
            .with_context(|| format!("the bundle has no {name} object"))
    }

    /// The scored counts: `correct` of `total`.
    fn proportion(&self) -> Result<(u64, u64)> {
        let metrics = self.section("metrics")?;
        let count = |key: &str| {
            metrics
                .get(key)
                .filter(|value| !value.is_boolean())
                .and_then(Value::as_u64)
                .with_context(|| format!("{}: summary.metrics.{key} missing", self.directory))
        };
        let (correct, total) = (count("correct")?, count("total")?);
        ensure!(total >= 1, "a proportion needs at least one sample");
        ensure!(correct <= total, "correct must lie within [0, total]");
        Ok((correct, total))
    }

    fn failed_checks(&self) -> Result<Vec<String>> {
        let checks = self
            .section("validation")?
            .get("checks")
            .and_then(Value::as_object)
            .with_context(|| format!("{}: summary.validation.checks missing", self.directory))?;
        let mut failed: Vec<String> = checks
            .iter()
            .filter(|(_, ok)| ok.as_bool() != Some(true))
            .map(|(name, _)| name.clone())
            .collect();
        failed.sort();
        Ok(failed)
    }

    fn direction(&self) -> Result<String> {
        self.summary
            .get("metric_definitions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|definition| {
                definition["path"] == "accuracy"
                    && ["higher", "lower"]
                        .contains(&definition["direction"].as_str().unwrap_or_default())
            })
            .and_then(|definition| definition["direction"].as_str().map(str::to_owned))
            .with_context(|| {
                format!(
                    "{}: the point declares no accuracy metric direction",
                    self.directory
                )
            })
    }

    fn launch(&self) -> Map<String, Value> {
        self.summary
            .get("launch")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    }

    /// The checkpoint the daemon's command served: the first value after
    /// `serve`, or the directory `--model-dir` names there.
    fn model_argument(&self) -> Option<String> {
        let command: Vec<String> = self
            .launch()
            .get("command")?
            .as_array()?
            .iter()
            .map(|item| {
                item.as_str()
                    .map_or_else(|| item.to_string(), str::to_owned)
            })
            .collect();
        let index = command.iter().position(|word| word == "serve")?;
        let rest = &command[index + 1..];
        if rest.first().map(String::as_str) == Some("--model-dir") {
            return rest.get(1).cloned();
        }
        rest.first().filter(|word| !word.starts_with('-')).cloned()
    }

    fn describe(&self) -> Result<Value> {
        let workload = self.section("workload")?;
        let (correct, total) = self.proportion()?;
        let launch = self.launch();
        Ok(json!({
            "directory": self.directory,
            "model": workload.get("model"),
            "provenance": workload.get("provenance"),
            "accuracy": correct as f64 / total as f64,
            "correct": correct,
            "total": total,
            "failed_requests": self.summary.get("failed_count"),
            "failed_checks": self.failed_checks()?,
            "selected_rows": self.summary.get("selected_rows"),
            "command": launch.get("command"),
            "git_head": launch.get("git_head"),
            "dirty": launch.get("dirty"),
        }))
    }
}

/// Refuses runs that do not share the comparison's premises, naming the first
/// that fails.
fn comparable(reference: &Bundle, candidate: &Bundle) -> Result<()> {
    for (role, bundle) in [("reference", reference), ("candidate", candidate)] {
        ensure!(
            bundle.run.get("status").and_then(Value::as_str) == Some("completed"),
            "{role} run {} is not completed: {:?}",
            bundle.directory,
            bundle.run.get("status")
        );
        ensure!(
            bundle.summary.get("task").and_then(Value::as_str) == Some("gsm8k"),
            "{role} run {} is not a gsm8k run",
            bundle.directory
        );
        ensure!(
            bundle.summary.get("dataset").and_then(Value::as_str) == Some("gsm8k"),
            "{role} run {} did not score gsm8k",
            bundle.directory
        );
        ensure!(
            !bundle.failed_checks()?.contains(&"gold_answers".to_owned()),
            "{role} run {} has rows without a gold answer",
            bundle.directory
        );
    }
    ensure!(
        reference
            .section("validation")?
            .get("valid")
            .and_then(Value::as_bool)
            == Some(true),
        "reference run {} is not a valid run",
        reference.directory
    );
    ensure!(
        reference.direction()? == candidate.direction()?,
        "the runs declare different accuracy directions"
    );
    let rows = reference.summary.get("selected_rows");
    ensure!(
        rows == candidate.summary.get("selected_rows"),
        "the runs scored different sample sets: {rows:?} and {:?}",
        candidate.summary.get("selected_rows")
    );
    let count = rows
        .and_then(Value::as_object)
        .context("the runs recorded no selected sample set")?
        .get("count")
        .and_then(Value::as_u64)
        .context("the selected sample set has no row count")?;
    ensure!(
        count == GSM8K_ROWS,
        "the runs scored {count} rows, not the protocol's {GSM8K_ROWS}"
    );
    let (reference_workload, candidate_workload) = (
        reference.section("workload")?,
        candidate.section("workload")?,
    );
    for key in ["sampling", "provenance", "model"] {
        ensure!(
            reference_workload.get(key) == candidate_workload.get(key),
            "the runs differ in workload.{key}: {:?} and {:?}",
            reference_workload.get(key),
            candidate_workload.get(key)
        );
    }
    for (role, bundle) in [("reference", reference), ("candidate", candidate)] {
        let rows = bundle
            .section("workload")?
            .get("load")
            .and_then(|load| load.get("num_prompts"))
            .and_then(Value::as_u64);
        ensure!(
            rows == Some(count),
            "{role} run scores {rows:?} rows but records {count} selected rows"
        );
        let (_, total) = bundle.proportion()?;
        ensure!(
            total == count,
            "{role} run scored {total} rows but records {count} selected rows"
        );
    }
    if let (Some(one), Some(other)) = (reference.model_argument(), candidate.model_argument()) {
        ensure!(
            one == other,
            "the runs served different checkpoints: {one} and {other}"
        );
    }
    Ok(())
}

/// The standard normal distribution's quantile: Wichura's algorithm AS241
/// (PPND16), as Python's `NormalDist().inv_cdf` computes it.
#[expect(
    clippy::excessive_precision,
    reason = "AS241's coefficients as published, which CPython's `NormalDist` takes too"
)]
pub fn normal_quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        let numerator = ((((((2_509.080_928_730_122_7 * r + 33_430.575_583_588_128) * r
            + 67_265.770_927_008_7)
            * r
            + 45_921.953_931_549_87)
            * r
            + 13_731.693_765_509_461)
            * r
            + 1_971.590_950_306_551_3)
            * r
            + 133.141_667_891_784_38)
            * r
            + 3.387_132_872_796_366_5;
        let denominator = ((((((5_226.495_278_852_545 * r + 28_729.085_735_721_943) * r
            + 39_307.895_800_092_71)
            * r
            + 21_213.794_301_586_597)
            * r
            + 5_394.196_021_424_751)
            * r
            + 687.187_007_492_057_9)
            * r
            + 42.313_330_701_600_91)
            * r
            + 1.0;
        return q * numerator / denominator;
    }
    let r = if q <= 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        let numerator = ((((((7.745_450_142_783_414e-4 * r + 0.022_723_844_989_269_184) * r
            + 0.241_780_725_177_450_6)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_5)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_546)
            * r
            + 1.423_437_110_749_683_5;
        let denominator = ((((((1.050_750_071_644_416_9e-9 * r + 5.475_938_084_995_345e-4) * r
            + 0.015_198_666_563_616_457)
            * r
            + 0.148_103_976_427_480_08)
            * r
            + 0.689_767_334_985_1)
            * r
            + 1.676_384_830_183_803_8)
            * r
            + 2.053_191_626_637_759)
            * r
            + 1.0;
        numerator / denominator
    } else {
        let r = r - 5.0;
        let numerator = ((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 0.001_242_660_947_388_078_4)
            * r
            + 0.026_532_189_526_576_124)
            * r
            + 0.296_560_571_828_504_9)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103;
        let denominator = ((((((2.044_263_103_389_939_7e-15 * r + 1.421_511_758_316_446e-7)
            * r
            + 1.846_318_317_510_054_8e-5)
            * r
            + 7.868_691_311_456_133e-4)
            * r
            + 0.014_875_361_290_850_615)
            * r
            + 0.136_929_880_922_735_8)
            * r
            + 0.599_832_206_555_888)
            * r
            + 1.0;
        numerator / denominator
    };
    if q < 0.0 { -x } else { x }
}

/// `compare`'s report: both runs' accuracies, failed checks and provenance; the
/// difference (candidate less reference), its standard error, the interval and
/// the change; the verdict, and the reasons for a rejection.
///
/// # Errors
///
/// When a bundle's records are missing or the runs are not comparable.
pub fn compare(reference: &Path, candidate: &Path, confidence: f64) -> Result<Value> {
    ensure!(
        confidence > 0.0 && confidence < 1.0,
        "confidence must lie in (0, 1)"
    );
    let (reference, candidate) = (read(reference)?, read(candidate)?);
    comparable(&reference, &candidate)?;
    let direction = reference.direction()?;
    let ((reference_correct, reference_total), (candidate_correct, candidate_total)) =
        (reference.proportion()?, candidate.proportion()?);
    let (p0, p1) = (
        reference_correct as f64 / reference_total as f64,
        candidate_correct as f64 / candidate_total as f64,
    );
    // The two runs are independent samples: both score one fixed sample set
    // with one checkpoint and one setting.
    let difference = p1 - p0;
    let standard_error = (p0 * (1.0 - p0) / reference_total as f64
        + p1 * (1.0 - p1) / candidate_total as f64)
        .sqrt();
    let half_width = normal_quantile(0.5 + confidence / 2.0) * standard_error;
    let within = difference.abs() <= half_width;
    let better = if direction == "higher" {
        difference > 0.0
    } else {
        difference < 0.0
    };
    let change = if within {
        "none"
    } else if better {
        "improvement"
    } else {
        "regression"
    };
    let mut reasons: Vec<&str> = if change == "none" {
        Vec::new()
    } else {
        vec![change]
    };
    if !candidate.failed_checks()?.is_empty() {
        reasons.push("failed_checks");
    }
    Ok(json!({
        "confidence": confidence,
        "method": "two-proportion difference, Wald normal interval at the sample size of the runs",
        "reference": reference.describe()?,
        "candidate": candidate.describe()?,
        "difference": {
            "value": difference,
            "standard_error": standard_error,
            "half_width": half_width,
            "interval": [difference - half_width, difference + half_width],
            "within_interval": within,
            "direction": direction,
            "change": change,
        },
        "verdict": if reasons.is_empty() { "accepted" } else { "rejected" },
        "reasons": reasons,
    }))
}

#[cfg(test)]
mod tests {
    /// Python's `NormalDist().inv_cdf` at 0.975 and 0.995.
    #[test]
    fn the_quantile_is_as241s() {
        assert!((super::normal_quantile(0.975) - 1.959_963_984_540_053_6).abs() < 1e-14);
        assert!((super::normal_quantile(0.995) - 2.575_829_303_548_9).abs() < 1e-14);
        assert!((super::normal_quantile(0.0001) + 3.719_016_485_455_68).abs() < 1e-12);
    }
}
