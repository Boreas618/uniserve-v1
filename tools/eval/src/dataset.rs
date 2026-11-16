//! The dataset formats a point's rows come from, each giving its rows in a
//! stated, deterministic order.
//!
//! A source is a local path the point names or, unset, the dataset's Hub
//! repository at `dataset_revision`. A point resolves exactly
//! `load.num_prompts` rows: a source that yields fewer fails the run. The
//! seeded selections shuffle as CPython's `random.Random(seed)` does
//! ([`crate::pyrandom`]), so a protocol's sample set is the same under every
//! implementation of the harness.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use serde_json::{Map, Value};

use crate::pyrandom::Random;
use crate::record::Example;
use crate::spec::Point;

/// The question an i2t row asks when the point states none.
pub const DEFAULT_I2T_QUESTION: &str = "Describe this image in detail.";

/// A dataset format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// JSON Lines objects in file order, each with `id`, `prompt` and the
    /// optional fields of [`Example`].
    Jsonl,
    /// ShareGPT's first two turns of each conversation, seeded, with their
    /// token counts under the point's tokenizer.
    ShareGpt,
    /// GSM8K's test split in file order, with each row's reference answer.
    Gsm8k,
    /// MJHQ-30K's prompts, seeded.
    Mjhq,
    /// PIE-Bench's image-editing pairs, seeded, as PNG.
    PieBench,
    /// Beans' photographs, seeded, as JPEG, each asked the point's question.
    Beans,
    /// UEval's prompts, seeded.
    Ueval,
    /// One prompt of exactly `video.prompt_tokens` tokens under the point's
    /// tokenizer, built from the base and filler texts of the file
    /// `dataset_path` names, repeated `num_prompts` times.
    FixedLength,
}

impl Kind {
    /// The format a point names.
    ///
    /// # Errors
    ///
    /// For an unknown name.
    pub fn of(name: &str) -> Result<Self> {
        Ok(match name {
            "jsonl" => Self::Jsonl,
            "sharegpt" => Self::ShareGpt,
            "gsm8k" => Self::Gsm8k,
            "mjhq" => Self::Mjhq,
            "pie-bench" => Self::PieBench,
            "beans" => Self::Beans,
            "ueval" => Self::Ueval,
            "fixed-length" => Self::FixedLength,
            _ => bail!(
                "unknown dataset {name:?}; known: beans, fixed-length, gsm8k, jsonl, mjhq, pie-bench, sharegpt, ueval"
            ),
        })
    }

    fn requires_tokenizer(self) -> bool {
        matches!(self, Self::ShareGpt | Self::FixedLength)
    }

    fn requires_path(self) -> bool {
        matches!(self, Self::Jsonl | Self::FixedLength)
    }

    /// The point's path and tokenizer as this format needs them; an empty
    /// string counts as unset.
    pub(crate) fn check_point(
        self,
        tokenizer: Option<&str>,
        dataset_path: Option<&str>,
        context: &str,
    ) -> Result<()> {
        ensure!(
            !self.requires_tokenizer() || tokenizer.is_some_and(|tokenizer| !tokenizer.is_empty()),
            "{context}.tokenizer is required for dataset {}",
            self.name()
        );
        ensure!(
            !self.requires_path() || dataset_path.is_some_and(|path| !path.is_empty()),
            "{context}.dataset_path is required for dataset {}",
            self.name()
        );
        Ok(())
    }

    fn name(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
            Self::ShareGpt => "sharegpt",
            Self::Gsm8k => "gsm8k",
            Self::Mjhq => "mjhq",
            Self::PieBench => "pie-bench",
            Self::Beans => "beans",
            Self::Ueval => "ueval",
            Self::FixedLength => "fixed-length",
        }
    }
}

/// The point's rows: exactly `load.num_prompts` of them.
///
/// # Errors
///
/// When the source cannot be read, a row does not fit its format, or the
/// source yields another count.
pub fn load(point: &Point) -> Result<Vec<Example>> {
    let kind = Kind::of(&point.dataset)?;
    let path = point
        .dataset_path
        .as_deref()
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    ensure!(
        !kind.requires_path() || path.is_some(),
        "dataset {:?} requires dataset_path",
        point.dataset
    );
    let tokenizer = if kind.requires_tokenizer() {
        Some(tokenizer(
            point
                .tokenizer
                .as_deref()
                .filter(|name| !name.is_empty())
                .unwrap_or(&point.model),
        )?)
    } else {
        None
    };
    let seed = point.load.seed;
    let count = point.load.num_prompts;
    let revision = point.dataset_revision.as_deref();
    let rows = match kind {
        Kind::Jsonl => jsonl(path.as_deref().unwrap_or(Path::new("")), count)?,
        Kind::ShareGpt => sharegpt(
            path,
            revision,
            seed,
            count,
            tokenizer.as_ref().context("a tokenizer")?,
        )?,
        Kind::Gsm8k => gsm8k(path, revision, count)?,
        Kind::Mjhq => mjhq(path, revision, seed, count)?,
        Kind::PieBench => pie_bench(path, seed, count)?,
        Kind::Beans => beans(path, revision, seed, count, point.question.as_deref())?,
        Kind::Ueval => ueval(path, revision, seed, count)?,
        Kind::FixedLength => fixed_length(
            path.as_deref().unwrap_or(Path::new("")),
            point.video.prompt_tokens,
            count,
            tokenizer.as_ref().context("a tokenizer")?,
        )?,
    };
    ensure!(
        rows.len() == count,
        "dataset {:?} resolved {} rows, expected num_prompts = {count}",
        point.dataset,
        rows.len()
    );
    Ok(rows)
}

/// The tokenizer of a local directory (its `tokenizer.json`) or of a Hub
/// repository.
///
/// # Errors
///
/// When neither holds a tokenizer.
pub fn tokenizer(name: &str) -> Result<tokenizers::Tokenizer> {
    let directory = Path::new(name);
    let file = if directory.is_dir() {
        directory.join("tokenizer.json")
    } else {
        hub(name, "model", None)?
            .get("tokenizer.json")
            .with_context(|| format!("the tokenizer of {name}"))?
    };
    tokenizers::Tokenizer::from_file(&file)
        .map_err(|error| anyhow::anyhow!("loading {}: {error}", file.display()))
}

/// A Hub repository of `kind` (`model` or `dataset`) at `revision`, through the
/// standard cache (`HF_HOME`, `HF_HUB_OFFLINE`).
fn hub(repository: &str, kind: &str, revision: Option<&str>) -> Result<hf_hub::api::sync::ApiRepo> {
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .context("the Hugging Face Hub client")?;
    let kind = match kind {
        "dataset" => hf_hub::RepoType::Dataset,
        _ => hf_hub::RepoType::Model,
    };
    let repo = match revision {
        Some(revision) => {
            hf_hub::Repo::with_revision(repository.to_owned(), kind, revision.to_owned())
        }
        None => hf_hub::Repo::new(repository.to_owned(), kind),
    };
    Ok(api.repo(repo))
}

/// One file of a Hub dataset at `revision`.
fn hub_file(repository: &str, file: &str, revision: Option<&str>) -> Result<PathBuf> {
    hub(repository, "dataset", revision)?
        .get(file)
        .with_context(|| format!("{file} of the dataset {repository}"))
}

/// The optional fields a JSON Lines row may carry beside `id` and `prompt`.
const JSONL_OPTIONAL: &[&str] = &[
    "messages",
    "prompt_len",
    "output_len",
    "max_tokens",
    "input_image_b64",
    "input_image_mime",
    "width",
    "height",
    "steps",
    "seed",
    "aspect_ratio",
    "num_frames",
    "input_references",
    "metadata",
];

/// Rows in file order up to `count`; later lines are neither parsed nor
/// checked.
fn jsonl(path: &Path, count: usize) -> Result<Vec<Example>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if rows.len() == count {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let number = number + 1;
        let raw: Value =
            serde_json::from_str(line).with_context(|| format!("jsonl row {number}"))?;
        let raw = raw
            .as_object()
            .with_context(|| format!("jsonl row {number} must be an object"))?;
        ensure!(
            raw.contains_key("id") && raw.contains_key("prompt"),
            "jsonl row {number} requires id and prompt"
        );
        ensure!(
            raw.contains_key("width") == raw.contains_key("height"),
            "jsonl row {number} must provide width and height together"
        );
        let mut fields = Map::new();
        fields.insert("id".into(), Value::String(text_of(&raw["id"])));
        fields.insert("prompt".into(), Value::String(text_of(&raw["prompt"])));
        for key in JSONL_OPTIONAL {
            if let Some(value) = raw.get(*key) {
                fields.insert((*key).to_owned(), value.clone());
            }
        }
        rows.push(
            serde_json::from_value(Value::Object(fields))
                .with_context(|| format!("jsonl row {number}"))?,
        );
    }
    Ok(rows)
}

/// A JSON value as Python's `str` renders a string or a number.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// ShareGPT's conversations with two turns or more, as `(prompt, completion)`
/// pairs, seeded, keeping pairs whose sides encode to two tokens or more (the
/// counts include the special tokens the tokenizer adds).
fn sharegpt(
    path: Option<PathBuf>,
    revision: Option<&str>,
    seed: u64,
    count: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<Example>> {
    let path = match path {
        Some(path) => path,
        None => hub_file(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
            revision,
        )?,
    };
    let data: Vec<Value> = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("{}", path.display()))?;
    let mut pairs: Vec<(String, String)> = data
        .iter()
        .filter_map(|row| {
            let turns = row
                .get("conversations")
                .or_else(|| row.get("conversation"))?
                .as_array()?;
            (turns.len() >= 2).then(|| {
                let side = |turn: &Value| turn.get("value").map(text_of).unwrap_or_default();
                (side(&turns[0]), side(&turns[1]))
            })
        })
        .collect();
    Random::new(seed).shuffle(&mut pairs);
    let mut rows = Vec::new();
    for (prompt, completion) in pairs {
        if rows.len() == count {
            break;
        }
        let length = |text: &str| -> Result<u64> {
            Ok(tokenizer
                .encode(text, true)
                .map_err(|error| anyhow::anyhow!("tokenizing: {error}"))?
                .len() as u64)
        };
        let (prompt_len, output_len) = (length(&prompt)?, length(&completion)?);
        if prompt_len < 2 || output_len < 2 {
            continue;
        }
        rows.push(Example {
            id: format!("sharegpt-{:06}", rows.len()),
            prompt,
            prompt_len: Some(prompt_len),
            output_len: Some(output_len),
            ..Example::default()
        });
    }
    Ok(rows)
}

/// The names a GSM8K directory may hold its test split under, in preference
/// order.
const GSM8K_FILES: &[&str] = &[
    "test.jsonl",
    "main/test-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
    "test.parquet",
];

/// GSM8K's test split in file order, neither shuffled nor filtered, up to
/// `count`; a row's id is `gsm8k-<position>` and its metadata holds its
/// reference answer.
fn gsm8k(path: Option<PathBuf>, revision: Option<&str>, count: usize) -> Result<Vec<Example>> {
    let path = match path {
        Some(path) if path.is_dir() => GSM8K_FILES
            .iter()
            .map(|name| path.join(name))
            .find(|candidate| candidate.is_file())
            .with_context(|| {
                format!(
                    "{} holds none of {}",
                    path.display(),
                    GSM8K_FILES.join(", ")
                )
            })?,
        Some(path) => path,
        None => hub_file("openai/gsm8k", "main/test-00000-of-00001.parquet", revision)?,
    };
    let suffix = path
        .extension()
        .map(|suffix| suffix.to_string_lossy().to_lowercase());
    let rows: Vec<Map<String, Value>> = match suffix.as_deref() {
        Some("parquet") => parquet_rows(&path)?,
        Some("jsonl") => json_lines(&path)?,
        _ => bail!(
            "gsm8k dataset_path must name a .parquet or .jsonl file, not {}",
            path.display()
        ),
    };
    let mut examples = Vec::new();
    for (position, row) in rows.iter().enumerate() {
        if examples.len() == count {
            break;
        }
        let field = |key: &str| {
            row.get(key)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
        };
        let question = field("question")
            .with_context(|| format!("gsm8k row {position} has no question: {}", path.display()))?;
        let answer = field("answer")
            .with_context(|| format!("gsm8k row {position} has no answer: {}", path.display()))?;
        let mut metadata = Map::new();
        metadata.insert("answer".into(), answer.into());
        examples.push(Example {
            id: format!("gsm8k-{:05}", examples.len()),
            prompt: question.to_owned(),
            metadata,
            ..Example::default()
        });
    }
    Ok(examples)
}

/// MJHQ-30K's metadata entries with a prompt, seeded.
fn mjhq(
    path: Option<PathBuf>,
    revision: Option<&str>,
    seed: u64,
    count: usize,
) -> Result<Vec<Example>> {
    let path = match path {
        Some(path) => path,
        None => hub_file("playgroundai/MJHQ-30K", "meta_data.json", revision)?,
    };
    let meta: Map<String, Value> = serde_json::from_slice(&std::fs::read(&path)?)?;
    let mut entries: Vec<(String, String)> = meta
        .iter()
        .filter_map(|(key, value)| {
            let prompt = value
                .as_object()?
                .get("prompt")
                .filter(|prompt| truthy(prompt))?;
            Some((key.clone(), text_of(prompt)))
        })
        .collect();
    Random::new(seed).shuffle(&mut entries);
    Ok(entries
        .into_iter()
        .take(count)
        .map(|(key, prompt)| Example {
            id: format!("mjhq-{key}"),
            prompt,
            ..Example::default()
        })
        .collect())
}

/// Python's truth of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

/// The mapping files a PIE-Bench root holds, in preference order.
const PIE_BENCH_MAPPINGS: &[&str] = &["mapping_file.json", "mapping_file_ti2i_benchmark.json"];

/// PIE-Bench's entries with an image, seeded before images and instructions are
/// checked, so an entry whose image or instruction is missing is skipped and the
/// next shuffled entry takes its place. Each image is sent as RGB PNG.
fn pie_bench(path: Option<PathBuf>, seed: u64, count: usize) -> Result<Vec<Example>> {
    let (mapping, images) = match path {
        Some(base) if base.is_file() => {
            let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
            (base, parent)
        }
        Some(base) => {
            let mapping = find_mapping(&base).with_context(|| {
                format!("no PIE-Bench mapping file found under {}", base.display())
            })?;
            (mapping, base)
        }
        None => {
            let snapshot = snapshot("UB-CVML-Group/PIE_Bench_pp")?;
            let mapping = find_mapping(&snapshot).with_context(|| {
                format!(
                    "no PIE-Bench mapping file found in snapshot {}",
                    snapshot.display()
                )
            })?;
            (mapping, snapshot)
        }
    };
    let entries: Map<String, Value> = serde_json::from_slice(&std::fs::read(&mapping)?)?;
    let mut entries: Vec<(String, Map<String, Value>)> = entries
        .into_iter()
        .filter_map(|(key, value)| {
            let value = value.as_object()?.clone();
            value.get("image_path").filter(|path| truthy(path))?;
            Some((key, value))
        })
        .collect();
    Random::new(seed).shuffle(&mut entries);
    let brackets = regex::Regex::new(r"[\[\]]")?;
    let mut rows = Vec::new();
    for (key, value) in entries {
        if rows.len() >= count {
            break;
        }
        let relative = text_of(&value["image_path"]);
        let Some(image) = image_file(&images, &relative) else {
            continue;
        };
        // A non-empty editing_instruction wins as written; else editing_prompt
        // or target_prompt, without their square brackets.
        let instruction = match value.get("editing_instruction").filter(|item| truthy(item)) {
            Some(instruction) => text_of(instruction).trim().to_owned(),
            None => value
                .get("editing_prompt")
                .filter(|item| truthy(item))
                .or_else(|| value.get("target_prompt").filter(|item| truthy(item)))
                .map(|target| brackets.replace_all(&text_of(target), "").trim().to_owned())
                .unwrap_or_default(),
        };
        if instruction.is_empty() {
            continue;
        }
        rows.push(Example {
            id: format!("pie-{key}"),
            prompt: instruction,
            input_image_b64: Some(png_base64(&image)?),
            input_image_mime: Some("image/png".into()),
            ..Example::default()
        });
    }
    Ok(rows)
}

/// The preferred mapping file under `root`: a known name directly under it, else
/// the first `mapping_file*.json` below it in name order.
fn find_mapping(root: &Path) -> Option<PathBuf> {
    if let Some(direct) = PIE_BENCH_MAPPINGS
        .iter()
        .map(|name| root.join(name))
        .find(|path| path.is_file())
    {
        return Some(direct);
    }
    let mut found: Vec<PathBuf> = walk(root)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("mapping_file") && name.ends_with(".json"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// An entry's image across the snapshot layouts: under the root and its parent,
/// each with and without `annotation_images`, else the first file below the root
/// of the same base name.
fn image_file(root: &Path, relative: &str) -> Option<PathBuf> {
    let parent = root.parent().unwrap_or(root);
    let candidates = [
        root.join(relative),
        root.join("annotation_images").join(relative),
        parent.join(relative),
        parent.join("annotation_images").join(relative),
    ];
    if let Some(found) = candidates.into_iter().find(|candidate| candidate.is_file()) {
        return Some(found);
    }
    let name = Path::new(relative).file_name()?;
    walk(root)
        .into_iter()
        .find(|path| path.file_name() == Some(name))
}

/// Every file below `root`, directories in name order.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        let mut entries: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect();
        entries.sort();
        for entry in entries.into_iter().rev() {
            if entry.is_dir() {
                pending.push(entry);
            } else {
                files.push(entry);
            }
        }
    }
    files
}

/// An image file as RGB PNG in base64: every format re-encoded, any alpha
/// dropped.
fn png_base64(path: &Path) -> Result<String> {
    let rgb = image::open(path)
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgb8();
    let mut encoded = Cursor::new(Vec::new());
    rgb.write_to(&mut encoded, image::ImageFormat::Png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()))
}

/// Beans' training split, its row indices seeded, each photograph sent as RGB
/// JPEG at quality 95 and asked `question`, else the default question. A row's
/// id is `beans-<position>-<source index>`.
fn beans(
    path: Option<PathBuf>,
    revision: Option<&str>,
    seed: u64,
    count: usize,
    question: Option<&str>,
) -> Result<Vec<Example>> {
    let files = match path {
        Some(path) => vec![path],
        None => hub_parquet("AI-Lab-Makerere/beans", revision, Some("train"))?,
    };
    let mut rows = Vec::new();
    for file in &files {
        rows.extend(parquet_rows(file)?);
    }
    let mut indices: Vec<usize> = (0..rows.len()).collect();
    Random::new(seed).shuffle(&mut indices);
    let question = question.unwrap_or(DEFAULT_I2T_QUESTION);
    let mut examples = Vec::new();
    for (position, index) in indices.into_iter().take(count).enumerate() {
        let bytes = rows[index]
            .get("image")
            .and_then(|image| image.get("bytes"))
            .and_then(Value::as_str)
            .with_context(|| format!("beans row {index} has no image bytes"))?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(bytes)?;
        let rgb = image::load_from_memory(&bytes)?.to_rgb8();
        let mut encoded = Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 95).encode_image(&rgb)?;
        examples.push(Example {
            id: format!("beans-{position:03}-{index}"),
            prompt: question.to_owned(),
            input_image_b64: Some(
                base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()),
            ),
            input_image_mime: Some("image/jpeg".into()),
            ..Example::default()
        });
    }
    Ok(examples)
}

/// The fields a UEval row may hold its prompt in, in preference order.
const UEVAL_PROMPT_FIELDS: &[&str] = &[
    "prompt",
    "question",
    "instruction",
    "input",
    "query",
    "text",
];

/// UEval's prompts: rows without one dropped before the seeded shuffle, ids
/// numbering the selected prompts in shuffled order.
fn ueval(
    path: Option<PathBuf>,
    revision: Option<&str>,
    seed: u64,
    count: usize,
) -> Result<Vec<Example>> {
    let rows: Vec<Map<String, Value>> = match path {
        Some(path)
            if path
                .extension()
                .is_some_and(|suffix| suffix.eq_ignore_ascii_case("parquet")) =>
        {
            parquet_rows(&path)?
        }
        Some(path) => {
            let text = std::fs::read_to_string(&path)?;
            if text.trim_start().starts_with('[') {
                let data: Vec<Value> = serde_json::from_str(&text)?;
                data.into_iter()
                    .filter_map(|row| row.as_object().cloned())
                    .collect()
            } else {
                text.lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(serde_json::from_str::<Value>)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .filter_map(|row| row.as_object().cloned())
                    .collect()
            }
        }
        None => {
            let mut rows = Vec::new();
            for file in hub_parquet("zlab-princeton/UEval", revision, None)? {
                rows.extend(parquet_rows(&file)?);
            }
            rows
        }
    };
    let mut prompts: Vec<String> = rows
        .iter()
        .filter_map(|row| {
            UEVAL_PROMPT_FIELDS.iter().find_map(|field| {
                row.get(*field)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            })
        })
        .collect();
    Random::new(seed).shuffle(&mut prompts);
    Ok(prompts
        .into_iter()
        .take(count)
        .enumerate()
        .map(|(index, prompt)| Example {
            id: format!("ueval-{index:06}"),
            prompt,
            ..Example::default()
        })
        .collect())
}

/// The fixed-length source: `{"base": TEXT, "filler": TEXT, "seed": N}`. The
/// prompt is the base's tokens truncated to `tokens`, or padded with whole or
/// partial copies of the filler's, decoded; it must encode back to exactly
/// `tokens` tokens (special tokens excluded). Every row repeats it with the
/// source's seed.
fn fixed_length(
    path: &Path,
    tokens: usize,
    count: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<Example>> {
    let source: Map<String, Value> = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("{} must be a JSON object", path.display()))?;
    let text = |key: &str| -> Result<&str> {
        source
            .get(key)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .with_context(|| format!("{} needs a non-empty {key:?}", path.display()))
    };
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("tokenizing: {error}"))?
            .get_ids()
            .to_vec())
    };
    let base = encode(text("base")?)?;
    let filler = encode(text("filler")?)?;
    ensure!(
        !base.is_empty() && !filler.is_empty(),
        "the tokenizer produced no prompt tokens"
    );
    let mut ids: Vec<u32> = base.into_iter().take(tokens).collect();
    while ids.len() < tokens {
        let missing = tokens - ids.len();
        ids.extend(filler.iter().take(missing));
    }
    let prompt = tokenizer
        .decode(&ids, true)
        .map_err(|error| anyhow::anyhow!("decoding: {error}"))?;
    let measured = encode(&prompt)?.len();
    ensure!(
        measured == tokens,
        "the fixed-length prompt measured {measured} tokens, expected {tokens}"
    );
    let seed = source.get("seed").and_then(Value::as_i64);
    Ok((0..count)
        .map(|index| Example {
            id: format!("fixed-length-{index:04}"),
            prompt: prompt.clone(),
            prompt_len: Some(tokens as u64),
            seed,
            ..Example::default()
        })
        .collect())
}

/// A JSON Lines file's object rows, in file order.
fn json_lines(path: &Path) -> Result<Vec<Map<String, Value>>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        let row = value.as_object().with_context(|| {
            format!(
                "JSON Lines row {} must be an object: {}",
                number + 1,
                path.display()
            )
        })?;
        rows.push(row.clone());
    }
    Ok(rows)
}

/// A Parquet file's rows in file order, as JSON objects: binary values in
/// base64, nested groups as objects.
fn parquet_rows(path: &Path) -> Result<Vec<Map<String, Value>>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader =
        SerializedFileReader::new(file).with_context(|| format!("reading {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        match row.to_json_value() {
            Value::Object(entries) => rows.push(entries),
            other => bail!(
                "a Parquet row of {} is not a group: {other}",
                path.display()
            ),
        }
    }
    Ok(rows)
}

/// The Parquet files of a Hub dataset at `revision`, in name order, those of
/// `split` alone when named (a file under a directory or with a name that starts
/// with the split).
fn hub_parquet(
    repository: &str,
    revision: Option<&str>,
    split: Option<&str>,
) -> Result<Vec<PathBuf>> {
    let repo = hub(repository, "dataset", revision)?;
    let info = repo
        .info()
        .with_context(|| format!("listing the dataset {repository}"))?;
    let mut names: Vec<String> = info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|name| name.ends_with(".parquet"))
        .filter(|name| {
            split.is_none_or(|split| {
                Path::new(name)
                    .components()
                    .any(|part| part.as_os_str().to_string_lossy().starts_with(split))
            })
        })
        .collect();
    names.sort();
    ensure!(
        !names.is_empty(),
        "the dataset {repository} holds no Parquet file"
    );
    names
        .iter()
        .map(|name| {
            repo.get(name)
                .with_context(|| format!("{name} of {repository}"))
        })
        .collect()
}

/// Every file of a Hub dataset at its default revision, in a local snapshot.
fn snapshot(repository: &str) -> Result<PathBuf> {
    let repo = hub(repository, "dataset", None)?;
    let info = repo
        .info()
        .with_context(|| format!("listing the dataset {repository}"))?;
    let mut root = None;
    for sibling in info.siblings {
        let file = repo.get(&sibling.rfilename)?;
        // The snapshot's root: the downloaded file's path less its name in the
        // repository.
        if root.is_none() {
            let depth = Path::new(&sibling.rfilename).components().count();
            root = file.ancestors().nth(depth).map(Path::to_path_buf);
        }
    }
    root.with_context(|| format!("the dataset {repository} holds no file"))
}
