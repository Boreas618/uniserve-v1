//! What `nn`'s checkpoint layer makes of the FastH3 multi-component checkpoint.
//!
//! A real pipeline root is the one place `nn`'s checkpoint layer meets the format as it is
//! exported: several components, each with its own index and its own file identities, NVFP4 block
//! scales in `F8_E4M3` and packed weights in `F4_E2M1` — the two names this crate's header reader
//! maps itself, because the `safetensors` crate has neither. Synthetic fixtures state the rules;
//! this states what a deployment actually contains.
//!
//! It runs in whichever of two modes the host allows, and says which:
//!
//! - **indexes**, when the checkpoint's shards are not on this host. Every component is read from
//!   its own `*.index.json`, which names each tensor and the shard it lives in, so the components,
//!   their tensor counts, the names more than one component holds, the roster the pipeline metadata
//!   states, and the arithmetic the checkpoint does about its own files are all answerable without a
//!   byte of payload.
//! - **headers**, when the shards are here: the same report plus each tensor's element type, read
//!   through [`Checkpoint`] and this crate's own header reader.
//!
//! The union is attempted first, because that is what a model does: one `Checkpoint` over every
//! root, components enumerated in each, the earliest root shadowing the later ones per tensor. A
//! refusal from it is reported as it came, never worked around: a component whose index names shards
//! that are not there must fail by name rather than contribute zero tensors silently.
//!
//! Everything is derived from the checkpoint: components from the directories that state an index or
//! hold shards, tensor names from those indexes, the roster from `modular_model_index.json`, and the
//! file arithmetic from the checkpoint's own metadata and content manifests. Nothing is a typed list
//! of names, so a component the checkpoint gains or loses shows up as a difference.
//!
//! Usage, from the repository root:
//!
//! ```text
//! cargo run --release -p nn --example h3_checkpoint -- [/path/to/checkpoint ...]
//! ```
//!
//! The first root wins over the later ones, which is how a delta export shadows its base.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::Checkpoint;

/// The pipeline metadata that names components by role, which discovery by shard sets does not read:
/// it is compared against, never trusted as the list of what a checkpoint holds.
const MODEL_INDEX: &str = "modular_model_index.json";

/// One component of one root, as the checkpoint's own files state it.
struct Component {
    /// Tensor names, from the component's indexes.
    tensors: BTreeSet<String>,
    /// Shard file names the indexes name.
    named: BTreeSet<String>,
    /// Shard files actually present.
    present: BTreeSet<String>,
    /// The index's own `metadata.total_size`, when it states one.
    total_size: Option<u64>,
    /// The element types its headers decoded, when the shards are here.
    dtypes: BTreeMap<String, usize>,
    /// Why the headers could not be read, when they could not.
    unread: Option<String>,
}

fn main() -> ExitCode {
    let roots: Vec<PathBuf> = std::env::args()
        .skip(1)
        .filter(|argument| !argument.starts_with('-'))
        .map(PathBuf::from)
        .collect();
    if roots.is_empty() {
        eprintln!("name the deployment roots on the command line");
        return ExitCode::from(2);
    }
    let borrowed: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
    let mut defects: Vec<String> = Vec::new();
    let mut observations: Vec<String> = Vec::new();
    let mut host_state: Vec<String> = Vec::new();

    println!("== roots, in precedence order (the first wins)");
    for (index, root) in roots.iter().enumerate() {
        println!("  {index}: {}", root.display());
    }

    // Every component of every root, from the directories that state an index or hold shards.
    let mut per_root: Vec<BTreeMap<String, Component>> = Vec::new();
    let mut shards_here = 0_usize;
    for root in &roots {
        let mut components: BTreeMap<String, Component> = BTreeMap::new();
        for directory in sorted_directories(root) {
            let name = file_name(&directory);
            let stated = Statements::of(&directory);
            if stated.indexes.is_empty() && stated.present.is_empty() {
                continue;
            }
            shards_here += stated.present.len();
            components.insert(
                name,
                Component {
                    tensors: stated.tensors,
                    named: stated.named,
                    present: stated.present,
                    total_size: stated.total_size,
                    dtypes: BTreeMap::new(),
                    unread: None,
                },
            );
        }
        per_root.push(components);
    }
    let mode = if shards_here == 0 {
        Mode::Indexes
    } else {
        Mode::Headers
    };

    // The union, as a model opens it.
    let union = Checkpoint::open_roots(&borrowed);
    println!("\n== the union, as a model opens it");
    match &union {
        Ok(checkpoint) => println!(
            "  opened: {}, {} components, {} tensors at the top level, {} shards mapped",
            if checkpoint.is_pipeline() {
                "pipeline"
            } else {
                "flat"
            },
            checkpoint.components().count(),
            checkpoint.names().count(),
            checkpoint.files().len()
        ),
        Err(error) => {
            println!("  refused: {error}");
            defects.push(format!(
                "the union over {} root(s) cannot be opened: {error}",
                roots.len()
            ));
        }
    }

    match mode {
        Mode::Indexes => {
            println!(
                "\n== mode: indexes only — the checkpoint's shards are absent on this host, so no \
                 header was read and no element type is reported. The components, their tensor \
                 counts, the names more than one holds, the roster and the arithmetic below come \
                 from the checkpoint's own JSON."
            );
            host_state.push(
                "the checkpoint's shards are absent on this host: no header was read, so the \
                 element types the header reader maps (F8_E4M3, F4_E2M1 among them) are not \
                 exercised by this run"
                    .to_owned(),
            );
        }
        Mode::Headers => println!(
            "\n== mode: headers — {shards_here} shards are present, so every component is also \
             opened and its tensors' element types are decoded by this crate's header reader."
        ),
    }

    // The headers, when there are shards to read: done before the first root is borrowed.
    let mut census: BTreeMap<String, usize> = BTreeMap::new();
    if mode == Mode::Headers {
        for (path, components) in roots.iter().zip(per_root.iter_mut()) {
            for (name, component) in components.iter_mut() {
                let directory = path.join(name);
                match Checkpoint::open(&directory) {
                    Ok(opened) => {
                        for tensor in opened.names() {
                            if let Some(location) = opened.tensor(tensor) {
                                *census.entry(location.dtype.to_string()).or_default() += 1;
                                *component
                                    .dtypes
                                    .entry(location.dtype.to_string())
                                    .or_default() += 1;
                            }
                        }
                    }
                    Err(error) => {
                        component.unread = Some(error.to_string());
                        defects.push(format!(
                            "component `{name}` of {} states shards this crate cannot open: {error}",
                            path.display()
                        ));
                    }
                }
            }
        }
    }

    // The components of the first root, as state.
    println!("\n== components of {}", roots[0].display());
    {
        let first = &per_root[0];
        if first.is_empty() {
            defects.push(format!(
                "{} holds no component that states an index or a shard set",
                roots[0].display()
            ));
        }
        for (name, component) in first {
            let state = State::of(component);
            let mut line = format!(
                "  {name}: {} tensors, {} shards named, {} present — {}",
                component.tensors.len(),
                component.named.len(),
                component.present.len(),
                state.describe()
            );
            if let Some(total) = component.total_size {
                line.push_str(&format!(", {} bytes the index states", group(total)));
            }
            println!("{line}");
        }
        let total: usize = first
            .values()
            .map(|component| component.tensors.len())
            .sum();
        println!("  {} components, {total} tensors", first.len());

        // The last segment of each tensor name, counted: a quantized export states its packed
        // weights, block scales and tensor scales as suffixes of one name, and this is where the
        // suffix set a loader must consume is visible before a shard is read.
        println!("\n== name suffixes, which is how a quantized export states its fields");
        for (name, component) in first {
            let mut suffixes: BTreeMap<&str, usize> = BTreeMap::new();
            for tensor in &component.tensors {
                let last = tensor.rsplit('.').next().unwrap_or(tensor.as_str());
                *suffixes.entry(last).or_default() += 1;
            }
            let mut ordered: Vec<(&str, usize)> = suffixes.into_iter().collect();
            ordered.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(right.0)));
            println!(
                "  {name}: {}",
                ordered
                    .iter()
                    .map(|(suffix, count)| format!("{suffix} {count}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            );
            if let Some((_, count)) = ordered
                .iter()
                .find(|(suffix, _)| *suffix == "weight_packed")
            {
                observations.push(format!(
                    "`{name}` states {count} packed NVFP4 weights as `.weight_packed` beside \
                     `.weight_scale` and `.weight_tensor_scale`; a loader that reads \
                     `.weight`/`.weight_scale`/`.weight_scale_2` reads a different export's \
                     vocabulary"
                ));
            }
        }
    }

    // Element types, when the shards allowed them.
    if mode == Mode::Headers {
        println!("\n== element types, decoded by this crate's header reader");
        for (dtype, count) in &census {
            let holders: Vec<&str> = per_root[0]
                .iter()
                .filter(|(_, component)| component.dtypes.contains_key(dtype))
                .map(|(name, _)| name.as_str())
                .collect();
            println!("  {dtype}: {count} tensors, in {}", holders.join(", "));
        }
        if census.is_empty() {
            println!("  none: no component's headers were read");
        }
    }

    // Names more than one component holds, read from the indexes. Within one root, components are
    // separate namespaces: a pipeline reads `vae.decoder.x` and `transformer.x` as different
    // tensors, so nothing is shadowed inside a root and the caller names the component it reads.
    println!("\n== names held by more than one component");
    {
        let first = &per_root[0];
        let mut holders: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (name, component) in first {
            for tensor in &component.tensors {
                holders
                    .entry(tensor.as_str())
                    .or_default()
                    .push(name.as_str());
            }
        }
        let shared: Vec<(&str, &Vec<&str>)> = holders
            .iter()
            .filter(|(_, components)| components.len() > 1)
            .map(|(tensor, components)| (*tensor, components))
            .collect();
        if shared.is_empty() {
            println!(
                "  none: every tensor name is held by exactly one component ({} names compared), so \
                 no two components can be read for one name and nothing in this root is resolved \
                 by precedence",
                holders.len()
            );
        } else {
            for (tensor, components) in &shared {
                println!(
                    "  {tensor}: held by {} — a pipeline keeps them apart by component, so no \
                     component wins it and the caller names the one it reads",
                    components.join(", ")
                );
            }
            println!("  {} shared names, listed above in full", shared.len());
        }
    }

    // Across roots: the precedence rule, made visible rather than asserted.
    println!("\n== precedence across roots");
    if roots.len() < 2 {
        println!(
            "  not exercised: one root shadows nothing. A second root makes this section compare \
             each root's names with the first root's."
        );
    } else {
        let mut shadowed = 0_usize;
        let mut shown = 0_usize;
        for (index, (root, components)) in roots.iter().zip(per_root.iter()).enumerate() {
            let names = all_names(components);
            for (later_root, later) in roots.iter().zip(per_root.iter()).skip(index + 1) {
                let later_names = all_names(later);
                let both: Vec<&String> = names.intersection(&later_names).collect();
                if both.is_empty() {
                    continue;
                }
                shadowed += both.len();
                if shown < 8 {
                    println!(
                        "  {} shadows {} on {} names, the first of them `{}`",
                        root.display(),
                        later_root.display(),
                        both.len(),
                        both[0]
                    );
                    shown += 1;
                }
            }
        }
        if shadowed == 0 {
            println!("  none: the roots hold no tensor name in common");
        } else {
            println!("  {shadowed} shadowed names in total, the earlier root winning each");
        }
    }

    // The arithmetic the checkpoint does about its own files, which fails when a component is
    // truncated. Every number and every path here is the checkpoint's.
    println!("\n== what the checkpoint states about its own files");
    let mut checked = 0_usize;
    // The component the checkpoint's own metadata says its shards belong to, which the manifest
    // section below reads its reference name from.
    let mut stated_module: Option<String> = None;
    for entry in root_files(&roots[0]) {
        let Some(parsed) = read_json(&entry) else {
            continue;
        };
        if let Some(module) = parsed.get("module").and_then(|module| module.as_str()) {
            checked += 1;
            let stated = parsed
                .get("shard_count")
                .and_then(serde_json::Value::as_u64);
            let sizes: Vec<u64> = parsed
                .get("shard_sizes")
                .and_then(|sizes| sizes.as_array())
                .map(|sizes| sizes.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default();
            stated_module = Some(module.to_owned());
            println!(
                "  {}: module `{module}`, shard count {}, sizes of {} shards",
                file_name(&entry),
                stated.map_or("unstated".to_owned(), |count| count.to_string()),
                sizes.len()
            );
            match per_root[0].get(module) {
                Some(component) => {
                    if let Some(stated) = stated
                        && stated as usize != component.named.len()
                    {
                        defects.push(format!(
                            "`{}` states {stated} shards for `{module}` and its index names {}",
                            file_name(&entry),
                            component.named.len()
                        ));
                    }
                    let bytes: u64 = sizes.iter().sum();
                    if !sizes.is_empty()
                        && let Some(total) = component.total_size
                        && bytes != total
                    {
                        defects.push(format!(
                            "`{}` sizes `{module}`'s payload at {bytes} bytes and its index states \
                             {total}",
                            file_name(&entry)
                        ));
                    }
                    if !sizes.is_empty() && sizes.len() == component.named.len() {
                        println!(
                            "    agrees with `{module}`: {} shards named, {} bytes of payload",
                            component.named.len(),
                            group(bytes)
                        );
                    }
                }
                None => observations.push(format!(
                    "`{}` states module `{module}`, which holds neither an index nor a shard in \
                     this root",
                    file_name(&entry)
                )),
            }
        }
        if let Some(files) = parsed.get("files").and_then(|files| files.as_array()) {
            checked += 1;
            let paths: Vec<(String, u64)> = files
                .iter()
                .filter_map(|file| {
                    Some((
                        file.get("path")?.as_str()?.to_owned(),
                        file.get("bytes")?.as_u64()?,
                    ))
                })
                .collect();
            // Attribute the manifest to the component whose index names the same shards: the
            // checkpoint does not say which component a content list describes.
            let shards: BTreeSet<String> = paths
                .iter()
                .map(|(path, _)| file_name(Path::new(path)))
                .filter(|name| name.ends_with(".safetensors"))
                .collect();
            let owner = per_root[0]
                .iter()
                .max_by_key(|(_, component)| component.named.intersection(&shards).count())
                .filter(|(_, component)| !component.named.intersection(&shards).next().is_none())
                .map(|(name, _)| name.clone());
            println!(
                "  {}: {} files, {} of them shards{}",
                file_name(&entry),
                paths.len(),
                shards.len(),
                owner
                    .as_ref()
                    .map(|owner| format!(", describing `{owner}`"))
                    .unwrap_or_default()
            );
            if let Some(owner) = owner {
                let directory = roots[0].join(&owner);
                let mut present = 0_usize;
                let mut wrong_size: Vec<String> = Vec::new();
                for (path, bytes) in &paths {
                    let absolute = directory.join(path);
                    if let Ok(metadata) = std::fs::metadata(&absolute) {
                        present += 1;
                        if metadata.len() != *bytes {
                            wrong_size.push(format!(
                                "`{path}` is {} bytes and the manifest says {bytes}",
                                metadata.len()
                            ));
                        }
                    }
                }
                if present == 0 {
                    println!(
                        "    0 of {} files are on this host: the payload is not staged here",
                        paths.len()
                    );
                    host_state.push(format!(
                        "the content manifest lists {} files for `{owner}` and none of them is on \
                         this host, so no byte size was compared",
                        paths.len()
                    ));
                } else {
                    println!("    {present} of {} files present", paths.len());
                    if present < paths.len() {
                        defects.push(format!(
                            "`{}` lists {} files for `{owner}` and {present} are on this host",
                            file_name(&entry),
                            paths.len()
                        ));
                    }
                    for wrong in &wrong_size {
                        defects.push(format!("`{}`: {wrong}", file_name(&entry)));
                    }
                }
            }
        }
    }
    if checked == 0 {
        println!(
            "  none: this root states no shard count and no content list, so there is nothing here \
             to compare against"
        );
    }

    // The quantization manifest's module names, against the index's tensor names. Two vocabularies
    // meet here — the training repository's module paths and the export's tensor names — and a
    // model's key mapping is exactly what has to bridge them, so the mismatch is reported rather
    // than smoothed over.
    println!("\n== the quantization manifest's module names against the index");
    let mut reported = 0_usize;
    for entry in root_files(&roots[0]) {
        let Some(parsed) = read_json(&entry) else {
            continue;
        };
        let Some(components) = parsed.get("components").and_then(|it| it.as_object()) else {
            continue;
        };
        for (label, component) in components {
            let Some(modules) = component.get("modules").and_then(|it| it.as_object()) else {
                continue;
            };
            if modules.is_empty() {
                continue;
            }
            // The component searched is the one the checkpoint's metadata names, because the
            // manifest's label is the training repository's name for it and not a directory's.
            let names: BTreeSet<&str> = match &stated_module {
                Some(module) if per_root[0].contains_key(module) => per_root[0][module]
                    .tensors
                    .iter()
                    .map(String::as_str)
                    .collect(),
                _ => per_root[0]
                    .values()
                    .flat_map(|component| component.tensors.iter().map(String::as_str))
                    .collect(),
            };
            let resolved = modules
                .keys()
                .filter(|module| {
                    let stem = module
                        .strip_prefix(&format!("{label}."))
                        .unwrap_or(module.as_str());
                    names.iter().any(|tensor| tensor.starts_with(stem))
                })
                .count();
            println!(
                "  {}: `{label}` names {} modules, {resolved} of their stems prefix a tensor name",
                file_name(&entry),
                modules.len()
            );
            if resolved == 0 && reported < 2 {
                // Two vocabularies meet here and one pair says it better than a count: the
                // training repository's module path, and the export's name for the same block.
                if let Some(module) = modules.keys().next() {
                    println!(
                        "    manifest `{module}`, index `{}`",
                        names
                            .iter()
                            .next()
                            .map_or("no tensor to compare against".to_owned(), |name| {
                                (*name).to_owned()
                            })
                    );
                    reported += 1;
                }
            }
        }
    }

    // The roster the pipeline metadata states, against what the root holds. Read by shape: a
    // component entry is a list whose last element names its subfolder and where it comes from.
    println!("\n== component roster in {MODEL_INDEX}, against what the first root holds");
    let mut declared: BTreeSet<String> = BTreeSet::new();
    let index = roots[0].join(MODEL_INDEX);
    match read_json(&index) {
        Some(value) => {
            if let Some(entries) = value.as_object() {
                println!("  {:<18} {:<44} {:<30} state", "role", "class", "source");
                for (role, entry) in entries {
                    let Some(parts) = entry.as_array() else {
                        continue;
                    };
                    let options = parts.last().and_then(|options| options.as_object());
                    let subfolder = options
                        .and_then(|options| options.get("subfolder"))
                        .and_then(|subfolder| subfolder.as_str());
                    let Some(subfolder) = subfolder else {
                        continue;
                    };
                    declared.insert(subfolder.to_owned());
                    let class = parts
                        .get(1)
                        .and_then(|class| class.as_str())
                        .unwrap_or("")
                        .to_owned();
                    let source = options
                        .and_then(|options| options.get("pretrained_model_name_or_path"))
                        .and_then(|source| source.as_str())
                        .unwrap_or("")
                        .to_owned();
                    let state = per_root[0]
                        .get(subfolder)
                        .map(State::of)
                        .map_or("no index, no shard".to_owned(), |state| state.describe());
                    println!("  {role:<18} {class:<44} {source:<30} {state}");
                }
            }
        }
        None => defects.push(format!("{MODEL_INDEX} cannot be read as JSON")),
    }
    let held: BTreeSet<String> = per_root[0].keys().cloned().collect();
    let missing: Vec<&String> = declared.difference(&held).collect();
    let extra: Vec<&String> = held.difference(&declared).collect();
    for name in &extra {
        observations.push(format!(
            "`{name}` states an index or holds a shard and {MODEL_INDEX} does not declare it"
        ));
    }
    if !missing.is_empty() {
        // A declared component with no weights here is not automatically a defect: the pipeline
        // index also declares components that never carry weights (schedulers, tokenizers,
        // processors), and it names a source for each. What the metadata cannot say is which of
        // the weight-bearing ones the deployment expects from this root, so the list is an
        // observation and the loader's rule is `open_roots`.
        println!(
            "  {} declared components hold no weights in this root: {}",
            missing.len(),
            missing
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
        observations.push(format!(
            "{} components declared in {MODEL_INDEX} hold no weights in this root; a component \
             without an index and without a shard is indistinguishable in the metadata from one \
             that never carries weights, so this is reported rather than called a defect",
            missing.len()
        ));
    }

    // A partial payload is a defect even in index mode: shards here and some the index names
    // missing is a truncated checkpoint, not a host that never had the payload.
    for (name, component) in &per_root[0] {
        if !component.present.is_empty() && component.present != component.named {
            let absent: Vec<&String> = component.named.difference(&component.present).collect();
            defects.push(format!(
                "component `{name}` holds {} of the {} shards its index names, first absent `{}`",
                component.present.len(),
                component.named.len(),
                absent.first().map_or("", |name| name.as_str())
            ));
        }
    }

    println!("\n== observations, not defects");
    if observations.is_empty() {
        println!("  none");
    } else {
        for note in &observations {
            println!("  - {note}");
        }
    }

    println!("\n== host state, not defects");
    if host_state.is_empty() {
        println!("  none: the shards are here, so both modes' evidence is in this report");
    } else {
        for note in &host_state {
            println!("  - {note}");
        }
    }

    println!("\n== defects");
    if defects.is_empty() {
        println!(
            "  none found. The lists above are the observations: a run that reports zero shared \
             names has found nothing, and cannot by itself tell an empty list from a rule that \
             accepts everything."
        );
    } else {
        for defect in &defects {
            println!("  - {defect}");
        }
        println!(
            "\n  {} defect(s); the first is: {}",
            defects.len(),
            defects[0]
        );
    }
    if defects.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Which evidence this run could gather.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The indexes only: no shard on this host.
    Indexes,
    /// The indexes and the shards' headers.
    Headers,
}

/// What a component's directory states about its payload.
enum State {
    /// An index and every shard it names.
    Complete,
    /// An index whose shards are absent: the payload was not staged here, or was not shipped.
    IndexOnly,
    /// Shards and no index: a single-file component, or an index that was not written.
    ShardsOnly,
    /// Some of the index's shards.
    Partial,
}

impl State {
    fn of(component: &Component) -> Self {
        if component.named.is_empty() {
            if component.present.is_empty() {
                Self::Partial
            } else {
                Self::ShardsOnly
            }
        } else if component.present.is_empty() {
            Self::IndexOnly
        } else if component.present == component.named {
            Self::Complete
        } else {
            Self::Partial
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Complete => "index and every shard it names".to_owned(),
            Self::IndexOnly => "index only, no shard here".to_owned(),
            Self::ShardsOnly => "shards without an index".to_owned(),
            Self::Partial => "part of the index's shards".to_owned(),
        }
    }
}

/// What a directory states about itself, from its own files.
struct Statements {
    /// Tensor names, from every `*.index.json` in the directory.
    tensors: BTreeSet<String>,
    /// Shard paths those indexes name.
    named: BTreeSet<String>,
    /// Shards present, by file name.
    present: BTreeSet<String>,
    /// The index's `metadata.total_size`, when it states one.
    total_size: Option<u64>,
    /// The index files it has.
    indexes: Vec<PathBuf>,
}

impl Statements {
    fn of(directory: &Path) -> Self {
        let mut statements = Self {
            tensors: BTreeSet::new(),
            named: BTreeSet::new(),
            present: BTreeSet::new(),
            total_size: None,
            indexes: Vec::new(),
        };
        let Ok(entries) = std::fs::read_dir(directory) else {
            return statements;
        };
        for entry in entries.filter_map(std::result::Result::ok) {
            let path = entry.path();
            let name = file_name(&path);
            if name.ends_with(".index.json") {
                statements.indexes.push(path.clone());
                let Some(parsed) = read_json(&path) else {
                    continue;
                };
                if let Some(map) = parsed.get("weight_map").and_then(|map| map.as_object()) {
                    for (tensor, shard) in map {
                        statements.tensors.insert(tensor.clone());
                        if let Some(shard) = shard.as_str() {
                            statements.named.insert(shard.to_owned());
                        }
                    }
                }
                if let Some(total) = parsed
                    .get("metadata")
                    .and_then(|metadata| metadata.get("total_size"))
                    .and_then(serde_json::Value::as_u64)
                {
                    statements.total_size = Some(total);
                }
            } else if name.ends_with(".safetensors") {
                statements.present.insert(name);
            }
        }
        statements.indexes.sort();
        statements
    }
}

/// The JSON files at a directory's top level, in name order.
fn root_files(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "json")
        })
        .collect();
    files.sort();
    files
}

/// A file parsed as JSON, or `None` when it cannot be read or parsed.
fn read_json(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// The directories of `root`, in name order.
fn sorted_directories(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut directories: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    directories.sort();
    directories
}

/// A path's last component, or the empty string.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Every tensor name a root's components hold.
fn all_names(components: &BTreeMap<String, Component>) -> BTreeSet<String> {
    components
        .values()
        .flat_map(|component| component.tensors.iter().cloned())
        .collect()
}

/// A byte count with thousands separators, because these are checkpoint sizes.
fn group(bytes: u64) -> String {
    let digits = bytes.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(' ');
        }
        grouped.push(digit);
    }
    grouped
}
