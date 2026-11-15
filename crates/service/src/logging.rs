//! The one logging convention of every `uniserve` process: process-wide tracing configuration and
//! compact terminal log formatting.
//!
//! Every command calls [`init`] once with its process label and its [`Levels`], before it starts
//! threads or a Tokio runtime (a host service after it detached). Each event is written as one
//! line of the form `(<process> pid=<pid>) <LEVEL> <MM-DD HH:MM:SS> [<file>:<line>] <fields>`,
//! where the process is the label (`Daemon` for `uniserve serve`, `Engine` for an engine process,
//! `StorageServer`, `BuildServer` and the Storage Server's helpers for the host services), with
//! the active spans, if any, written as `outer{..}:inner{..}: ` before the fields.
//!
//! The levels are tracing's (`off`, `error`, `warn`, `info`, `debug`, `trace`) and come from the
//! command line alone: `--log-level` for every target, and the level a module states for the
//! targets it owns (the daemon's HTTP stack takes `--log-level-http`). The environment states
//! none. A child process of the same deployment is given its parent's level on its own command
//! line ([`arguments`]).
//!
//! The daemon and the engine log to stdout. A command whose own output is stdout
//! (`uniserve keys`) and a host service, whose standard error is the log beside its socket,
//! log to stderr ([`Output`]). ANSI colors follow the `fmt` layer's default: they are on unless
//! `NO_COLOR` is set to a non-empty value, with no terminal detection, so redirected output also
//! carries escape sequences.

use std::sync::OnceLock;
use std::{fmt, process};

use time::UtcOffset;
use time::macros::format_description;
use tracing::level_filters::LevelFilter;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt as _;

const CYAN: &str = "\x1b[0;36m";
const GREY: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const WHITE: &str = "\x1b[37m";
const RESET: &str = "\x1b[0m";
const LOCAL_TIME_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[month]-[day] [hour]:[minute]:[second]");

/// The level of every target when a process's command line states none.
pub const DEFAULT_LEVEL: LevelFilter = LevelFilter::INFO;

/// Where a process's log lines are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// Standard output: the log every process of a deployment shares.
    Stdout,
    /// Standard error, for a command whose standard output is its result.
    Stderr,
}

/// A process's log levels: one for every target, and one for each set of targets a module owns
/// and the command line names (the daemon's HTTP stack, `--log-level-http`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Levels {
    /// The level of every target that no entry of `targets` names.
    pub default: LevelFilter,
    /// Target prefixes and their levels, as tracing matches a target (`hyper` covers
    /// `hyper::proto::h1`), overriding `default`.
    pub targets: Vec<(&'static str, LevelFilter)>,
}

impl Levels {
    /// Every target at `default`, or at [`DEFAULT_LEVEL`] when the command line states none.
    pub fn new(default: Option<LevelFilter>) -> Self {
        Self {
            default: default.unwrap_or(DEFAULT_LEVEL),
            targets: Vec::new(),
        }
    }

    /// These levels with every target of `targets` at `level`; unchanged when no level is
    /// stated for them.
    pub fn with(mut self, targets: &[&'static str], level: Option<LevelFilter>) -> Self {
        if let Some(level) = level {
            self.targets
                .extend(targets.iter().map(|target| (*target, level)));
        }
        self
    }

    /// The command-line arguments that give a child process the level of every target
    /// (`--log-level`, in tracing's lowercase name). Target levels stay with the process whose
    /// module owns the targets.
    pub fn arguments(&self) -> Vec<String> {
        vec![
            "--log-level".to_owned(),
            self.default.to_string().to_ascii_lowercase(),
        ]
    }

    /// The tracing filter these levels select.
    fn filter(&self) -> Targets {
        Targets::new()
            .with_default(self.default)
            .with_targets(self.targets.iter().copied())
    }
}

/// The levels [`init`] installed: this process's one log configuration.
static INSTALLED: OnceLock<Levels> = OnceLock::new();

/// Installs the process-wide tracing subscriber with `levels`, writing to `output`, keeps
/// `levels` as the process's configuration ([`arguments`]), and reports every later panic of the
/// process as an error event of that log ([`report_panics`]).
///
/// Installation failure, for example because a global subscriber is already set, is reported on
/// stderr and is not fatal; panics then keep the default report on stderr.
pub fn init(process: &str, levels: &Levels, output: Output) {
    let formatter = UniserveEventFormatter::new(process);
    let writer = match output {
        Output::Stdout => BoxMakeWriter::new(std::io::stdout),
        Output::Stderr => BoxMakeWriter::new(std::io::stderr),
    };

    if let Err(err) = tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .event_format(formatter)
                .with_writer(writer)
                .with_filter(levels.filter()),
        )
        .try_init()
    {
        // The tracing subscriber itself failed to install, so we cannot log
        // through `tracing`. Surface the failure directly on stderr rather than
        // swallowing it, since it means subsequent log output may be missing.
        eprintln!("failed to install tracing subscriber: {err}");
        return;
    }
    let _ = INSTALLED.set(levels.clone());
    report_panics();
}

/// Replaces the default panic report with an error event of the process's log, which carries
/// the panic's message and location, the thread and, when `RUST_BACKTRACE` asks for one, the
/// backtrace.
///
/// The log is where a process's output goes whichever process holds its standard error: a
/// supervised process's standard error may be the pipe of a supervisor that is gone (a hot
/// restart's successor adopts its processes without that pipe), so the default report on
/// standard error alone could be lost.
fn report_panics() {
    std::panic::set_hook(Box::new(|panic| {
        let thread = std::thread::current();
        let backtrace = std::backtrace::Backtrace::capture();
        let backtrace = match backtrace.status() {
            std::backtrace::BacktraceStatus::Captured => format!("\n{backtrace}"),
            _ => String::new(),
        };
        tracing::error!(
            thread = thread.name().unwrap_or("<unnamed>"),
            "the process {panic}{backtrace}"
        );
    }));
}

/// The command-line arguments that give a child process of this one, such as an engine process
/// its daemon starts, this process's level of every target (`--log-level`); none before
/// [`init`]. Target levels stay with the process whose module owns the targets.
pub fn arguments() -> Vec<String> {
    INSTALLED.get().map(Levels::arguments).unwrap_or_default()
}

#[derive(Debug, Clone, Copy)]
struct UniserveLocalTimer {
    local_offset: UtcOffset,
}

impl Default for UniserveLocalTimer {
    /// Captures the local UTC offset once, falling back to UTC when it cannot
    /// be determined. The timer keeps this offset, so a later offset change
    /// such as a DST transition is not reflected in timestamps.
    fn default() -> Self {
        let local_offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
        Self { local_offset }
    }
}

impl FormatTime for UniserveLocalTimer {
    /// Writes the current time in the captured local offset as
    /// `MM-DD HH:MM:SS`.
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let now = time::OffsetDateTime::now_utc().to_offset(self.local_offset);
        let formatted = now.format(LOCAL_TIME_FORMAT).map_err(|_| fmt::Error)?;
        w.write_str(&formatted)
    }
}

#[derive(Debug, Clone)]
struct UniserveEventFormatter {
    prefix: String,
    timer: UniserveLocalTimer,
}

impl UniserveEventFormatter {
    /// Creates the formatter of `label`'s process, fixing the process prefix
    /// and the timer's local offset at construction.
    fn new(label: &str) -> Self {
        Self {
            prefix: format!("({label} pid={})", process::id()),
            timer: UniserveLocalTimer::default(),
        }
    }

    /// Writes the process prefix.
    fn write_process_prefix(&self, writer: &mut Writer<'_>, ansi: bool) -> fmt::Result {
        write_colored(writer, ansi, Some(CYAN), &self.prefix)?;
        writer.write_char(' ')
    }

    /// Writes tracing's name of the level, colored when `ansi` is set.
    fn write_level(&self, writer: &mut Writer<'_>, level: &Level, ansi: bool) -> fmt::Result {
        let (text, color) = match *level {
            Level::TRACE => ("TRACE", WHITE),
            Level::DEBUG => ("DEBUG", WHITE),
            Level::INFO => ("INFO", GREEN),
            Level::WARN => ("WARN", YELLOW),
            Level::ERROR => ("ERROR", RED),
        };
        write_colored(writer, ansi, Some(color), text)
    }

    /// Writes the event timestamp, or `<unknown time>` when it cannot be
    /// formatted.
    fn write_timestamp(&self, writer: &mut Writer<'_>, ansi: bool) -> fmt::Result {
        if ansi {
            writer.write_str(GREY)?;
        }
        if self.timer.format_time(writer).is_err() {
            writer.write_str("<unknown time>")?;
        }
        if ansi {
            writer.write_str(RESET)?;
        }
        Ok(())
    }

    /// Writes the source location when event metadata provides one: the path
    /// shortened by `shorten_file_path` unless `full_path` is set.
    fn write_location(
        &self,
        writer: &mut Writer<'_>,
        file: Option<&str>,
        line: Option<u32>,
        full_path: bool,
        ansi: bool,
    ) -> fmt::Result {
        let Some(file) = file else {
            return Ok(());
        };
        let file = if full_path {
            file
        } else {
            shorten_file_path(file)
        };
        if ansi {
            writer.write_str(GREY)?;
        }
        match line {
            Some(line) => write!(writer, "[{file}:{line}]")?,
            None => write!(writer, "[{file}]")?,
        }
        if ansi {
            writer.write_str(RESET)?;
        }
        Ok(())
    }

    /// Writes the active span hierarchy, root first, as
    /// `outer{fields}:inner{fields}: `, writing a span without formatted
    /// fields by name alone; writes nothing outside any span.
    fn write_scope<S, N>(&self, ctx: &FmtContext<'_, S, N>, writer: &mut Writer<'_>) -> fmt::Result
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
        N: for<'writer> FormatFields<'writer> + 'static,
    {
        let Some(scope) = ctx.event_scope() else {
            return Ok(());
        };

        let mut seen = false;
        for span in scope.from_root() {
            if seen {
                writer.write_str(":")?;
            }
            seen = true;
            writer.write_str(span.metadata().name())?;

            let ext = span.extensions();
            if let Some(fields) = ext.get::<FormattedFields<N>>()
                && !fields.is_empty()
            {
                write!(writer, "{{{fields}}}")?;
            }
        }

        if seen {
            writer.write_str(": ")?;
        }

        Ok(())
    }
}

impl<S, N> FormatEvent<S, N> for UniserveEventFormatter
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    /// Formats one tracing event with process, level, time, scope, and source context.
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();
        let ansi = writer.has_ansi_escapes();

        self.write_process_prefix(&mut writer, ansi)?;
        self.write_level(&mut writer, meta.level(), ansi)?;
        writer.write_char(' ')?;
        self.write_timestamp(&mut writer, ansi)?;
        writer.write_char(' ')?;

        // Use the full file path only when DEBUG (or more verbose) is enabled
        // for any target, independent of this event's level. Filenames alone
        // are often ambiguous, but full paths are too noisy for INFO-level
        // output.
        let full_path = LevelFilter::current() >= LevelFilter::DEBUG;
        self.write_location(&mut writer, meta.file(), meta.line(), full_path, ansi)?;
        writer.write_char(' ')?;
        self.write_scope(ctx, &mut writer)?;
        ctx.format_fields(writer.by_ref(), event)?;
        writer.write_char('\n')
    }
}

/// Shortens a source file path for log output while preserving enough context
/// for common Rust entrypoint and module filenames.
///
/// - For `mod.rs`, keep the parent directory as `parent/mod.rs`.
/// - For `src/lib.rs` and `src/main.rs`, keep one additional directory as `crate/src/lib.rs` or
///   `crate/src/main.rs` when available.
/// - Other files are displayed as just the basename.
fn shorten_file_path(file: &str) -> &str {
    let mut parts = file.rsplit('/');
    let name = parts.next().unwrap_or(file);
    let parent = parts.next();
    let grandparent = parts.next();

    let Some(parent) = parent else {
        return file;
    };

    if name == "mod.rs" {
        return &file[file.len() - parent.len() - 1 - name.len()..];
    }

    if !matches!(name, "lib.rs" | "main.rs") || parent != "src" {
        return name;
    }
    let Some(grandparent) = grandparent else {
        return file;
    };

    &file[file.len() - grandparent.len() - 1 - parent.len() - 1 - name.len()..]
}

/// Writes text with the requested ANSI style when color is enabled.
fn write_colored(
    writer: &mut Writer<'_>,
    ansi: bool,
    color: Option<&str>,
    text: &str,
) -> fmt::Result {
    if ansi {
        if let Some(color) = color {
            writer.write_str(color)?;
        }
        writer.write_str(text)?;
        if color.is_some() {
            writer.write_str(RESET)?;
        }
        return Ok(());
    }

    writer.write_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A level stated for a module's targets holds for those targets and their submodules, and
    /// every other target keeps the default; an unstated one changes nothing.
    #[test]
    fn target_levels_override_the_default_for_their_targets_alone() {
        let filter = Levels::new(Some(LevelFilter::ERROR))
            .with(&["hyper", "axum"], Some(LevelFilter::DEBUG))
            .with(&["reqwest"], None)
            .filter();

        assert!(filter.would_enable("hyper::proto::h1", &Level::DEBUG));
        assert!(filter.would_enable("axum", &Level::DEBUG));
        assert!(!filter.would_enable("reqwest", &Level::WARN));
        assert!(filter.would_enable("daemon::leader", &Level::ERROR));
        assert!(!filter.would_enable("daemon::leader", &Level::WARN));
    }

    /// A process whose command line states no level logs at INFO.
    #[test]
    fn an_unstated_level_is_info() {
        let filter = Levels::new(None).filter();

        assert!(filter.would_enable("engine", &Level::INFO));
        assert!(!filter.would_enable("engine", &Level::DEBUG));
    }

    #[test]
    fn location_path_uses_filename_for_non_ambiguous_files() {
        assert_eq!(shorten_file_path("src/cmd/src/logging.rs"), "logging.rs");
        assert_eq!(shorten_file_path("src/chat/lib.rs"), "lib.rs");
        assert_eq!(shorten_file_path("src/chat/main.rs"), "main.rs");
        assert_eq!(shorten_file_path("src/chat/src/xmod.rs"), "xmod.rs");
    }

    #[test]
    fn location_path_keeps_more_context_for_common_entrypoint_filenames() {
        assert_eq!(shorten_file_path("src/lib.rs"), "src/lib.rs");
        assert_eq!(shorten_file_path("src/chat/src/lib.rs"), "chat/src/lib.rs");
        assert_eq!(shorten_file_path("src/cmd/src/main.rs"), "cmd/src/main.rs");
        assert_eq!(shorten_file_path("mod.rs"), "mod.rs");
        assert_eq!(shorten_file_path("src/chat/src/tool/mod.rs"), "tool/mod.rs");
    }
}
