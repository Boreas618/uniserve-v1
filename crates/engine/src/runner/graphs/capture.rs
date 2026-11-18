//! The device side of a captured graph: capturing a forward into segments, and the executable a
//! replay launches.
//!
//! A capture runs a bucket's forward once on a capturing stream: its launches become the nodes
//! of the stream's graph, and nothing runs. A graph break (`ops::graph_break`) ends the segment
//! being captured and begins the next; the host work it states is kept with the segment before
//! it. The capture is split in two steps, each timed by its caller: [`Recording::capture`]
//! records the templates, [`Recording::instantiate`] makes the executable a replay launches.
//!
//! The capture in progress is this thread's ([`SESSION`]): a capture is begun, recorded and
//! ended on one thread, the capture thread of its device, and the driver's capture object is
//! bound to that thread.

use std::cell::RefCell;
use std::sync::Arc;

use ops::breaks::HostWork;
use ops::{Generation, Kernel};
use platform::{Capture, CaptureMode, Graph, GraphExec, GraphNodeKind, Stream};

use crate::worker::{Error, Result};

thread_local! {
    /// The capture in progress on this thread, or none.
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// One capture in progress: the segment being captured and the segments already ended.
struct Session {
    stream: Stream,
    /// The driver's capture of the current segment; `None` once a failure ended it.
    capture: Option<Capture>,
    /// The segments a graph break ended, each with the break's host work.
    ended: Vec<(Graph, HostWork)>,
}

impl Session {
    /// Ends the current segment at a graph break stating `work`, and begins the next.
    fn broken(&mut self, work: HostWork) -> platform::Result<()> {
        let Some(capture) = self.capture.take() else {
            return Ok(());
        };

        self.ended.push((capture.end()?, work));
        self.capture = Some(Capture::begin(&self.stream, CaptureMode::Relaxed)?);
        Ok(())
    }
}

/// The break taker of a capturing context: ends the segment being captured on this thread and
/// begins the next, keeping the break's host work.
pub(crate) fn take_break(work: HostWork) -> ops::Result<()> {
    SESSION.with(|session| match session.borrow_mut().as_mut() {
        Some(session) => session.broken(work).map_err(ops::Error::Platform),
        None => Err(ops::Error::Break(
            "a graph break was stated on a thread that captures nothing".into(),
        )),
    })
}

/// The templates one capture recorded: one graph per stretch between graph breaks, each with the
/// host work of the break that ends it (`None` for the last).
pub(crate) struct Recording {
    segments: Vec<(Graph, Option<HostWork>)>,
}

impl Recording {
    /// Captures `forward` on `stream` in relaxed mode, so the work other threads issue on their
    /// own streams meanwhile (the device's serving forwards) is neither captured nor disturbs
    /// the capture. Returns the templates and what `forward` returned.
    ///
    /// The caller installed a context on this thread that launches on `stream` and whose break
    /// taker is [`take_break`].
    ///
    /// # Errors
    ///
    /// When another capture is in progress on this thread, `forward` fails, or the driver
    /// refuses the capture.
    pub(crate) fn capture<T>(
        stream: &Stream,
        forward: impl FnOnce() -> Result<T>,
    ) -> Result<(Self, T)> {
        let capture = Capture::begin(stream, CaptureMode::Relaxed)?;
        let began = SESSION.with(|session| {
            let mut session = session.borrow_mut();
            if session.is_some() {
                return false;
            }

            *session = Some(Session {
                stream: stream.clone(),
                capture: Some(capture),
                ended: Vec::new(),
            });
            true
        });
        if !began {
            return Err(Error::State(
                "a capture is already in progress on this thread",
            ));
        }

        let output = forward();

        // The session is taken back whatever the forward returned: dropping it ends the
        // driver's capture, so a failed forward leaves the stream usable.
        let Some(mut session) = SESSION.with(|session| session.borrow_mut().take()) else {
            return Err(Error::Invariant(
                "the capture's session was taken during its forward".into(),
            ));
        };
        let output = output?;

        let last = session
            .capture
            .take()
            .ok_or_else(|| Error::Invariant("the capture ended before its forward did".into()))?
            .end()?;

        let mut segments: Vec<(Graph, Option<HostWork>)> = session
            .ended
            .into_iter()
            .map(|(graph, work)| (graph, Some(work)))
            .collect();
        segments.push((last, None));
        Ok((Self { segments }, output))
    }

    /// Instantiates every segment (`cuGraphInstantiate`): the executable a replay launches,
    /// holding `kernels`, every kernel its segments call.
    ///
    /// # Errors
    ///
    /// When the driver refuses an instantiation or a template's nodes cannot be read.
    pub(crate) fn instantiate(self, kernels: Vec<Arc<Kernel>>) -> Result<Executable> {
        // Read before the segments are instantiated, so that a renewal declared meanwhile
        // leaves the executable stale rather than counted current.
        let renewals = crate::profiling::graph_renewals();

        let mut segments = Vec::with_capacity(self.segments.len());
        for (template, then) in self.segments {
            segments.push(Segment {
                exec: template.instantiate()?,
                template,
                then,
            });
        }

        let kernel_nodes = Executable::count_kernel_nodes(&segments)?;
        Ok(Executable {
            segments,
            _kernels: kernels,
            kernel_nodes,
            renewals,
        })
    }
}

/// One captured segment: the kernels and copies between two graph breaks.
///
/// The fields are declared in the order they are released: the executable first, then the
/// template it was instantiated from.
struct Segment {
    /// The executable a replay launches.
    exec: GraphExec,
    /// The graph the executable was instantiated from, which a renewal instantiates again.
    template: Graph,
    /// The host work of the graph break that ends the segment; `None` for the last segment.
    then: Option<HostWork>,
}

/// One bucket's captured forward on the device: its segments and the kernels they call.
///
/// The segments are released before the kernels, so a kernel stays loaded while a graph that
/// calls it can still be launched; the driver releases an executable still in flight once it
/// completes.
pub(crate) struct Executable {
    segments: Vec<Segment>,
    /// Every kernel the segments call, held so none is unloaded while the executable can be
    /// launched.
    _kernels: Vec<Arc<Kernel>>,
    /// The kernel nodes one replay launches, over every segment: what a replay adds to the
    /// device's count of kernel launches.
    kernel_nodes: u64,
    /// The process's count of graph renewals ([`crate::profiling::graph_renewals`]) read before
    /// the segments were last instantiated: the executable is current while the count still
    /// reads this ([`Executable::refresh`]).
    renewals: u64,
}

impl Executable {
    /// Replays the executable on `stream`, after the work already queued on it: each segment's
    /// launch, and between two segments the host work of the break that separates them.
    ///
    /// # Errors
    ///
    /// A launch the driver refuses, or a break's host work that fails.
    pub(crate) fn launch(&self, stream: &Stream) -> Result<()> {
        for segment in &self.segments {
            segment.exec.launch(stream)?;

            if let Some(work) = &segment.then {
                work().map_err(|source| Error::from(ops::Error::Break(source)))?;
            }
        }
        Ok(())
    }

    /// Instantiates every segment's executable again from its template when the process
    /// declared every executable stale since this one was instantiated
    /// ([`crate::profiling::graph_renewals`]): what a replay needs after the process's CUPTI
    /// recording changed how graphs are traced, or the storage the graph addresses was mapped
    /// again. A current executable is left as it is.
    ///
    /// Each executable is renewed before its own next replay, so a renewal costs the
    /// instantiation of the buckets that run, one at a time as each first runs again, and never
    /// holds a batch for the instantiation of every bucket the runner captured.
    ///
    /// # Errors
    ///
    /// When the driver refuses an instantiation; the executable then stays stale, and the next
    /// replay instantiates it again.
    pub(crate) fn refresh(&mut self) -> Result<()> {
        let renewals = crate::profiling::graph_renewals();
        if renewals == self.renewals {
            return Ok(());
        }

        for segment in &mut self.segments {
            // An executable still in flight is released by the driver when it completes.
            segment.exec = segment.template.instantiate()?;
        }
        self.renewals = renewals;
        Ok(())
    }

    /// The kernel nodes one replay launches.
    pub(crate) fn kernel_nodes(&self) -> u64 {
        self.kernel_nodes
    }

    /// The segments of the executable: one more than its graph breaks.
    pub(crate) fn segments(&self) -> usize {
        self.segments.len()
    }

    /// The kernel nodes of `segments`' templates, read from the driver once per executable.
    fn count_kernel_nodes(segments: &[Segment]) -> Result<u64> {
        let mut kernels = 0;
        for segment in segments {
            for node in segment.template.nodes()? {
                if node.kind()? == GraphNodeKind::Kernel {
                    kernels += 1;
                }
            }
        }
        Ok(kernels)
    }
}

/// The kernels of `generation` that the launches of `table` call, each once.
///
/// A graph holds them so that a kernel stays loaded for as long as a graph that calls it can be
/// replayed.
///
/// # Errors
///
/// When the generation holds no kernel for a launch of the table.
pub(crate) fn kernels_of(
    generation: &Generation,
    table: &[(&'static str, String, String)],
) -> Result<Vec<Arc<Kernel>>> {
    let mut held: Vec<Arc<Kernel>> = Vec::new();

    for (operator, key, _) in table {
        if *operator == ops::breaks::BREAK {
            continue;
        }

        let kernel = generation.kernel(operator, key)?;
        if !held.iter().any(|existing| Arc::ptr_eq(existing, kernel)) {
            held.push(Arc::clone(kernel));
        }
    }
    Ok(held)
}
