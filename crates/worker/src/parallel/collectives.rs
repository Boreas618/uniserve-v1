//! Stream-ordered NCCL collectives and registered window lifetime.

use std::sync::Arc;

use indexmap::IndexMap;
use uniserve_cuda::nccl::{self, Communicator, CtaPolicy, DataType, ReduceOp, UniqueId};
use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};

use crate::{CUDAStream, Error, Result};

/// An NCCL communicator bound to one computation stream.
///
/// Synchronous operations run on the computation stream. Asynchronous
/// operations run on a transfer stream forked from it, which shares its SM
/// partition; consumers join them by event. Registered window backing lives
/// through communicator retirement, independently of the execution contexts
/// and graphs borrowing it.
pub struct StreamCommunicator<T> {
    device: DeviceId,
    resources: Option<Resources<T>>,
}

struct Resources<T> {
    communicator: Communicator,
    // A borrowed view of the owner's stream; the owner outlives this value.
    computation: Stream,
    transfer: CUDAStream,
    // The latest asynchronous operation not yet joined on the computation
    // stream. Synchronous operations order themselves after it.
    pending: Option<Arc<Event>>,
    // Registered `(address, bytes)` windows and the owners of their memory.
    windows: IndexMap<(u64, usize), T>,
}

impl<T> StreamCommunicator<T> {
    /// Every group member calls this once, with the same distributed unique ID.
    ///
    /// Communicators of full-device streams use NCCL's copy-engine algorithms
    /// for registered windows; SM-partitioned streams use its CTA algorithms.
    ///
    /// # Safety
    /// `owner` outlives the returned communicator: operations keep submitting
    /// to its stream until [`Self::close`].
    pub unsafe fn new(owner: &CUDAStream, size: i32, rank: i32, id: &UniqueId) -> Result<Self> {
        let device = owner.device();
        // A borrowed default-stream owner resolves against the current context.
        let _device = DeviceGuard::new(device)?;
        // SAFETY: the caller keeps the owner, and therefore its stream, alive.
        let computation = unsafe { Stream::borrowed(device, owner.handle()?) }?;

        let policy = if owner.full_device() {
            CtaPolicy::Zero
        } else {
            CtaPolicy::Default
        };
        let communicator = Communicator::init(device, size, rank, id, policy)?;

        let transfer = match owner.fork() {
            Ok(stream) => stream,
            Err(error) => {
                let _ = communicator.abort();
                return Err(error);
            }
        };

        Ok(Self {
            device,
            resources: Some(Resources {
                communicator,
                computation,
                transfer,
                pending: None,
                windows: IndexMap::new(),
            }),
        })
    }

    pub fn transfer_stream(&self) -> Result<usize> {
        self.resources()?.transfer.handle()
    }

    fn resources(&self) -> Result<&Resources<T>> {
        self.resources
            .as_ref()
            .ok_or(Error::State("computation collective is closed"))
    }

    fn resources_mut(&mut self) -> Result<&mut Resources<T>> {
        self.resources
            .as_mut()
            .ok_or(Error::State("computation collective is closed"))
    }

    /// The communicator and the stream an operation is submitted on.
    fn submission(&mut self, asynchronous: bool) -> Result<(&mut Communicator, &Stream)> {
        let resources = self.resources_mut()?;
        let stream = if asynchronous {
            resources.transfer.stream()?
        } else {
            // A projection can invoke another collective before consuming its
            // gathered tail. Keep both streams on the same communicator order.
            if let Some(pending) = resources.pending.take() {
                resources.computation.wait_event(&pending)?;
            }
            &resources.computation
        };
        Ok((&mut resources.communicator, stream))
    }

    fn start(&mut self, call: impl FnOnce(&mut Self) -> Result<()>) -> Result<Arc<Event>> {
        let resources = self.resources()?;
        resources.transfer.wait(&resources.computation)?;
        let completed = Arc::new(Event::new(EventFlags::default()));
        let result = call(self);

        let resources = self.resources_mut()?;
        completed.record(resources.transfer.stream()?)?;
        if let Err(error) = result {
            // Join partial submissions before the caller can retire inputs.
            resources.computation.wait_event(&completed)?;
            return Err(error);
        }
        resources.pending = Some(Arc::clone(&completed));
        Ok(completed)
    }

    /// Orders `consumer` after an asynchronous operation's completion.
    pub fn join(&mut self, event: &Arc<Event>, consumer: usize) -> Result<()> {
        // SAFETY: the caller joins on a live stream of this device.
        unsafe { Stream::borrowed(self.device, consumer) }?.wait_event(event)?;

        let resources = self.resources_mut()?;
        if consumer == resources.computation.raw()
            && resources
                .pending
                .as_ref()
                .is_some_and(|pending| Arc::ptr_eq(pending, event))
        {
            // A completed eager join must not leak an event dependency into
            // a subsequent, independent graph capture.
            resources.pending = None;
        }
        Ok(())
    }

    /// Gathers `count` elements from every rank into `output`, rank-major.
    /// Asynchronous gathers return their completion event.
    ///
    /// # Safety
    /// Input and output spans must hold the declared elements on this device
    /// and remain alive through the submitted operation and graph replays.
    pub unsafe fn all_gather(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>> {
        let submit = |owner: &mut Self| {
            let (communicator, stream) = owner.submission(asynchronous)?;
            // SAFETY: forwarded from the caller.
            unsafe { communicator.all_gather(input, output, count, dtype, stream) }?;
            Ok(())
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// Exchanges `count`-element blocks with every rank. Asynchronous
    /// exchanges return their completion event.
    ///
    /// # Safety
    /// Each span holds `count` elements per rank and outlives all device uses.
    pub unsafe fn all_to_all(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
        asynchronous: bool,
    ) -> Result<Option<Arc<Event>>> {
        let submit = |owner: &mut Self| {
            let (communicator, stream) = owner.submission(asynchronous)?;
            // SAFETY: forwarded from the caller.
            unsafe { communicator.all_to_all(input, output, count, dtype, stream) }?;
            Ok(())
        };

        if asynchronous {
            self.start(submit).map(Some)
        } else {
            submit(self).map(|()| None)
        }
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn all_reduce(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        op: ReduceOp,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.all_reduce(data, data, count, dtype, op, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The in-place tensor holds `count` elements and outlives device uses.
    pub unsafe fn broadcast(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        root: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.broadcast(data, data, count, dtype, root, stream) }?;
        Ok(())
    }

    /// Sums `count` elements per rank and leaves this rank's block in `output`.
    ///
    /// # Safety
    /// Input holds `count` elements per rank and output holds `count` elements.
    /// Both remain allocated through device reads and writes.
    pub unsafe fn reduce_scatter(
        &mut self,
        output: u64,
        input: u64,
        count: usize,
        dtype: DataType,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.reduce_scatter(input, output, count, dtype, ReduceOp::Sum, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The span holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn send(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.send(data, count, dtype, peer, stream) }?;
        Ok(())
    }

    /// # Safety
    /// The destination holds `count` elements until its transfer and graph uses finish.
    pub unsafe fn recv(
        &mut self,
        data: u64,
        count: usize,
        dtype: DataType,
        peer: i32,
    ) -> Result<()> {
        let (communicator, stream) = self.submission(false)?;
        // SAFETY: forwarded from the caller.
        unsafe { communicator.recv(data, count, dtype, peer, stream) }?;
        Ok(())
    }

    /// Group matching point-to-point operations into one NCCL submission.
    /// The group is ended even when `call` fails.
    pub fn grouped<R>(&mut self, call: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        let mut outcome = None;
        nccl::group(|| {
            outcome = Some(call(self));
            Ok(())
        })?;
        // `group` runs the submission whenever the group starts.
        outcome.unwrap_or_else(|| {
            Err(Error::Invariant(
                "grouped collective submission did not run".into(),
            ))
        })
    }

    /// Retain one symmetric allocation through all contexts sharing this stream.
    ///
    /// # Safety
    /// `backing` keeps the VMM allocation `[data, data + bytes)` live; every rank
    /// registers matching windows in the same order, outside graph capture.
    pub unsafe fn register(&mut self, data: u64, bytes: usize, backing: T) -> Result<()> {
        let resources = self.resources_mut()?;
        if resources.windows.contains_key(&(data, bytes)) {
            return Ok(());
        }

        // SAFETY: forwarded from the caller; `backing` is retained below until
        // the communicator is destroyed or its resources are leaked.
        unsafe { resources.communicator.register_window(data, bytes) }?;
        resources.windows.insert((data, bytes), backing);
        Ok(())
    }

    /// Close collectively after all borrowed graph and tensor accesses retire.
    /// Aborted execution retains handles, mappings and streams until process exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let Some(resources) = self.resources.take() else {
            return Ok(());
        };
        let Resources {
            communicator,
            computation: _,
            mut transfer,
            pending: _,
            windows,
        } = resources;

        // Without a completed destroy, the communicator, its window backing,
        // its transfer stream and its context reference stay alive until
        // process exit.
        let retain = |transfer: CUDAStream, windows: IndexMap<(u64, usize), T>| {
            std::mem::forget(transfer);
            std::mem::forget(windows);
        };

        if aborted {
            std::mem::forget(communicator);
            retain(transfer, windows);
            return Ok(());
        }

        if let Err(error) = transfer.synchronize() {
            std::mem::forget(communicator);
            retain(transfer, windows);
            return Err(error);
        }
        // Deregisters every window, then destroys the communicator. A failure
        // leaks the communicator's remaining resources.
        if let Err(error) = communicator.destroy() {
            retain(transfer, windows);
            return Err(error.into());
        }
        if let Err(error) = transfer.close(false) {
            retain(transfer, windows);
            return Err(error);
        }
        Ok(())
    }

    /// Visit backing references held by a foreign-language resource owner.
    pub fn backings(&self) -> impl Iterator<Item = &T> {
        self.resources
            .iter()
            .flat_map(|resources| resources.windows.values())
    }
}

impl<T> Drop for StreamCommunicator<T> {
    fn drop(&mut self) {
        // Implicit destruction cannot enter a collective that peers may never
        // reach. Normal callers explicitly close after retiring graph readers.
        let _ = self.close(true);
    }
}
