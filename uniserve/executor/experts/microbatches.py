"""Cooperative host turns for numerical microbatches.

An expert step may split a rank's forward into microbatches, each with its
own CUDA stream and expert exchange. Every rank of the union must submit
the microbatches' collectives in one order, while their device work
overlaps on the distinct streams. A :class:`Rotation` passes one host turn
between the microbatches' calls: calls start in index order, hand the turn
to the next active index whenever they yield after an expert dispatch
(:meth:`Rotation.yield_turn`) and leave the rotation when they return.

:class:`MicrobatchThreads` runs each microbatch's calls on its own
persistent thread, so a call that yields keeps its Python frames suspended
on that thread until the turn returns to it, and thread-local numerical
state (the current stream, inference mode, the cuBLAS handle) exists
before graph capture. Waiting for a turn blocks on a lock, which releases
the interpreter lock: only the turn holder runs Python.
"""

from __future__ import annotations

import functools
import queue
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, TypeVar

__all__ = ["MicrobatchAborted", "MicrobatchThreads", "Rotation"]

T = TypeVar("T")


class MicrobatchAborted(RuntimeError):  # noqa: N818 - names the outcome
    """A microbatch call stopped because its invocation was aborted.

    Raised in place of a call's turn after another call of the invocation
    failed; the failing call's error remains the invocation's error.
    """


class Rotation:
    """The host turn shared by one invocation's microbatch calls.

    Each index runs on its own thread. An invocation admits every index
    (:meth:`begin`), runs one call per index (:meth:`run`), and ends after
    every call returned (:meth:`end`). A rotation is reusable after a failed
    invocation.

    Args:
        count: The number of microbatches.

    Raises:
        ValueError: ``count`` is not positive.
    """

    def __init__(self, count: int) -> None:
        if count <= 0:
            raise ValueError("microbatches require at least one context")

        # Every field below is read and written under `_ready`'s lock.
        self._ready = threading.Condition()
        self._active = [False] * count
        self._turn = 0
        self._running = False
        self._closed = False
        self._aborted = False
        self._failure: int | None = None

        # The index whose call the current thread runs in this rotation.
        self._local = threading.local()

    @property
    def failure(self) -> int | None:
        """The index whose call failed first in this invocation, if any."""
        with self._ready:
            return self._failure

    def begin(self) -> None:
        """Admit one call per microbatch; the first index holds the turn.

        Raises:
            RuntimeError: An invocation is running, or the rotation closed.
        """
        with self._ready:
            if self._running:
                raise RuntimeError("microbatch invocation is already running")
            if self._closed:
                raise RuntimeError("microbatch execution is closed")

            self._active = [True] * len(self._active)
            self._turn = 0
            self._running = True
            self._aborted = False
            self._failure = None

    def run(self, index: int, call: Callable[[], T]) -> T:
        """Run the admitted ``index``'s call once in this invocation.

        Waits for the index's turn, then runs ``call`` on this thread. On
        return or failure the index leaves the rotation and passes the turn
        on; the first failing call aborts the invocation, which wakes every
        suspended peer. The caller starts every index before waiting for
        any of them.

        Raises:
            MicrobatchAborted: The invocation was aborted before ``call``
                ran or while it waited for a turn.
            Exception: Whatever ``call`` raised.
        """
        previous = getattr(self._local, "index", None)
        self._local.index = index
        failed = True
        try:
            with self._ready:
                self._wait(index)

            result = call()
            failed = False
            return result
        finally:
            self._local.index = previous
            with self._ready:
                # A call interrupted by an abort is not a failure of its own:
                # the call that aborted the invocation remains its cause.
                if failed and not self._aborted:
                    self._failure = index
                    self._aborted = True

                self._active[index] = False
                self._pass(index)
                self._ready.notify_all()

    def yield_turn(self) -> None:
        """Pass the host turn on and wait until it returns.

        Called after submitting an expert dispatch. Outside a call of this
        rotation it does nothing, so numerical layers make the same call in
        ordinary forwards.

        Raises:
            MicrobatchAborted: The invocation was aborted meanwhile.
        """
        index = getattr(self._local, "index", None)
        if index is None:
            return

        with self._ready:
            self._pass(index)
            self._ready.notify_all()
            self._wait(index)

    def end(self) -> None:
        """Complete an invocation after every call returned."""
        with self._ready:
            self._running = False
            self._active = [False] * len(self._active)
            self._failure = None

    def close(self) -> None:
        """Refuse further invocations.

        Raises:
            RuntimeError: An invocation is running.
        """
        with self._ready:
            if self._running:
                raise RuntimeError("cannot close running microbatches")

            self._closed = True

    def _wait(self, index: int) -> None:
        # Caller holds the lock; waiting releases it (and the GIL).
        self._ready.wait_for(lambda: self._turn == index or self._aborted)
        if self._aborted:
            raise MicrobatchAborted("microbatch execution aborted")

    def _pass(self, index: int) -> None:
        # The next active index after `index`, cyclically; `index` itself
        # when it is the only active one. Caller holds the lock.
        count = len(self._active)
        for offset in range(1, count + 1):
            following = (index + offset) % count
            if self._active[following]:
                self._turn = following
                return


# A job runs one call on a microbatch thread and reports its own outcome.
_Job = Callable[[], None]


def _serve(jobs: queue.SimpleQueue[_Job | None]) -> None:
    """A microbatch thread: run jobs in order until the stop marker."""
    while (job := jobs.get()) is not None:
        job()


class MicrobatchThreads:
    """Persistent microbatch threads and the rotation of their calls.

    Thread ``i`` is a daemon named ``microbatch-{i}`` and runs every call of
    microbatch ``i``, inside ``context(i)``: the integrator passes the
    microbatch's device, stream and inference-mode contexts there, which
    are thread-local. The threads persist across invocations so their
    thread-local numerical state exists before graph capture.

    Args:
        count: The number of microbatches.
        context: ``context(index)`` returns the context manager each call of
            microbatch ``index`` runs in; none by default.

    Raises:
        ValueError: ``count`` is not positive.
    """

    def __init__(
        self,
        count: int,
        context: Callable[[int], AbstractContextManager[Any]] | None = None,
    ) -> None:
        self._rotation = Rotation(count)
        self._context = context
        self._closed = False

        # Each thread holds only its job queue. An instance an aborting rank
        # abandons without closing leaves idle daemon threads, which never
        # block interpreter exit.
        self._jobs: list[queue.SimpleQueue[_Job | None]] = []
        self._threads: list[threading.Thread] = []
        for index in range(count):
            jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
            thread = threading.Thread(
                target=_serve,
                args=(jobs,),
                name=f"microbatch-{index}",
                daemon=True,
            )
            thread.start()
            self._jobs.append(jobs)
            self._threads.append(thread)

    @property
    def count(self) -> int:
        """The number of microbatches."""
        return len(self._threads)

    @property
    def rotation(self) -> Rotation:
        """The rotation of :meth:`run`; exchanges yield through it."""
        return self._rotation

    def each(self, call: Callable[[int], object]) -> None:
        """Run ``call(index)`` once on every thread, outside the rotation.

        For thread-local preparation, such as a microbatch thread's cuBLAS
        handle. Returns after every call finished.

        Raises:
            RuntimeError: The threads closed.
            Exception: The first failing call's error, in index order.
        """
        if self._closed:
            raise RuntimeError("microbatch execution is closed")

        calls = [functools.partial(call, index) for index in range(self.count)]
        errors, _ = self._submit(calls, None)
        for error in errors:
            if error is not None:
                raise error

    def run(self, calls: Sequence[Callable[[], T]]) -> list[T]:
        """Run one call per microbatch in rotation; results in index order.

        Calls start in index order and pass the host turn whenever they
        yield (:meth:`Rotation.yield_turn`). Returns once every call
        returned. On failure every suspended call is woken and every turn
        retires before the error is raised. Orders no device work: the
        caller orders the microbatch streams around the invocation.

        Raises:
            ValueError: ``calls`` does not hold one call per microbatch.
            RuntimeError: The threads closed, or an invocation is running.
            Exception: The first failing call's error; the calls it
                interrupted raised :class:`MicrobatchAborted`.
        """
        if len(calls) != self.count:
            raise ValueError("each microbatch needs one call")

        self._rotation.begin()
        try:
            errors, results = self._submit(calls, self._rotation)
            failure = self._rotation.failure
        finally:
            self._rotation.end()

        # The first failing call caused every abort of the others.
        first = None if failure is None else errors[failure]
        if first is not None:
            raise first
        for error in errors:
            if error is not None:
                raise error

        return [results[index] for index in range(len(calls))]

    def close(self) -> None:
        """Stop and join the threads after the last invocation.

        A rank that aborts leaves the threads instead: one may be blocked
        on a failed peer.

        Raises:
            RuntimeError: An invocation is running.
        """
        self._rotation.close()
        if self._closed:
            return

        self._closed = True
        for jobs in self._jobs:
            jobs.put(None)
        for thread in self._threads:
            thread.join()

    def _submit(
        self,
        calls: Sequence[Callable[[], T]],
        rotation: Rotation | None,
    ) -> tuple[list[BaseException | None], dict[int, T]]:
        """Run ``calls[i]`` on thread ``i`` and wait for every outcome.

        Inside ``rotation`` each call runs its turn. Waiting for the
        outcomes releases the interpreter lock.

        Returns:
            Each call's error (``None`` when it returned) and the results
            of the calls that returned, by index.
        """
        errors: list[BaseException | None] = [None] * len(calls)
        results: dict[int, T] = {}
        # A call's thread writes its outcome before reporting its index, so
        # the submitter reads every outcome after the last report.
        finished: queue.SimpleQueue[int] = queue.SimpleQueue()

        def job(index: int, call: Callable[[], T]) -> None:
            def contained() -> T:
                scope = (
                    nullcontext()
                    if self._context is None
                    else self._context(index)
                )
                with scope:
                    return call()

            # Every call reports once, failures included, so the wait below
            # always completes.
            try:
                if rotation is None:
                    results[index] = contained()
                else:
                    results[index] = rotation.run(index, contained)
            except BaseException as error:  # noqa: BLE001 - reported
                errors[index] = error
            finally:
                finished.put(index)

        for index, call in enumerate(calls):
            self._jobs[index].put(functools.partial(job, index, call))
        for _ in calls:
            finished.get()

        return errors, results
