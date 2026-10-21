"""Graph capture, replay, export and release shared by every binding."""

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from uniserve.cuda_graph import (
    break_graph,
    capture,
    capture_begin,
    capture_end,
    export,
    pool,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _linear(device):
    torch.manual_seed(0)
    module = torch.nn.Linear(256, 256, bias=False)
    return module.to(device=device, dtype=torch.bfloat16)


@torch.inference_mode()
def test_capture_after_warmup_on_another_thread():
    """Numerical warmup does not require capture on the same host thread."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    module = _linear(device)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    with torch.cuda.stream(stream):
        expected = module(x).clone()
    stream.synchronize()

    with ThreadPoolExecutor(max_workers=1) as thread:
        graph = thread.submit(
            capture, lambda: module(x), stream=stream
        ).result()
    try:
        with torch.cuda.stream(stream):
            actual = graph.replay().clone()
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        graph.release()


@torch.inference_mode()
def test_releasing_a_graph_keeps_its_pool_siblings_replayable():
    """Graphs sharing a Rust-assigned pool survive each other's release."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    module = _linear(device)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    shared, *_ = pool(0)
    with torch.cuda.stream(stream):
        expected = module(x).clone()
    graphs = [
        capture(lambda: module(x), stream=stream, pool=shared) for _ in range(2)
    ]
    try:
        graphs[0].release()
        with torch.cuda.stream(stream):
            actual = graphs[1].replay().clone()
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.release()


def test_pool_reports_its_allocator_identity():
    handle, first, second, total = pool(0)
    assert (first, second) == tuple(handle.id)
    assert total == torch.cuda.get_device_properties(0).total_memory


@torch.inference_mode()
def test_failed_capture_can_be_retried():
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    module = _linear(device)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    shared, *_ = pool(0)
    failure = ValueError("numerical call failed")

    def rejected():
        module(x)
        raise failure

    with pytest.raises(RuntimeError) as raised:
        capture(rejected, stream=stream, pool=shared)
    assert raised.value.__cause__ is failure

    with torch.cuda.stream(stream):
        expected = module(x).clone()
    graph = capture(lambda: module(x), stream=stream, pool=shared)
    try:
        with torch.cuda.stream(stream):
            actual = graph.replay().clone()
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        graph.release()


@torch.inference_mode()
def test_replay_advances_registered_generators():
    """Successive replays match eager draws from the same seeds."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device)

    def draw():
        return torch.cat(
            (
                torch.rand(128, device=device),
                torch.rand(128, device=device, generator=generator),
            )
        )

    with torch.cuda.stream(stream):
        draw()
    graph = capture(draw, stream=stream, generators=(generator,))
    try:
        torch.cuda.manual_seed(1234)
        generator.manual_seed(99)
        with torch.cuda.stream(stream):
            actual = [graph.replay().clone() for _ in range(2)]
            torch.cuda.manual_seed(1234)
            generator.manual_seed(99)
            expected = [draw() for _ in range(2)]
        stream.synchronize()
        assert not torch.equal(actual[0], actual[1])
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    finally:
        graph.release()


class _Executable:
    """A graph executable instantiated through the driver.

    The worker instantiates the captured graphs a capture exports this way.
    """

    def __init__(self, graph: int) -> None:
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._driver.cuGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        self._driver.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._driver.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self._raw = ctypes.c_void_p()
        status = self._driver.cuGraphInstantiateWithFlags(
            ctypes.byref(self._raw), graph, 0
        )
        assert status == 0

    def launch(self, stream: torch.cuda.Stream) -> None:
        assert self._driver.cuGraphLaunch(self._raw, stream.cuda_stream) == 0

    def destroy(self) -> None:
        assert self._driver.cuGraphExecDestroy(self._raw) == 0


@torch.inference_mode()
def test_exported_graphs_launch_to_the_replayed_output():
    """An exported segment, instantiated outside torch, equals a replay.

    The instantiated graph stays valid once torch releases its captured
    graph.
    """
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    module = _linear(device)
    x = torch.empty(64, 256, device=device, dtype=torch.bfloat16)
    with torch.cuda.stream(stream):
        module(x)
    graph = capture(lambda: module(x), stream=stream)
    executable = None
    try:
        exported, graphs = export(graph)
        assert exported is graph
        assert len(graphs) == 1
        executable = _Executable(graphs[0])

        outputs = []
        for run in (graph.replay, lambda: executable.launch(stream)):
            # The same input before each run, and a cleared output, so the
            # output read afterwards was written by that run.
            with torch.cuda.stream(stream):
                x.copy_(torch.linspace(-1, 1, x.numel()).view_as(x))
                graph.output.zero_()
                run()
                outputs.append(graph.output.clone())
            stream.synchronize()
            # Once the first output is read, torch releases its captured
            # graph; the instantiated one runs on.
            graph.handover()
        assert outputs[0].abs().sum() > 0
        assert torch.equal(outputs[0], outputs[1])
    finally:
        stream.synchronize()
        if executable is not None:
            executable.destroy()
        graph.release()


@torch.inference_mode()
def test_export_names_one_graph_per_segment():
    """A graph broken once exports two segments, which only Rust launches.

    The work between them is a break op the worker notes and performs, so
    torch replays no graph with breaks.
    """
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    value = torch.ones(16, device=device)

    recording = capture_begin(None, stream.cuda_stream, device.index)
    try:
        with torch.cuda.stream(stream):
            value.add_(1)
            break_graph()
            value.add_(3)
    except BaseException as error:
        recording.end(failure=error)
        raise
    graph, graphs = capture_end(recording, instantiate=True)
    try:
        assert len(graphs) == 2 and len(set(graphs)) == 2
        with pytest.raises(RuntimeError, match="break ops"):
            graph.replay()
    finally:
        graph.release()
