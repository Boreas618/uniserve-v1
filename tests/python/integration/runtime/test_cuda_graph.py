"""Graph capture, replay, export and release shared by every binding."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import torch

from uniserve_models.interface.graphs import capture, export, pool

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


@torch.inference_mode()
def test_segmented_capture_replays_eager_submissions_in_order():
    """Work a graph cannot record runs between segments at replay."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    value = torch.ones(16, device=device)
    submissions = []

    @contextmanager
    def segments(submit):
        submissions.append(submit)
        yield

    def body():
        value.add_(1)
        # The eager call doubles the value between the two segments.
        submissions[-1](lambda: value.mul_(2))
        value.add_(3)
        return value

    with torch.cuda.stream(stream):
        value.fill_(1)
    graph = capture(body, stream=stream, segments=segments)
    try:
        with torch.cuda.stream(stream):
            value.fill_(1)
            graph.replay()
        stream.synchronize()
        # (1 + 1) * 2 + 3: the capture itself ran only the graph segments'
        # recording, so the replay alone produced this value.
        assert torch.equal(value, torch.full_like(value, 7))
    finally:
        graph.release()


def _launch(executable: int, stream: torch.cuda.Stream) -> None:
    """Launch a raw graph executable on ``stream`` through the driver."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    driver.cuGraphLaunch.restype = ctypes.c_int
    assert driver.cuGraphLaunch(executable, stream.cuda_stream) == 0


@torch.inference_mode()
def test_exported_executables_launch_to_the_replayed_output():
    """Launching an exported graph's executable equals replaying it."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    module = _linear(device)
    x = torch.empty(64, 256, device=device, dtype=torch.bfloat16)
    with torch.cuda.stream(stream):
        module(x)
    graph = capture(lambda: module(x), stream=stream)
    try:
        exported, executables, breaks = export(graph)
        assert exported is graph
        assert len(executables) == 1 and breaks == 0

        outputs = []
        for run in (graph.replay, lambda: _launch(executables[0], stream)):
            # The same input before each run, and a cleared output, so the
            # output read afterwards was written by that run.
            with torch.cuda.stream(stream):
                x.copy_(torch.linspace(-1, 1, x.numel()).view_as(x))
                graph.output.zero_()
                run()
                outputs.append(graph.output.clone())
            stream.synchronize()
        assert outputs[0].abs().sum() > 0
        assert torch.equal(outputs[0], outputs[1])
    finally:
        graph.release()


@torch.inference_mode()
def test_export_counts_the_breaks_between_segments():
    """A segmented graph exports one executable per segment."""
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    value = torch.ones(16, device=device)
    submissions = []

    @contextmanager
    def segments(submit):
        submissions.append(submit)
        yield

    def body():
        value.add_(1)
        submissions[-1](lambda: value.mul_(2))
        value.add_(3)

    owner = object()
    graph = capture(body, stream=stream, segments=segments)
    try:
        exported, executables, breaks = export(graph, owner)
        assert exported is owner
        assert len(executables) == 2 and len(set(executables)) == 2
        assert breaks == 1
    finally:
        graph.release()
