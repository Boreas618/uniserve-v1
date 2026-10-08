"""Graph capture, replay and release shared by every binding."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import torch

from uniserve_models.interface.graphs import capture, pool

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
