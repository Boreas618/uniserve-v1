"""Graph capture, replay and release shared by every binding."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from uniserve.cuda_graph import break_graph, capture, pool, set_graph_trace

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
            actual = [graph.replay().clone()]
            set_graph_trace(True)
            actual.append(graph.replay().clone())
            set_graph_trace(False)
            actual.append(graph.replay().clone())
            torch.cuda.manual_seed(1234)
            generator.manual_seed(99)
            expected = [draw() for _ in range(3)]
        stream.synchronize()
        assert not torch.equal(actual[0], actual[1])
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    finally:
        set_graph_trace(False)
        graph.release()


@torch.inference_mode()
def test_a_graph_broken_once_replays_its_work_between_segments():
    """The work recorded at a break runs between the two segments.

    The work doubles the value the first segment wrote on the captured
    stream, so a replay computes ``(x + 1) * 2 + 3`` only when the work runs
    after the first segment and before the second.
    """
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    value = torch.ones(16, device=device)

    def step():
        value.add_(1)
        break_graph(lambda: value.mul_(2))
        value.add_(3)

    graph = capture(step, stream=stream)
    try:
        assert graph.segments == 2
        with torch.cuda.stream(stream):
            value.fill_(5)
            graph.replay()
        stream.synchronize()
        assert torch.equal(value, torch.full_like(value, (5 + 1) * 2 + 3))
    finally:
        stream.synchronize()
        graph.release()
