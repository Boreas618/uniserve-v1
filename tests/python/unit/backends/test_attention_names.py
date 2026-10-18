"""The attention provider names ``uniserve serve`` forwards to the library."""

import pytest
import torch

from uniserve.backends.attention import PROVIDERS, resolve

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", PROVIDERS)
def test_a_provider_name_resolves_to_that_provider(name):
    # Resolution constructs the provider's factory; its kernels load when a
    # layer is prepared.
    assert resolve(name, device=torch.device("cpu")).name == name


def test_auto_serves_every_call_portably_off_cuda():
    assert resolve("auto", device=torch.device("cpu")).name == "auto:cpu:torch"


def test_a_name_without_a_provider_is_refused_with_the_accepted_names():
    accepted = ", ".join(("auto", *PROVIDERS))
    with pytest.raises(ValueError, match=f"accepted names are {accepted}$"):
        resolve("sdpa", device=torch.device("cpu"))
