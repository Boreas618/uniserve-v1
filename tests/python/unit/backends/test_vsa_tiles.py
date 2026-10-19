"""VSA providers serve the tile sizes their kernels address.

128-row tiles have kernels on SM100 (data-center Blackwell), SM90 (Hopper)
and SM120 devices only, so resolving them elsewhere names that requirement
as a setup error, which is what a worker reports when it refuses such a
model at startup. Automatic resolution takes the first provider of its probe
order that serves the device, so an SM100 device keeps the SM100 kernels.
"""

import pytest
import torch
from uniserve_kernels.attention import vsa_native

from uniserve.backends.attention import vsa
from uniserve.errors import SetupError

pytestmark = pytest.mark.unit


def test_128_row_tiles_resolve_only_where_a_kernel_serves_them():
    with pytest.raises(SetupError, match="SM100.*SM90.*SM120"):
        vsa.resolve("auto", device=torch.device("cpu"), tile=128)
    with pytest.raises(SetupError, match="does not serve 128-row tiles"):
        vsa.resolve("triton", device=torch.device("cpu"), tile=128)
    if torch.cuda.is_available() and vsa_native.supported():
        backend = vsa.resolve("auto", device=torch.device("cuda"), tile=128)
        assert isinstance(backend, vsa.Backend)
        assert backend.name == "sm100"
