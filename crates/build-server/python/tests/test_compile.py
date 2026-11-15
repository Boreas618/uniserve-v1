"""The compile request: how the Build Server compiles one descriptor.

The Build Server runs ``python -m uniserve_build_server compile`` in the
build module's directory with the request as JSON on standard input
(``Request`` in ``crates/build-server/src/source.rs``). These tests run the
program exactly so, with a CUDA-kind package of their own (``recording``)
whose build module records what it was handed, and a compiler of their own
(``recording/nvcc``) that records how the CUDA driver ran it, so the protocol
and the declared compilation are checked without a CUDA toolkit or a device.
"""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import types

import pytest
import triton
import triton.language as tl
from uniserve_build_server import triton as driver
from uniserve_build_server.request import Request, Target
from uniserve_build_server.toolchain import available, missing

#: The recording package: its build module writes what it was handed.
RECORDING = pathlib.Path(__file__).parent / "recording"

#: The compiler that records its arguments beside what it writes.
NVCC = RECORDING / "nvcc"


def request(package: pathlib.Path, output: pathlib.Path, **changes) -> dict:
    """A request as the Build Server writes one for the recording module."""
    target = Target.named("sm_100a")
    written = {
        "op": "recording",
        "backend": "cuda",
        "descriptor": "dtype_bf16_columns_128",
        "fields": {"dtype": "bf16", "columns": 128, "pdl": False},
        "parameters": {"tile": 64, "selected": {"mActType": "silu"}},
        "source": "cuda",
        "module": "build",
        "launch": None,
        "compile": None,
        "entries": {"launch": "uniserve_recording"},
        "unit": None,
        "package": str(package),
        "output": str(output),
        "target": {
            "name": target.name,
            "backend": target.backend,
            "capability": target.capability,
            "specific": target.specific,
            "families": list(target.families),
            "warp_size": target.warp_size,
            "shared_memory": target.shared_memory,
            "tensor_memory": target.tensor_memory,
            "warpgroup_mma": target.warpgroup_mma,
            "dependent_launch": target.dependent_launch,
            "fp8": target.fp8,
        },
        "jobs": 3,
        "cache": str(output / "cache"),
        "toolchain": {
            "interpreter": sys.executable,
            "driver": "ab" * 32,
            "kernels": available("cd" * 32),
            "distributions": {},
            "triton": available("3.7.1"),
            "cute_dsl": available("4.6.2"),
            "cuda": available(
                {
                    "version": "13.0.48",
                    "root": "/cuda",
                    "nvcc": "/cuda/bin/nvcc",
                    "include": "/cuda/include",
                    "lib": "/cuda/lib",
                    "stubs": "/cuda/lib/stubs",
                }
            ),
            "cudnn": available(
                {
                    "version": "9.12.0",
                    "include": "/cudnn/include",
                    "lib": "/cudnn/lib",
                }
            ),
            "host_compiler": missing("no c++ on this host's search path"),
        },
    }
    written.update(changes)
    return written


def compile_in(module_directory: pathlib.Path, stated: dict):
    """Run the compilation as the Build Server runs it."""
    return subprocess.run(
        [sys.executable, "-m", "uniserve_build_server", "compile"],
        cwd=module_directory,
        input=json.dumps(stated),
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_build_module_is_handed_the_typed_request(tmp_path):
    output = tmp_path / "output"
    compiled = compile_in(RECORDING / "src", request(RECORDING, output))
    assert compiled.returncode == 0, compiled.stderr

    handed = json.loads((output / "handed.json").read_text())
    assert handed == {
        "descriptor": "dtype_bf16_columns_128",
        "fields": {"dtype": "bf16", "columns": 128, "pdl": False},
        "parameters": {"tile": 64, "selected": {"mActType": "silu"}},
        "entries": {"launch": "uniserve_recording"},
        "target": "sm_100a",
        "capability": 100,
        "families": ["sm_100a", "sm_100f"],
        "jobs": 3,
        "nvcc": "/cuda/bin/nvcc",
    }


def test_a_missing_toolchain_refuses_the_build_with_its_reason(tmp_path):
    stated = request(RECORDING, tmp_path / "output")
    stated["toolchain"]["cuda"] = missing("no CUDA 13.0 toolkit")
    compiled = compile_in(RECORDING / "src", stated)
    assert compiled.returncode != 0
    assert "no CUDA 13.0 toolkit" in compiled.stderr


def test_a_source_without_a_build_module_is_refused(tmp_path):
    compiled = compile_in(
        RECORDING / "src",
        request(RECORDING, tmp_path / "output", source="cubin"),
    )
    assert compiled.returncode != 0
    assert "cubin" in compiled.stderr


def declared(output: pathlib.Path, compile: dict, **fields) -> dict:
    """A request for a declared CUDA compilation, compiled by the recording
    compiler.
    """  # noqa: D205
    stated = request(
        RECORDING,
        output,
        module=None,
        compile=compile,
        fields=fields,
        descriptor="_".join(
            f"{name}_{value}" for name, value in fields.items()
        ),
    )
    stated["toolchain"]["cuda"]["available"]["nvcc"] = str(NVCC)
    return stated


def recorded(output: pathlib.Path) -> list[list[str]]:
    """Each run of the recording compiler, as its arguments."""
    lines = (output / "nvcc.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_a_declared_native_compilation_binds_and_links(tmp_path):
    """Its definitions are bound for the descriptor and its libraries linked
    where the toolchain description states them.
    """  # noqa: D205
    output = tmp_path / "output"
    compile = {
        "units": ["src/entry.cpp", "src/dense.cu"],
        "standard": "c++17",
        "definitions": {
            "UNISERVE_CAUSAL": {
                "field": "mask",
                "values": {"causal": 1, "dense": 0},
            },
            "UNISERVE_SEQUENCES": {"field": "cu_seq_lens", "absent": True},
            "UNISERVE_TILE": {"parameter": "tile"},
            "UNISERVE_INPUT": {
                "field": "dtype",
                "values": {"bf16": "CUDA_R_16BF", "f32": "CUDA_R_32F"},
            },
        },
        "libraries": ["cublasLt", "cudnn"],
        "host_contraction": False,
        "kernel": None,
    }
    stated = declared(output, compile, mask="causal", dtype="bf16")
    compiled = compile_in(RECORDING / "src", stated)
    assert compiled.returncode == 0, compiled.stderr

    # The units compile at once, so their runs are recorded in either order.
    *units, linked = recorded(output)
    assert sorted(arguments[-1] for arguments in units) == sorted(
        str(RECORDING / source) for source in compile["units"]
    )
    for arguments in units:
        assert "-std=c++17" in arguments and "-arch=sm_100a" in arguments
        # Each definition is bound for this descriptor: the value the field
        # selects, the absent field's value, the parameter.
        for bound in (
            "-DUNISERVE_CAUSAL=1",
            "-DUNISERVE_SEQUENCES=1",
            "-DUNISERVE_TILE=64",
            "-DUNISERVE_INPUT=CUDA_R_16BF",
        ):
            assert bound in arguments, arguments
        # The package's own headers, a library's headers, and its own option.
        assert f"-I{RECORDING / 'include'}" in arguments
        assert "-I/cudnn/include" in arguments
        assert "-ffp-contract=off" in arguments
    # The libraries where the description states them, and the runtime.
    for argument in ("-L/cuda/lib", "-L/cudnn/lib", "-lcublasLt", "-lcudnn"):
        assert argument in linked, linked
    assert "-lcudart" in linked
    assert linked[linked.index("-o") + 1] == str(output / "module.so")
    # The objects lived in a scratch directory the module does not keep.
    assert sorted(path.name for path in output.iterdir()) == [
        "module.so",
        "nvcc.jsonl",
    ]


def test_a_declared_cubin_writes_its_kernel_launch_metadata(tmp_path):
    output = tmp_path / "output"
    compile = {
        "units": ["src/topk_softmax.cu"],
        "standard": "c++20",
        "definitions": {
            "UNISERVE_EXPERTS": {"field": "experts"},
            "UNISERVE_WARPS": {"parameter": "warps"},
        },
        "libraries": [],
        "host_contraction": True,
        "kernel": {
            "name": "uniserve_topk_softmax",
            "warps": {"parameter": "warps"},
            "shared": 0,
            "signature": [
                ["scores", {"pointer": "scores"}],
                ["rows", "i32"],
                [
                    "scale",
                    {
                        "field": "scale",
                        "values": {"none": "*fp32", "bf16": "*bf16"},
                    },
                ],
            ],
        },
    }
    stated = declared(output, compile, scores="bf16", experts=128, scale="none")
    stated["parameters"] = {"warps": 4}
    compiled = compile_in(RECORDING / "src", stated)
    assert compiled.returncode == 0, compiled.stderr

    (arguments,) = recorded(output)
    assert "-cubin" in arguments
    assert arguments[arguments.index("-o") + 1] == str(output / "module.cubin")
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata == {
        "name": "uniserve_topk_softmax",
        "num_warps": 4,
        "num_ctas": 1,
        "shared": 0,
        "launch_pdl": False,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
        # In the kernel's parameter order, spelled as `ops` reads a signature.
        "signature": {"scores": "*bf16", "rows": "i32", "scale": "*fp32"},
        # The module's compile constants.
        "definitions": {"UNISERVE_EXPERTS": 128, "UNISERVE_WARPS": 4},
    }
    assert list(metadata["signature"]) == ["scores", "rows", "scale"]


def test_a_failed_compile_refuses_the_build_naming_the_unit(tmp_path):
    output = tmp_path / "output"
    compile = {
        "units": ["src/dense.cu"],
        "standard": "c++17",
        "definitions": {},
        "libraries": [],
        "host_contraction": True,
        "kernel": None,
    }
    stated = declared(output, compile, mask="dense")
    stated["toolchain"]["cuda"]["available"]["nvcc"] = "/bin/false"
    compiled = compile_in(RECORDING / "src", stated)
    assert compiled.returncode != 0
    assert f"compiling {RECORDING / 'src/dense.cu'} failed" in compiled.stderr


def test_a_target_the_description_does_not_state_is_refused():
    with pytest.raises(SystemExit, match="no target 'sm_75'"):
        Target.named("sm_75")


def test_tuning_data_configures_the_first_table_the_descriptor_fits(tmp_path):
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning" / "forward.toml").write_text(
        "[[configuration]]\n"
        'descriptors = { dtype = "bf16", columns = 64 }\n'
        "stages = 1\n"
        "[[configuration]]\n"
        'descriptors = { dtype = "bf16" }\n'
        "stages = 2\n"
        "tile = [128, 64]\n"
        "[[configuration]]\n"
        "stages = 3\n"
    )
    (tmp_path / "tuning" / "narrow.toml").write_text(
        '[[configuration]]\ndescriptors = { dtype = "f16" }\nstages = 1\n'
    )

    def handed(**fields):
        stated = request(tmp_path, tmp_path / "output", fields=fields)
        return Request.read(io.StringIO(json.dumps(stated)))

    # The first table whose fields the descriptor's all equal, with its
    # `descriptors` left out; a table naming no fields fits every descriptor.
    forward = "tuning/forward.toml"
    assert handed(dtype="bf16", columns=128).tuned(forward) == {
        "stages": 2,
        "tile": [128, 64],
    }
    assert handed(dtype="bf16", columns=64).tuned(forward) == {"stages": 1}
    assert handed(dtype="f32", columns=64).tuned(forward) == {"stages": 3}
    with pytest.raises(SystemExit, match="no configuration of"):
        handed(dtype="bf16", columns=64).tuned("tuning/narrow.toml")


@triton.jit
def kernel(x, columns, scale, BLOCK: tl.constexpr, CENTRED: tl.constexpr, PDL: tl.constexpr):  # noqa: N803
    """A kernel whose launch table the tests state; it is never compiled."""
    tl.store(x + tl.arange(0, BLOCK), scale)


#: The source module a launch table names its kernel in.
KERNELS = types.SimpleNamespace(kernel=kernel)

#: A launch table of :data:`KERNELS`, which states a compilation without a GPU
#: or a compiler being asked for one.
LAUNCH = {
    "kernel": "kernel",
    "warps": {"field": "columns", "steps": [[256, 8], [128, 4]], "otherwise": 2},
    "aligned": ["x", "columns"],
    "types": {
        "x": {"pointer": "dtype"},
        "columns": "i32",
        "scale": {"field": "wide", "values": {"true": "fp64", "false": "fp32"}},
    },
    "constants": {
        "BLOCK": {"field": "columns"},
        "CENTRED": {"field": "centred", "absent": False},
    },
}


def test_a_launch_table_states_a_triton_compilation():
    stated = driver.stated(
        LAUNCH,
        KERNELS,
        "dtype_bf16_columns_128_wide_false_pdl_true",
        {"dtype": "bf16", "columns": 128, "wide": False, "pdl": True},
        {},
        Target.named("sm_100a"),
    )
    # The signature is the kernel's parameter order, with the element type
    # spelled as Triton spells it and the type the field selects.
    assert stated["signature"] == {
        "x": "*bf16",
        "columns": "i32",
        "scale": "fp32",
        "BLOCK": "constexpr",
        "CENTRED": "constexpr",
        "PDL": "constexpr",
    }
    assert stated["constexprs"] == {"BLOCK": 128, "CENTRED": False, "PDL": True}
    # The target has dependent launch, so the module launches dependently.
    assert stated["options"] == {"num_warps": 4, "launch_pdl": True}
    # The columns the key fixes are a multiple of sixteen, so both are aligned.
    assert stated["attributes"] == {
        (0,): [["tt.divisibility", 16]],
        (1,): [["tt.divisibility", 16]],
    }


def test_a_launch_table_aligns_a_fixed_width_only_when_it_is_whole_vectors():
    stated = driver.stated(
        LAUNCH,
        KERNELS,
        "dtype_f32_columns_100_wide_true_pdl_false",
        {"dtype": "f32", "columns": 100, "wide": True, "pdl": False},
        {},
        Target.named("sm_100a"),
    )
    assert stated["attributes"] == {(0,): [["tt.divisibility", 16]]}
    assert stated["signature"]["scale"] == "fp64"
    assert stated["options"] == {"num_warps": 2, "launch_pdl": False}


def test_a_launch_table_that_leaves_a_parameter_unbound_is_refused():
    unbound = {**LAUNCH, "constants": {"BLOCK": {"field": "columns"}}}
    with pytest.raises(SystemExit, match="CENTRED"):
        driver.stated(
            unbound,
            KERNELS,
            "dtype_bf16_columns_128_wide_false",
            {"dtype": "bf16", "columns": 128, "wide": False},
            {},
            Target.named("sm_100a"),
        )
