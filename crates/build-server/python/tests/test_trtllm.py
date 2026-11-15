"""The trtllm-gen libraries' tables: how a package's kernels are selected.

A package that launches trtllm-gen kernels publishes the cubins its
descriptor can select, chosen from the library's generated table by the
options each kernel was built with. These tests state small tables in the
libraries' own formats (the FMHA library's positional
`TllmGenFmhaKernelMetaInfo` entries, the batched-GEMM library's
`BatchedGemmConfig` entries with labelled options) and check the selection
a package's statement makes of them, without a pinned library or a compiler.
"""

from __future__ import annotations

import pytest
from uniserve_build_server import trtllm
from uniserve_build_server.request import Target

#: FlashInfer's declarations the FMHA table's members take.
FMHA_ENUMERATIONS = """
enum Data_type {
  DATA_TYPE_FP16,
  DATA_TYPE_BF16,
  DATA_TYPE_FP32,
};
enum class TrtllmGenAttentionMaskType {
  // Dense mask.
  Dense = 0,
  Causal,
  SlidingOrChunkedCausal,
};
"""

#: An FMHA table: one prefill kernel per mask, a skipped-softmax variant and a
#: family kernel, in the library's format.
FMHA_TABLE = """
#pragma once
struct TllmGenFmhaKernelMetaInfo
{
    Data_type mDataTypeQ;
    int mHeadDimQk;
    int mSM;
    const char* mFuncName;
    int mMaskType;
    bool mSkipsSoftmaxWhenPossible;
    const char* sha256;
};

static const TllmGenFmhaKernelMetaInfo sTllmGenFmhaKernelMetaInfos[] = {
{ DATA_TYPE_BF16, 128, kSM_100, "fmhaSm100aKernel_Causal", 1, false, "aa"},
{ DATA_TYPE_BF16, 128, kSM_100, "fmhaSm100aKernel_Dense", 0, false, "bb"},
{ DATA_TYPE_BF16, 128, kSM_100, "fmhaSm100aKernel_CausalSkips", 1, true, "cc"},
{ DATA_TYPE_FP16, 128, kSM_100f, "fmhaSm100fKernel_Causal", 1, false, "dd"},
};
"""

#: The batched-GEMM library's declarations: its element types (through the
#: library's encoding macro), an enumeration of its options, and the table's
#: struct.
BMM_DECLARATIONS = """
namespace trtllm { namespace gen {
enum class Dtype : uint32_t {
#define TLLM_ENCODE_DTYPE(BlockFormatBit, SignedBit, IntegerBit,         \\
                          NumBits, Uid)                                  \\
  uint32_t {                                                                  \\
    (BlockFormatBit << 24) | (SignedBit << 20) | (IntegerBit << 16) |         \\
    (NumBits << 8) | (Uid)                                                    \\
  }
  Bfloat16 = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u,
                               /*bits*/ 16u, /*uid*/ 0u),
  E2m1     = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u,
                               /*bits*/  4u, /*uid*/ 2u),
#undef TLLM_ENCODE_DTYPE
};
}}
namespace gemm {
enum class BiasType : uint32_t {
  None = 0,
  M = 1,
};
}
struct BatchedGemmConfig {
  uint8_t const* mData{nullptr};
  uint32_t mSize{0};
  char const* mFunctionName{nullptr};
  char const* mHash{nullptr};
  BatchedGemmOptions mOptions;
  tg::CudaArch mSm{tg::CudaArch::Sm100a};
};
"""

#: A batched-GEMM table of three kernels, in the library's format.
BMM_TABLE = """
namespace batchedGemm {
static constexpr size_t tllmGenBatchedGemmListLen = 3;

static const batchedGemm::BatchedGemmConfig tllmGenBatchedGemmList[] = {
#ifndef EXCLUDE_SM_100
{nullptr, 0, "bmm_E2m1_t128x8", "11",
{ /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mBiasType */ gemm::BiasType(1)
, /* mTileN */ 8
, /* mBatchedM */ {}
 }, gemm::SmVersion::Sm100f},
{nullptr, 0, "bmm_E2m1_t128x16", "22",
{ /* mDtypeA */ trtllm::gen::Dtype(17826818)
, /* mBiasType */ gemm::BiasType(1)
, /* mTileN */ 16
, /* mBatchedM */ {}
 }, gemm::SmVersion::Sm100f},
{nullptr, 0, "bmm_Bfloat16_t128x8", "33",
{ /* mDtypeA */ trtllm::gen::Dtype(1052672)
, /* mBiasType */ gemm::BiasType(0)
, /* mTileN */ 8
, /* mBatchedM */ {}
 }, gemm::SmVersion::Sm100a},
#endif // EXCLUDE_SM_100
};

} // namespace batchedGemm
"""


def fmha_kernels():
    enums = trtllm.enumerations([FMHA_ENUMERATIONS])
    return trtllm.fmha_table(FMHA_TABLE, enums), enums


def bmm_kernels():
    enums = trtllm.enumerations([BMM_DECLARATIONS])
    return trtllm.bmm_table(BMM_TABLE, BMM_DECLARATIONS, enums), enums


def selected(kernels, enums, stated, target="sm_100a"):
    options = {name: trtllm.resolve(v, enums) for name, v in stated.items()}
    chosen = trtllm.select(kernels, options, Target.named(target))
    return [kernel.name for kernel in chosen]


def test_an_fmha_kernel_is_selected_by_its_options_not_its_name():
    kernels, enums = fmha_kernels()
    # The table states the mask as an `int`; the package names it by the
    # runner's enumerator.
    assert selected(
        kernels,
        enums,
        {
            "mDataTypeQ": "Data_type::DATA_TYPE_BF16",
            "mMaskType": "TrtllmGenAttentionMaskType::Causal",
            "mSkipsSoftmaxWhenPossible": False,
        },
    ) == ["fmhaSm100aKernel_Causal"]


def test_a_selection_keeps_any_of_several_stated_values():
    kernels, enums = fmha_kernels()
    assert selected(
        kernels,
        enums,
        {
            "mDataTypeQ": "Data_type::DATA_TYPE_BF16",
            "mMaskType": [
                "TrtllmGenAttentionMaskType::Causal",
                "TrtllmGenAttentionMaskType::Dense",
            ],
            "mSkipsSoftmaxWhenPossible": False,
        },
    ) == ["fmhaSm100aKernel_Causal", "fmhaSm100aKernel_Dense"]


def test_a_kernel_runs_on_the_targets_whose_families_include_its_own():
    kernels, enums = fmha_kernels()
    stated = {"mHeadDimQk": 128, "mSkipsSoftmaxWhenPossible": False}
    # `sm_100a` runs its own kernels and the family's; `sm_100f` only the
    # family's.
    assert len(selected(kernels, enums, stated, "sm_100a")) == 3
    assert selected(kernels, enums, stated, "sm_100f") == [
        "fmhaSm100fKernel_Causal"
    ]
    family = next(k for k in kernels if k.name == "fmhaSm100fKernel_Causal")
    assert (family.family, family.cubin, family.sha256) == (
        "sm_100f",
        "fmhaSm100fKernel_Causal.cubin",
        "dd",
    )


def test_an_option_the_table_does_not_have_is_refused():
    kernels, enums = fmha_kernels()
    with pytest.raises(SystemExit, match="no option mHeadDimV"):
        selected(kernels, enums, {"mHeadDimV": 128})


def test_a_selection_no_kernel_meets_is_refused():
    kernels, enums = fmha_kernels()
    with pytest.raises(SystemExit, match="no kernel for sm_100a"):
        selected(kernels, enums, {"mHeadDimQk": 64})


def test_an_enumerator_the_headers_do_not_declare_is_refused():
    _, enums = fmha_kernels()
    with pytest.raises(SystemExit, match="no Data_type::DATA_TYPE_E4M3"):
        trtllm.resolve("Data_type::DATA_TYPE_E4M3", enums)


def test_a_gemm_kernel_reads_its_labelled_options_and_encoded_types():
    kernels, enums = bmm_kernels()
    # `Dtype::E2m1` is the library's encoding of its macro's arguments.
    assert selected(
        kernels, enums, {"mDtypeA": "Dtype::E2m1", "mBiasType": "BiasType::M"}
    ) == ["bmm_E2m1_t128x8", "bmm_E2m1_t128x16"]  # the table's order
    assert selected(kernels, enums, {"mDtypeA": "Dtype::Bfloat16"}) == [
        "bmm_Bfloat16_t128x8"
    ]
    gemm = next(k for k in kernels if k.name == "bmm_E2m1_t128x8")
    # The library's loader names a cubin by the capitalized kernel name.
    assert (gemm.family, gemm.cubin, gemm.sha256) == (
        "sm_100f",
        "Bmm_E2m1_t128x8.cubin",
        "11",
    )
    assert gemm.options["mTileN"] == 8


def test_a_filtered_gemm_table_holds_the_selected_entries_and_their_count():
    kernels, enums = bmm_kernels()
    chosen = trtllm.select(
        kernels,
        {"mTileN": 16},
        Target.named("sm_100a"),
    )
    filtered = trtllm.filtered_header(BMM_TABLE, chosen)
    assert "tllmGenBatchedGemmListLen = 1;" in filtered
    assert '"bmm_E2m1_t128x16"' in filtered
    assert '"bmm_E2m1_t128x8"' not in filtered
    assert filtered.rstrip().endswith("} // namespace batchedGemm")
    # The filtered table is a table of the same format.
    again = trtllm.bmm_table(filtered, BMM_DECLARATIONS, enums)
    assert [kernel.name for kernel in again] == ["bmm_E2m1_t128x16"]


def test_a_manifest_option_may_take_its_value_from_a_descriptor_field():
    parameters = {
        "selected": {
            "mTileN": 8,
            "mBiasType": {
                "field": "bias",
                "values": {"none": "BiasType::None", "m": "BiasType::M"},
            },
        }
    }
    assert trtllm.manifest_selected(parameters, {"bias": "m"}) == {
        "mTileN": 8,
        "mBiasType": "BiasType::M",
    }
    with pytest.raises(SystemExit, match="no value for bias = mn"):
        trtllm.manifest_selected(parameters, {"bias": "mn"})
