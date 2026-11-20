//! The generator of the native packages' C ABI headers, and the check that the committed headers
//! are its output.
//!
//! A header is cbindgen's rendering of the crate source that defines one operator interface's
//! items, restricted to those items and named as the packages' sources name them, followed by the
//! size and every field offset of each structure as Rust lays it out ([`Layout`]), asserted. The
//! committed files (`crates/ops/python/src/uniserve_kernels/include/`) are compared with the
//! output; `UPDATE_EXPECT=1 cargo test -p ops abi::headers` writes it there, which is the only way
//! the files change.
//!
//! cbindgen reads one source file (and the modules it declares) per header rather than the whole
//! crate: it does not resolve Rust paths, and several operator modules name their structure
//! `Call`.

use std::collections::HashMap;
use std::fmt::Write as _;

use cbindgen::{Builder, Config, DocumentationStyle, ItemType, Language, RenameRule};
use expect_test::{ExpectFile, expect_file};

use super::Layout;
use crate::{attention, conv, fp4, frame, gemm, kernel, moe, mx};

/// The committed header `$name`: a file of the kernel-source library in this crate's tree,
/// named from the crate's directory so the comparison reads this tree's file wherever the
/// workspace is.
macro_rules! committed {
    ($name:literal) => {
        expect_file![concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/python/src/uniserve_kernels/include/",
            $name
        )]
    };
}

/// The header every operator header includes: the element-type numbering and the error buffer.
const SHARED: &str = "uniserve_abi.h";

/// The C name of the error buffer's size ([`super::MESSAGE`]), which every call structure's
/// `error` field is declared with.
const MESSAGE: (&str, &str) = ("MESSAGE", "UNISERVE_MESSAGE");

/// A structure a header declares: its C name, and its size and field offsets as Rust lays it out.
struct Structure {
    name: &'static str,
    size: usize,
    fields: &'static [(&'static str, usize)],
}

/// One generated header.
struct Header {
    /// The committed file the generator's output must equal.
    file: ExpectFile,
    /// The crate source the items are read from, relative to the crate's root.
    source: &'static str,
    /// What the header declares, for its banner.
    subject: &'static str,
    /// Whether this is the header every operator header includes, which declares no structure.
    shared: bool,
    /// Each item's name in this crate and its name in C.
    items: Vec<(&'static str, &'static str)>,
    /// The structures whose layouts the header asserts.
    layouts: Vec<Structure>,
}

impl Header {
    /// An operator interface's header, which includes the shared one.
    fn new(file: ExpectFile, source: &'static str, subject: &'static str) -> Self {
        Self {
            file,
            source,
            subject,
            shared: false,
            items: Vec::new(),
            layouts: Vec::new(),
        }
    }

    /// The header the operator headers share, read from this module's own source.
    fn shared(file: ExpectFile, subject: &'static str) -> Self {
        Self {
            shared: true,
            ..Self::new(file, "src/abi.rs", subject)
        }
    }

    /// Declares the call structure `T` as `name`, with its layout asserted.
    fn structure<T: Layout>(mut self, name: &'static str) -> Self {
        self.items.push((T::NAME, name));
        self.layouts.push(Structure {
            name,
            size: T::SIZE,
            fields: T::FIELDS,
        });
        self
    }

    /// Declares the enumeration or constant this crate names `item` as `name`.
    fn item(mut self, item: &'static str, name: &'static str) -> Self {
        self.items.push((item, name));
        self
    }

    /// cbindgen's configuration of this header.
    fn config(&self) -> Config {
        let mut config = Config {
            language: Language::Cxx,
            header: Some(self.banner()),
            trailer: (!self.layouts.is_empty()).then(|| self.assertions()),
            pragma_once: true,
            // The declarations need only the fixed-width integers; the assertions need `offsetof`.
            no_includes: true,
            sys_includes: vec!["cstddef".to_owned(), "cstdint".to_owned()],
            documentation: true,
            documentation_style: DocumentationStyle::Cxx,
            usize_is_size_t: true,
            ..Config::default()
        };
        if !self.shared {
            config.includes = vec![SHARED.to_owned()];
        }

        // An enumerator is its enumeration's C name and its variant, in capitals: a build module
        // spells it from a descriptor field the same way.
        config.enumeration.enum_class = false;
        config.enumeration.prefix_with_name = true;
        config.enumeration.rename_variants = RenameRule::UpperCase;

        // Only the header's items, and what their fields name, are declared; an operator module's
        // other constants are not the packages'.
        config.export.include = self
            .items
            .iter()
            .map(|(item, _)| (*item).to_owned())
            .collect();
        config.export.rename = self
            .items
            .iter()
            .chain([&MESSAGE])
            .map(|(item, name)| ((*item).to_owned(), (*name).to_owned()))
            .collect::<HashMap<_, _>>();
        config.export.item_types = if self.shared {
            vec![ItemType::Enums, ItemType::Constants]
        } else {
            vec![ItemType::Enums, ItemType::Structs]
        };
        config
    }

    /// The comment the header opens with: what it declares, where from, and how it is checked
    /// and written.
    fn banner(&self) -> String {
        let contents = if self.shared {
            "An enumerator is its enumeration's name and its variant in capitals, the variant being \
             the value's spelling in a kernel key: `bf16` is `UNISERVE_DTYPE_BF16`."
        } else {
            "The assertions at the end state the size and every field offset of each structure as \
             Rust lays it out."
        };
        let paragraphs = [
            "SPDX-License-Identifier: Apache-2.0".to_owned(),
            format!(
                "{}, generated by cbindgen from the operator crate's `crates/ops/{}` (`ops::abi`): \
                 do not edit.",
                self.subject, self.source
            ),
            contents.to_owned(),
            "`cargo test -p ops` compares this file with the generator's output, and this command \
             writes it:"
                .to_owned(),
        ];
        let mut banner = paragraphs
            .iter()
            .map(|paragraph| comment(paragraph))
            .collect::<Vec<_>>()
            .join("//\n");
        banner.push_str("//\n//   UPDATE_EXPECT=1 cargo test -p ops abi::headers");
        banner
    }

    /// The trailer: each structure's size and field offsets, as Rust lays it out.
    fn assertions(&self) -> String {
        let mut text = String::new();
        for Structure { name, size, fields } in &self.layouts {
            let message = format!("\"the operator crate's layout of {name}\"");
            writeln!(text, "static_assert(sizeof({name}) == {size}, {message});").unwrap();
            for (field, offset) in *fields {
                writeln!(
                    text,
                    "static_assert(offsetof({name}, {field}) == {offset}, {message});"
                )
                .unwrap();
            }
        }
        text.trim_end().to_owned()
    }

    /// The header cbindgen renders.
    fn render(&self) -> String {
        let source = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), self.source);
        let bindings = Builder::new()
            .with_config(self.config())
            .with_src(&source)
            .generate()
            .unwrap_or_else(|error| panic!("cbindgen cannot read {source}: {error}"));
        let mut rendered = Vec::new();
        bindings.write(&mut rendered);
        String::from_utf8(rendered).unwrap()
    }
}

/// `text` as `//` comment lines no wider than the headers' 100 columns, words kept whole.
fn comment(text: &str) -> String {
    const WIDTH: usize = 100 - "// ".len();
    let mut lines = vec![String::new()];
    for word in text.split_whitespace() {
        let line = lines.last_mut().unwrap();
        if !line.is_empty() && line.len() + 1 + word.len() > WIDTH {
            lines.push(word.to_owned());
        } else {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
    }
    lines.iter().map(|line| format!("// {line}\n")).collect()
}

/// Every header the native packages compile against.
fn headers() -> Vec<Header> {
    vec![
        Header::shared(
            committed!("uniserve_abi.h"),
            "What every operator header shares: the element types' numbers and the error buffer",
        )
        .item("ElementType", "uniserve_dtype")
        .item(MESSAGE.0, MESSAGE.1),
        Header::new(
            committed!("uniserve_load.h"),
            "src/kernel.rs",
            "The call structure of a native module's load entry, which hands it the directory of its \
             files",
        )
        .structure::<kernel::LoadCall>("uniserve_load_call"),
        Header::new(
            committed!("uniserve_conv.h"),
            "src/conv.rs",
            "The call structure of `conv`'s packages",
        )
        .structure::<conv::Call>("uniserve_conv_call"),
        Header::new(
            committed!("uniserve_gemm.h"),
            "src/gemm.rs",
            "The call structure of `gemm`'s packages, and the descriptor's weight orders and \
             epilogues",
        )
        .structure::<gemm::Call>("UniserveGemmCall")
        .item("WeightOrder", "uniserve_weight_order")
        .item("Epilogue", "uniserve_epilogue"),
        Header::new(
            committed!("uniserve_nvfp4_gemm.h"),
            "src/fp4.rs",
            "The call structure of `nvfp4_gemm`'s packages",
        )
        .structure::<fp4::Call>("uniserve_nvfp4_gemm_call"),
        Header::new(
            committed!("uniserve_mxfp8_gemm.h"),
            "src/mx.rs",
            "The call structure of `mxfp8_gemm`'s packages",
        )
        .structure::<mx::Call>("uniserve_mxfp8_gemm_call"),
        Header::new(
            committed!("uniserve_frame_moments.h"),
            "src/frame.rs",
            "The call structure of `frame_moments`'s packages",
        )
        .structure::<frame::FrameMomentsCall>("uniserve_frame_moments_call"),
        Header::new(
            committed!("uniserve_frame_norm_pad.h"),
            "src/frame.rs",
            "The call structure of `frame_norm_pad`'s packages, and the normalizations it numbers",
        )
        .structure::<frame::FrameNormPadCall>("uniserve_frame_norm_pad_call")
        .item("Normalization", "uniserve_normalization"),
        Header::new(
            committed!("uniserve_paged_attention.h"),
            "src/attention/mod.rs",
            "The call structure of `paged_attention`'s packages, and the phases it numbers",
        )
        .structure::<attention::Call>("uniserve_paged_attention_call")
        .item("Phase", "uniserve_phase"),
        Header::new(
            committed!("uniserve_dense_attention.h"),
            "src/attention/mod.rs",
            "The call structure of `dense_attention`'s packages, and the masks a module is \
             compiled for",
        )
        .structure::<attention::dense::DenseCall>("uniserve_dense_attention_call")
        .item("Mask", "uniserve_mask"),
        Header::new(
            committed!("uniserve_sparse_attention.h"),
            "src/attention/sparse.rs",
            "The call structure of `sparse_attention`'s packages",
        )
        .structure::<attention::sparse::SparseCall>("uniserve_sparse_attention_call"),
        Header::new(
            committed!("uniserve_moe_sort.h"),
            "src/moe/sort.rs",
            "The call structure of `moe_sort`'s packages",
        )
        .structure::<moe::sort::Call>("uniserve_moe_sort_call"),
        Header::new(
            committed!("uniserve_moe_unpermute.h"),
            "src/moe/combine.rs",
            "The call structure of `moe_unpermute`'s packages, and the row layouts it numbers",
        )
        .structure::<moe::combine::Call>("uniserve_unpermute_call")
        .item("Layout", "uniserve_unpermute_layout"),
        Header::new(
            committed!("uniserve_expert_exchange.h"),
            "src/moe/exchange.rs",
            "The structures of `expert_exchange`'s packages: the launch, the fused layer it may \
             read, the transport's lifecycle and views, and the group channel they are handed",
        )
        .structure::<moe::exchange::FusedLayer>("uniserve_fused_layer")
        .structure::<moe::exchange::Call>("uniserve_exchange_call")
        .structure::<moe::exchange::HostGroup>("UniserveGroup")
        .structure::<moe::exchange::Lifecycle>("uniserve_exchange_lifecycle")
        .structure::<moe::exchange::Info>("uniserve_exchange_info")
        .item("Mode", "uniserve_exchange")
        .item("ExpertWeights", "uniserve_exchange_weights"),
        Header::new(
            committed!("uniserve_gather_gemm.h"),
            "src/moe/grouped.rs",
            "The call structure of `gather_gemm`'s packages",
        )
        .structure::<moe::grouped::GatherCall>("uniserve_gather_call"),
        Header::new(
            committed!("uniserve_route_gemm.h"),
            "src/moe/grouped.rs",
            "The call structure of `route_gemm`'s packages",
        )
        .structure::<moe::grouped::RouteCall>("uniserve_route_call"),
        Header::new(
            committed!("uniserve_gather_gemm_nvfp4.h"),
            "src/moe/grouped.rs",
            "The call structure of `gather_gemm_nvfp4`'s packages",
        )
        .structure::<moe::grouped::GatherNvfp4Call>("uniserve_gather_gemm_nvfp4_call"),
        Header::new(
            committed!("uniserve_route_gemm_nvfp4.h"),
            "src/moe/grouped.rs",
            "The call structure of `route_gemm_nvfp4`'s packages",
        )
        .structure::<moe::grouped::RouteNvfp4Call>("uniserve_route_gemm_nvfp4_call"),
        Header::new(
            committed!("uniserve_gather_gemm_mxfp4.h"),
            "src/moe/mxfp4.rs",
            "The call structure of `gather_gemm_mxfp4`'s packages",
        )
        .structure::<moe::mxfp4::GatherMxfp4Call>("uniserve_gather_gemm_mxfp4_call"),
        Header::new(
            committed!("uniserve_route_gemm_mxfp4.h"),
            "src/moe/mxfp4.rs",
            "The call structure of `route_gemm_mxfp4`'s packages",
        )
        .structure::<moe::mxfp4::RouteMxfp4Call>("uniserve_route_gemm_mxfp4_call"),
    ]
}

#[test]
fn the_committed_headers_are_the_generators_output() {
    for header in headers() {
        header.file.assert_eq(&header.render());
    }
}

/// A build module spells the enumerator a descriptor field names from the field's spelling in the
/// key, as `<enumeration>_<spelling>` in capitals: each rendered variant's name is its spelling.
#[test]
fn a_key_spells_the_enumerators_of_its_fields() {
    fn spelled<T: std::fmt::Debug + std::fmt::Display>(values: &[T]) {
        for value in values {
            assert_eq!(
                format!("{value:?}").to_uppercase(),
                value.to_string().to_uppercase()
            );
        }
    }

    spelled(&[gemm::WeightOrder::KxN, gemm::WeightOrder::NxK]);
    spelled(&[gemm::Epilogue::None, gemm::Epilogue::Bias]);
    spelled(&[attention::Mask::Causal, attention::Mask::Dense]);
    spelled(&[attention::Phase::Prefill, attention::Phase::Decode]);
    spelled(&[
        moe::exchange::Mode::Dispatch,
        moe::exchange::Mode::Combine,
        moe::exchange::Mode::Fused,
    ]);
    spelled(&[
        moe::exchange::ExpertWeights::None,
        moe::exchange::ExpertWeights::Nvfp4,
        moe::exchange::ExpertWeights::Mxfp4,
    ]);
    spelled(&[
        moe::combine::Layout::Permuted,
        moe::combine::Layout::Expanded,
    ]);
}
