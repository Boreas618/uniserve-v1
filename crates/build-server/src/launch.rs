//! A Triton package's launch table: how a descriptor binds its kernel's
//! parameters (design section 5.2).
//!
//! A Triton package's kernel is a `@triton.jit` function of its source module,
//! `src/<op>.py`; the kernel's own parameter list states the names and the
//! order of its arguments. The launch table, `[launch]` in the manifest,
//! states what the parameter list does not: the type of each argument the
//! operator passes at a launch, the value of each compile-time parameter, the
//! warp count, and the arguments the operator's views keep aligned. Each of
//! them is a literal or is read from the descriptor's fields, so one table
//! states the compilation of every descriptor the package covers, and the
//! Triton driver (`uniserve_build_server.triton`) is the one implementation of
//! the rest:
//!
//! - an element type a field names is spelled as Triton spells it by the
//!   driver's one table, keyed by the names `ops::abi` prints;
//! - the kernel's `PDL` parameter is its programmatic dependent launch: true
//!   when the descriptor's `pdl` field is and the target has dependent launch
//!   (`targets.toml`), and the module's launch carries the attribute exactly
//!   then;
//! - an aligned integer argument whose value the descriptor's field of the
//!   same name fixes is attributed as a multiple of sixteen only when that
//!   value is one.
//!
//! A package whose compilation needs a value the fields give only through a
//! computation (the accumulator a reduction's element type implies, the
//! steps a conversion takes) names a derivation, a function of its source
//! module that returns those values as further fields; nothing else of the
//! package is code.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The name a Triton package's kernel parameter takes for its programmatic
/// dependent launch, bound by the driver ([`Launch`]).
pub const DEPENDENT_LAUNCH_PARAMETER: &str = "PDL";

/// The descriptor field that asks for a programmatic dependent launch.
pub const DEPENDENT_LAUNCH_FIELD: &str = "pdl";

/// A Triton package's launch table, `[launch]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// The kernel: a `@triton.jit` function of the package's source module
    /// `src/<op>.py`, or the one a field's value selects of several.
    pub kernel: Kernel,
    /// The warp count of one program.
    pub warps: Warps,
    /// The arguments the operator's views keep on sixteen-element bounds,
    /// which lets the compiler load and store whole vectors. A module is
    /// compiled ahead of its launches, so nothing observes a pointer: the
    /// statement is the package's, about its operator's views.
    #[serde(default)]
    pub aligned: Aligned,
    /// A function of the source module that returns further fields computed
    /// from the descriptor's (`derive(fields, parameters, target)`), which the
    /// types, constants and warps may read as they read the descriptor's own;
    /// it refuses a descriptor the kernel cannot be compiled for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derive: Option<String>,
    /// The type of each argument the operator passes at a launch, by the
    /// kernel's parameter name.
    #[serde(default)]
    pub types: BTreeMap<String, ArgumentType>,
    /// The value of each compile-time parameter but `PDL`, by name.
    #[serde(default)]
    pub constants: BTreeMap<String, Constant>,
    /// Further options of `triton.compile`, by Triton's names, each a value
    /// as a constant states one: `enable_fp_fusion = false` for a kernel whose
    /// arithmetic must round at every operation, `num_stages = { field =
    /// "stages" }` for a pipeline depth a descriptor chooses. The warp count
    /// and the dependent launch are the table's and the driver's, not options.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, Constant>,
}

/// The kernel a launch table compiles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Kernel {
    /// One kernel for every descriptor.
    Named(String),
    /// The kernel the field's value selects: `{ field = "function", values =
    /// { gather = "_gather_kernel", scatter_add = "_scatter_kernel" } }`. The
    /// types and constants are then those of every kernel the table selects,
    /// each kernel taking the ones its parameters name.
    Selected {
        field: String,
        values: BTreeMap<String, String>,
    },
}

/// The arguments a launch table states aligned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Aligned {
    /// The same arguments for every descriptor.
    Listed(Vec<String>),
    /// The arguments the field's value selects: `{ field = "whole_rows",
    /// values = { true = ["x", "y"], false = [] } }`, for views whose rows
    /// start on whole vectors only when the row width is a multiple of one.
    Selected {
        field: String,
        values: BTreeMap<String, Vec<String>>,
    },
}

impl Default for Aligned {
    fn default() -> Self {
        Self::Listed(Vec::new())
    }
}

/// The warp count of one program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Warps {
    /// One count for every descriptor.
    Fixed(u32),
    /// The count of the first step whose threshold the field's value reaches,
    /// the steps in descending order of their thresholds, else `otherwise`:
    /// `{ field = "tile", steps = [[2048, 8]], otherwise = 4 }` gives a tile
    /// of 2048 or more eight warps and a smaller one four.
    Steps {
        field: String,
        steps: Vec<(u64, u32)>,
        otherwise: u32,
    },
    /// The field's value, which a derivation computes where the count follows
    /// from several fields: `{ field = "warps" }`.
    Field { field: String },
}

/// The type of one launch argument.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum ArgumentType {
    /// A pointer to elements of the type the field names: `{ pointer = "dtype" }`.
    Pointer { pointer: String },
    /// A scalar of the element type the field names: `{ element = "dtype" }`.
    Element { element: String },
    /// The type the field's value selects: `{ field = "wide", values = { true
    /// = "i64", false = "i32" } }`.
    Selected {
        field: String,
        values: BTreeMap<String, String>,
    },
    /// A Triton type: `"i32"`, `"fp32"`, `"*fp32"`.
    Named(String),
}

/// The value of one compile-time parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Constant {
    /// The value the field's value selects.
    Selected {
        field: String,
        values: BTreeMap<String, toml::Value>,
    },
    /// The field's value; `absent` is the value when the descriptor does not
    /// name the field (a variant the operator prints only when it is set).
    Field {
        field: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        absent: Option<toml::Value>,
    },
    /// The value of one of the manifest's `[parameters]`, which a revision's
    /// arguments may replace: `{ parameter = "tile" }`.
    Parameter { parameter: String },
    /// A literal.
    Literal(toml::Value),
}

impl ArgumentType {
    /// The descriptor field the type is read from, when it is read from one.
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::Pointer { pointer: field }
            | Self::Element { element: field }
            | Self::Selected { field, .. } => Some(field),
            Self::Named(_) => None,
        }
    }
}

impl Constant {
    /// The descriptor field the value is read from, when it is read from one.
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::Selected { field, .. } | Self::Field { field, .. } => Some(field),
            Self::Parameter { .. } | Self::Literal(_) => None,
        }
    }

    /// The manifest parameter the value is read from, when it is read from one.
    pub fn parameter(&self) -> Option<&str> {
        match self {
            Self::Parameter { parameter } => Some(parameter),
            _ => None,
        }
    }
}

impl Launch {
    /// The fields the table reads, so a manifest that reads one no descriptor
    /// declares is refused when it is loaded. A derived field is the
    /// derivation's to give, so a package with a derivation is checked by its
    /// compilation alone.
    pub fn fields(&self) -> Vec<&str> {
        let mut fields = Vec::new();
        if let Kernel::Selected { field, .. } = &self.kernel {
            fields.push(field.as_str());
        }
        if let Warps::Steps { field, .. } | Warps::Field { field } = &self.warps {
            fields.push(field.as_str());
        }
        if let Aligned::Selected { field, .. } = &self.aligned {
            fields.push(field.as_str());
        }
        fields.extend(self.types.values().filter_map(ArgumentType::field));
        fields.extend(
            self.constants
                .values()
                .chain(self.options.values())
                .filter_map(Constant::field),
        );
        fields
    }

    /// The manifest parameters the table reads.
    pub fn parameters(&self) -> Vec<&str> {
        self.constants
            .values()
            .chain(self.options.values())
            .filter_map(Constant::parameter)
            .collect()
    }

    /// Why the table may not be stated, when it may not.
    pub fn invalid(&self) -> Option<String> {
        let named: Vec<&String> = match &self.kernel {
            Kernel::Named(name) => vec![name],
            Kernel::Selected { values, .. } => values.values().collect(),
        };
        if named.is_empty() || named.iter().any(|name| name.is_empty()) {
            return Some("the launch table names no kernel".into());
        }
        if self.constants.contains_key(DEPENDENT_LAUNCH_PARAMETER) {
            return Some(format!(
                "`{DEPENDENT_LAUNCH_PARAMETER}` is the dependent launch, which the driver binds \
                 from the `{DEPENDENT_LAUNCH_FIELD}` field and the target"
            ));
        }
        if let Warps::Steps { steps, .. } = &self.warps
            && steps.windows(2).any(|pair| pair[0].0 <= pair[1].0)
        {
            return Some(
                "the warp steps are listed in descending order of their thresholds".into(),
            );
        }
        if let Some(option) = ["num_warps", "launch_pdl"]
            .into_iter()
            .find(|option| self.options.contains_key(*option))
        {
            return Some(format!(
                "`{option}` is the launch table's `warps` and the driver's dependent launch, not an \
                 option"
            ));
        }
        if self.derive.as_deref().is_some_and(str::is_empty) {
            return Some("a derivation names a function".into());
        }
        None
    }
}
