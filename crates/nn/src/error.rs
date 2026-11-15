//! What the neural-network library fails with.
//!
//! A function or a module propagates the failure of the tensor operation or the launch it made,
//! unchanged; nothing here retries or takes another path. The loading boundary adds the failures
//! of reading a checkpoint and converting a weight, and a module adds the one failure of its own:
//! a tensor whose shape or element type it was not loaded for.
//!
//! The tensor library carries the device backend's failure, and this crate's, across its own interfaces
//! (`Module::forward` returns the tensor library's result) as a boxed error. Converting back
//! recovers the typed cause, so a failure that crossed a `Module::forward` matches as the variant
//! it was raised as.

use std::path::PathBuf;

/// A failed function, module call, checkpoint read or weight conversion.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A tensor operation was refused: operands of other shapes, element types or devices than
    /// the operation takes.
    #[error(transparent)]
    Tensor(tensor::Error),

    /// The device side refused a call: a launch, an allocation, a collective, a profiler range,
    /// or a value outside a launch (a row count no packed product is compiled for, an element
    /// type with no host encoding). A refused launch names its operator and its key.
    #[error(transparent)]
    Ops(#[from] ops::Error),

    /// A file could not be read.
    #[error("io: {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The operating system's failure.
        #[source]
        source: std::io::Error,
    },

    /// A checkpoint's index or configuration is not well formed.
    #[error("checkpoint {path}: {problem}")]
    Checkpoint {
        /// The checkpoint's root or the file within it.
        path: PathBuf,
        /// What is wrong with it.
        problem: String,
    },

    /// A tensor a module asked for is not in the checkpoint.
    #[error("the checkpoint has no tensor `{0}`")]
    MissingTensor(String),

    /// What a module asked of a weight and what the checkpoint holds disagree.
    #[error("weight `{tensor}`: {problem}")]
    Weight {
        /// The weight's checkpoint name.
        tensor: String,
        /// The disagreement.
        problem: String,
    },

    /// A tensor's shape, element type or rank does not fit what the function or module takes.
    #[error("{owner}: {problem}")]
    Shape {
        /// The function or module that refused.
        owner: &'static str,
        /// What it was given and what it takes.
        problem: String,
    },

    /// A loader's policy does not describe the checkpoint it is asked to read.
    ///
    /// A policy (an element type, a quantization) is chosen once for a model; a stage whose
    /// weights are not of that kind would be read as something it is not.
    #[error("loader policy: {problem}")]
    Policy {
        /// What the policy asks for and what the checkpoint holds.
        problem: String,
    },

    /// A capability of the design that this crate does not implement, named precisely.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A checkpoint configuration field that is absent, of the wrong JSON type, or outside the
    /// values its reader accepts.
    ///
    /// `owner` is the document the field was read from (a sidecar's path, or the object within
    /// it), `field` the key, and `problem` what the reader requires of it, so a refused
    /// checkpoint says which value it would have to state.
    #[error("{owner}{separator}{field} {problem}", separator = if .field.is_empty() { "" } else { "." })]
    Config {
        /// The document or object the field belongs to.
        owner: String,
        /// The field's key.
        field: String,
        /// What is wrong with the value.
        problem: String,
    },

    /// A model family's own refusal: a configuration, checkpoint, request or capability it
    /// does not serve.
    ///
    /// Every family raises its refusals as this one value, naming itself by its registered id
    /// ([`ModelId`]), and a failure of the shared library it calls passes through as the variant
    /// it was raised as.
    #[error("{model}: {refusal}: {problem}")]
    Model {
        /// The family that refused.
        model: ModelId,
        /// What it refused.
        refusal: Refusal,
        /// What was asked of it and why it cannot serve it, naming the field, file or extent.
        problem: String,
    },
}

impl Error {
    /// A shape disagreement of `owner`.
    pub fn shape(owner: &'static str, problem: impl Into<String>) -> Self {
        Self::Shape {
            owner,
            problem: problem.into(),
        }
    }

    /// A policy that does not describe the checkpoint.
    pub(crate) fn policy(problem: impl Into<String>) -> Self {
        Self::Policy {
            problem: problem.into(),
        }
    }

    /// A configuration field of `owner` that does not hold what its reader requires.
    pub(crate) fn config(
        owner: impl Into<String>,
        field: impl Into<String>,
        problem: impl Into<String>,
    ) -> Self {
        Self::Config {
            owner: owner.into(),
            field: field.into(),
            problem: problem.into(),
        }
    }

    /// A disagreement between what was asked of the weight `tensor` and the checkpoint.
    pub(crate) fn weight(tensor: &str, problem: impl Into<String>) -> Self {
        Self::Weight {
            tensor: tensor.to_owned(),
            problem: problem.into(),
        }
    }
}

/// What a model family refuses ([`Error::Model`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A checkpoint configuration the family does not build, a deployment split that does not
    /// divide what it shards, or a call whose inputs do not describe the loaded model.
    Config,
    /// A checkpoint whose layout or files the family cannot open: a missing sidecar, a stage
    /// that holds no shard set.
    Checkpoint,
    /// A request, a layout or a tensor a module of the family was not loaded for.
    Request,
    /// A capability of the family's design that it does not implement, named precisely.
    Unsupported,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Config => "configuration",
            Self::Checkpoint => "checkpoint",
            Self::Request => "request",
            Self::Unsupported => "unsupported",
        })
    }
}

/// A model family as its own refusals name it: the identifier the family registers
/// ([`crate::family::Family::id`]).
///
/// A family's code raises a refusal through its id, so every family shares one error type and
/// a family that composes another's modules propagates that family's refusals unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelId(pub &'static str);

impl ModelId {
    /// The refusal `refusal` of this family, stating `problem`.
    pub fn refuse(self, refusal: Refusal, problem: impl Into<String>) -> Error {
        Error::Model {
            model: self,
            refusal,
            problem: problem.into(),
        }
    }

    /// A configuration, split or call this family does not serve ([`Refusal::Config`]).
    pub fn config(self, problem: impl Into<String>) -> Error {
        self.refuse(Refusal::Config, problem)
    }

    /// A checkpoint whose layout or files this family cannot open ([`Refusal::Checkpoint`]).
    pub fn checkpoint(self, problem: impl Into<String>) -> Error {
        self.refuse(Refusal::Checkpoint, problem)
    }

    /// A request, layout or tensor a module was not loaded for ([`Refusal::Request`]).
    pub fn request(self, problem: impl Into<String>) -> Error {
        self.refuse(Refusal::Request, problem)
    }

    /// A capability this family does not implement ([`Refusal::Unsupported`]).
    pub fn unsupported(self, problem: impl Into<String>) -> Error {
        self.refuse(Refusal::Unsupported, problem)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl From<tensor::Error> for Error {
    /// The tensor library's failure, with the typed cause recovered when the failure is one the
    /// tensor library carried for this crate or for the device backend.
    fn from(error: tensor::Error) -> Self {
        match error {
            // A captured backtrace wraps the failure it was taken for.
            tensor::Error::WithBacktrace { inner, backtrace } => match Self::from(*inner) {
                Self::Tensor(inner) => Self::Tensor(tensor::Error::WithBacktrace {
                    inner: Box::new(inner),
                    backtrace,
                }),
                typed => typed,
            },

            tensor::Error::Backend(source) => match source.downcast::<Self>() {
                Ok(own) => *own,
                Err(source) => match source.downcast::<ops::Error>() {
                    Ok(device) => Self::Ops(*device),
                    Err(source) => Self::Tensor(tensor::Error::Backend(source)),
                },
            },

            other => Self::Tensor(other),
        }
    }
}

impl From<Error> for tensor::Error {
    /// This crate's failure as the tensor library carries it across `Module::forward`.
    fn from(error: Error) -> Self {
        match error {
            Error::Tensor(error) => error,
            Error::Ops(error) => error.into(),
            other => Self::Backend(Box::new(other)),
        }
    }
}

/// A result of this crate.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn a_failure_carried_by_the_tensor_library_comes_back_as_the_variant_it_was_raised_as() {
        let raised = Error::shape("rms_norm", "a weight of 4 elements for rows of 8");
        let carried = tensor::Error::from(raised);
        assert!(matches!(
            Error::from(carried),
            Error::Shape {
                owner: "rms_norm",
                ..
            }
        ));

        let device = ops::Error::NoContext {
            operation: "a launch",
        };
        let carried = tensor::Error::from(Error::from(device));
        assert!(matches!(
            Error::from(carried),
            Error::Ops(ops::Error::NoContext { .. })
        ));
    }

    #[test]
    fn a_tensor_operation_s_own_refusal_stays_the_tensor_library_s() {
        let refused = tensor::Error::RequiresContiguous { op: "all_reduce" };
        assert!(matches!(
            Error::from(refused),
            Error::Tensor(tensor::Error::RequiresContiguous { .. })
        ));
    }
}
