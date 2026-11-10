//! Image input of a chat family, as request processing lowers it.
//!
//! A chat template writes one placeholder token where a message's image stands. Request processing
//! replaces that placeholder by the image's block: the block takes as many cache rows as the
//! family's vision tower encodes the image to, and advances the rotary position of the text after
//! it by its span. Both follow from the image's size by the family's processor rule, which the
//! family states once ([`ImageProcessor::block`]); the worker sizes the image by the same rule when
//! it encodes it.

/// The block an image takes in a request's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageBlock {
    /// Cache rows the block writes: the tower's merged rows of the image.
    pub tokens: u32,
    /// Logical positions the block spans: the advance of the rotary position of the text after
    /// it.
    pub positions: u32,
}

/// A chat family's image processor, bound to one checkpoint's processor files.
pub trait ImageProcessor: Send + Sync {
    /// The token text the family's chat template writes for one image, which the image's block
    /// replaces.
    fn placeholder(&self) -> &str;

    /// The text a chat template that flattens message content to a string is given for one
    /// image: the family's own image markup around [`ImageProcessor::placeholder`], as the
    /// template writes it for an image part it reads itself.
    fn markup(&self) -> &str;

    /// The block an image of `width` x `height` pixels takes.
    ///
    /// # Errors
    ///
    /// Refuses an image the processor cannot size, naming why.
    fn block(&self, width: u32, height: u32) -> Result<ImageBlock, String>;
}

/// The image side of a chat family, as request processing serves it.
pub trait ImageFamily: Send + Sync {
    /// The checkpoint files the processor reads, relative to the checkpoint's root.
    fn processor_files(&self) -> &'static [&'static str];

    /// Binds the processor to the contents of [`ImageFamily::processor_files`], in that order.
    ///
    /// # Errors
    ///
    /// Fails when the files are not the family's processor files.
    fn bind(&self, files: &[String]) -> Result<Box<dyn ImageProcessor>, String>;
}
