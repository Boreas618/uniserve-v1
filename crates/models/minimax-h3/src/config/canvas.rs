//! The canvases a checkpoint generates, as the table a request selects one from.
//!
//! A canvas is named by the pair a request states it with: the resolution class (the target's
//! short edge) and the aspect ratio. Every H3 checkpoint generates the 768p class only: the canvas
//! rule's canvas of every named ratio at that class (`docs/minimax_h3/minimax_h3.md`).

/// One canvas a checkpoint generates: how a request names it and the raster it resolves to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Canvas {
    /// The resolution class a request names it by: its target's short edge, in pixels.
    pub short_edge: u32,
    /// The aspect ratio a request names it by, as `(width, height)` of the ratio.
    pub aspect: (u32, u32),
    /// The raster's width in pixels.
    pub width: u32,
    /// The raster's height in pixels.
    pub height: u32,
}

impl Canvas {
    /// The aspect ratio as a request names it, such as `16:9`.
    pub fn name(&self) -> String {
        format!("{}:{}", self.aspect.0, self.aspect.1)
    }
}

/// The aspect ratios a request may name besides `auto`, landscape first.
pub const NAMED_ASPECT_RATIOS: [(u32, u32); 6] =
    [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)];

/// The 768p resolution class: the short edge at which the canvas rule generates every named
/// ratio's canvas.
pub const CANVAS_SHORT_EDGE: u32 = 768;

/// The canvases every H3 checkpoint generates, and so the ones a deployment prepares layouts for.
///
/// An enumeration of layouts walks this table: a row keyed for a canvas the engine can never be
/// asked for is a cell no launch reads.
pub const fn offered_canvases() -> &'static [Canvas] {
    &NAMED_CANVASES
}

/// The canvas rule's canvases at [`CANVAS_SHORT_EDGE`]: every named aspect ratio's raster.
///
/// The rule resolves a ratio into a canvas whose short edge starts at the class, whose area is at
/// most `768 * 1344` and whose sides are the nearest multiples of [`super::CANVAS_MULTIPLE`]; the
/// request rules compute it for any ratio a request names, and this table is the result for the
/// named ones.
pub const NAMED_CANVASES: [Canvas; 6] = [
    Canvas {
        short_edge: 768,
        aspect: (21, 9),
        width: 1536,
        height: 672,
    },
    Canvas {
        short_edge: 768,
        aspect: (16, 9),
        width: 1344,
        height: 768,
    },
    Canvas {
        short_edge: 768,
        aspect: (4, 3),
        width: 1024,
        height: 768,
    },
    Canvas {
        short_edge: 768,
        aspect: (1, 1),
        width: 768,
        height: 768,
    },
    Canvas {
        short_edge: 768,
        aspect: (3, 4),
        width: 768,
        height: 1024,
    },
    Canvas {
        short_edge: 768,
        aspect: (9, 16),
        width: 768,
        height: 1344,
    },
];

impl Canvas {
    /// Whether this canvas's raster is the ruled one for its class and ratio.
    #[cfg(test)]
    pub fn is_ruled(&self) -> bool {
        // Both sides on the VAE's 32-pixel grid, and the raster's ratio the one the request names
        // within the one 32-pixel step the rule rounds: the short edge starts at the class's and
        // the area is capped, so a wide ratio's short side is shorter than the class's edge.
        let (aspect_width, aspect_height) = (i64::from(self.aspect.0), i64::from(self.aspect.1));
        let (width, height) = (i64::from(self.width), i64::from(self.height));
        self.width.is_multiple_of(super::CANVAS_MULTIPLE)
            && self.height.is_multiple_of(super::CANVAS_MULTIPLE)
            && (width * aspect_height - height * aspect_width).abs()
                <= i64::from(super::CANVAS_MULTIPLE) * aspect_width.max(aspect_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_named_table_names_every_ratio_once_on_the_vae_grid() {
        let ratios = NAMED_CANVASES.map(|canvas| canvas.aspect);
        assert_eq!(ratios, NAMED_ASPECT_RATIOS);
        for canvas in NAMED_CANVASES {
            // Every canvas survives the video VAE's 16x spatial compression on a whole number of
            // 2x2 transformer patches.
            assert_eq!(canvas.short_edge, CANVAS_SHORT_EDGE);
            assert!(canvas.is_ruled(), "{}", canvas.name());
        }
    }

    #[test]
    fn a_canvas_is_named_by_its_aspect_ratio() {
        assert_eq!(NAMED_CANVASES[1].name(), "16:9");
        assert_eq!(
            (NAMED_CANVASES[1].width, NAMED_CANVASES[1].height),
            (1344, 768)
        );
    }
}
