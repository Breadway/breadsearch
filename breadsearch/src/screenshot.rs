//! `--screenshot` CLI mode: render breadsearch's search panel, capture it
//! via `bread-screenshots`, then exit — driven by `bread-ecosystem`'s
//! `bread-capture` orchestrator, or run standalone for one-off captures.
//!
//! Same reasoning as breadbox: one view ("search"), full known-size canvas
//! capture since the panel isn't its own layer surface. Captures whatever
//! the panel shows at settle time — normally the "Type to search…" empty
//! state, since there's no query to type in an automated run.

use clap::Parser;
use gtk4::prelude::*;
use std::path::PathBuf;
use std::time::Duration;

/// Extra settle time after `map` for the first frame to actually paint
/// before grim runs — `map` fires once the surface exists, not once
/// anything has been drawn into it.
const SETTLE_DELAY: Duration = Duration::from_millis(300);

#[derive(Parser)]
#[command(name = "breadsearch")]
pub struct Cli {
    /// Render the named view, capture it, then exit instead of running
    /// normally. Known views: "search".
    #[arg(long)]
    pub screenshot: Option<String>,

    /// PNG path to write the capture to. Required together with --screenshot.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Capture canvas width — matches the isolated compositor's output width
    /// (`bread-capture --isolate-width`).
    #[arg(long, default_value_t = 1920)]
    pub width: u32,

    /// Capture canvas height — see `width`.
    #[arg(long, default_value_t = 1080)]
    pub height: u32,
}

#[derive(Clone)]
pub struct ScreenshotRequest {
    pub view: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
}

impl Cli {
    /// `None` for a normal run. Exits the process with an error if
    /// `--screenshot` was given without `--output`, before any GTK setup
    /// happens.
    pub fn screenshot_request(&self) -> Option<ScreenshotRequest> {
        let view = self.screenshot.clone()?;
        let Some(output) = self.output.clone() else {
            eprintln!("breadsearch: --screenshot requires --output");
            std::process::exit(1);
        };
        Some(ScreenshotRequest { view, output, width: self.width, height: self.height })
    }
}

/// Wire up the given view's screenshot sequence against an already-built,
/// not-yet-presented window. Every path here ends by exiting the process —
/// it never returns control to the normal search UI.
pub fn dispatch(window: &gtk4::ApplicationWindow, req: ScreenshotRequest) {
    match req.view.as_str() {
        "search" => {
            let output = req.output;
            let (width, height) = (req.width as i32, req.height as i32);
            window.connect_map(move |_| {
                let output = output.clone();
                gtk4::glib::timeout_add_local_once(SETTLE_DELAY, move || {
                    finish(bread_screenshots::capture_region(0, 0, width, height, &output));
                });
            });
        }
        other => {
            eprintln!("breadsearch: unknown screenshot view '{other}' (known: search)");
            std::process::exit(1);
        }
    }
}

fn finish(result: anyhow::Result<()>) {
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("breadsearch: screenshot capture failed: {e}");
            std::process::exit(1);
        }
    }
}
