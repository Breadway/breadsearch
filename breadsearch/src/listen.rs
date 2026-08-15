//! Long-running command subscription for `bread.command.search.*`.
//!
//! `breadsearch` is still a one-shot toggle overlay by default.
//! `breadsearch listen` is the optional persistent process that can honor
//! bus commands. See `EVENTS.md`.

use bread_utils::bread_client::{BreadClient, BreadEvent};

use crate::bread_events::APP_ID;

/// Subscribe to `bread.command.search.**` and block until the process is killed.
///
/// breadd being absent is not an error: [`BreadClient::subscribe`] reconnects
/// with backoff, and `on_event` simply isn't called until the daemon is up.
pub fn run() {
    let client = BreadClient::connect(APP_ID);
    if client.health().is_none() {
        eprintln!(
            "breadsearch: breadd unreachable; command subscription will connect when it comes back"
        );
    }

    let _commands = client.subscribe("bread.command.search.**", |event| {
        handle_command(&event);
    });

    eprintln!("breadsearch: listening for bread.command.search.**");
    loop {
        std::thread::park();
    }
}

/// Reacts to `bread.command.search.*` verbs. Only `open` is honored today —
/// other verbs are ignored, not stubbed as no-ops that pretend to succeed.
fn handle_command(event: &BreadEvent) {
    let Some(verb) = command_verb(&event.event) else {
        return;
    };
    match verb {
        "open" => handle_open(),
        other => {
            eprintln!("breadsearch: ignoring unrecognized bread.command.search.{other}");
        }
    }
}

fn handle_open() {
    // Same as running `breadsearch` from a keybind: the PID-file toggle
    // shows the overlay (or dismisses it if it is already up).
    let result = spawn_self();
    let client = BreadClient::connect(APP_ID);
    match result {
        Ok(_) => client.emit("bread.search.open.done", serde_json::json!({})),
        Err(e) => {
            eprintln!("breadsearch: bread.command.search.open failed: {e}");
            client.emit(
                "bread.search.open.failed",
                serde_json::json!({ "error": e.to_string() }),
            );
        }
    }
}

fn spawn_self() -> std::io::Result<std::process::Child> {
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("breadsearch"));
    std::process::Command::new(exe).spawn()
}

fn command_verb(event_name: &str) -> Option<&str> {
    event_name.strip_prefix("bread.command.search.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_verb_strips_search_prefix() {
        assert_eq!(command_verb("bread.command.search.open"), Some("open"));
        assert_eq!(command_verb("bread.command.search.query"), Some("query"));
        assert_eq!(command_verb("bread.command.box.open"), None);
        assert_eq!(command_verb("bread.search.opened"), None);
    }
}
