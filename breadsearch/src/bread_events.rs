//! `bread.search.*` event integration — optional, non-blocking. See
//! `EVENTS.md` at the repo root for the full contract. breadsearch works
//! identically with or without breadd running; every call here is
//! fire-and-forget (`BreadClient::emit` never blocks or errors this
//! process) so a missing or restarting breadd never affects the overlay.

use bread_utils::bread_client::BreadClient;

/// This app's id in bread's sibling-app namespace registry
/// (`bread_shared::apps::KNOWN_APPS`) — events publish as `bread.search.*`.
pub const APP_ID: &str = "search";

fn client() -> BreadClient {
    BreadClient::connect(APP_ID)
}

pub fn emit_opened() {
    client().emit("bread.search.opened", serde_json::json!({}));
}

pub fn emit_opened_result(path: &str) {
    client().emit(
        "bread.search.opened_result",
        serde_json::json!({ "path": path }),
    );
}
