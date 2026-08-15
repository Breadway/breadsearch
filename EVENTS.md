# breadsearch — bread event integration

breadsearch is a standalone document-search overlay: it works exactly the
same with or without `breadd` running. When breadd *is* present, the
`breadsearch` GTK overlay publishes events into the shared bread automation
fabric. See the parent `bread` repo's `Documentation.md` — specifically its
"Namespaces" and "Integrating a bread\* app" sections — for the general
convention this follows.

App id: **`search`**. Transport: `bread-utils`'s `bread_client` module
(feature `bread-client`) — the overlay links it directly. Each `emit` is
its own short-lived connection (`BreadClient::emit` is fire-and-forget,
the same stance as `bread-emit`). breadmill, the indexing daemon, does
not talk to breadd.

## Events published (`bread.search.*`)

| Event | Data | When |
|-------|------|------|
| `bread.search.opened` | `{}` | The overlay window maps (the search panel is shown). |
| `bread.search.opened_result` | `{ "path": "<hit path>" }` | The user opens a hit — Enter / click opens the file, Ctrl+Enter reveals its folder. `path` is the hit's document path, not the parent folder. |

## Commands honored (`bread.command.search.*`)

None. The overlay is a short-lived toggle process with no existing command
surface (no show/hide/query IPC beyond the PID-file toggle and breadmill's
own query socket). Adding verbs would mean inventing a control plane that
does not exist; if/when breadsearch grows one, the corresponding
`bread.command.search.*` verbs should be added at the same time, not stubbed
out ahead of it.

## Fail-safe behavior

- If breadd isn't installed or isn't running, `emit` is a silent no-op
  (`BreadClient::emit` never blocks or errors the caller) — breadsearch's
  overlay and breadmill's indexing/query path are entirely unaffected.
