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
the same stance as `bread-emit`). Command verbs are only received while
`breadsearch listen` is running — that process holds the
`bread.command.search.**` subscription open. breadmill, the indexing
daemon, does not talk to breadd.

## Events published (`bread.search.*`)

| Event | Data | When |
|-------|------|------|
| `bread.search.opened` | `{}` | The overlay window maps (the search panel is shown). |
| `bread.search.opened_result` | `{ "path": "<hit path>" }` | The user opens a hit — Enter / click opens the file, Ctrl+Enter reveals its folder. `path` is the hit's document path, not the parent folder. |
| `bread.search.open.done` | `{}` | `bread.command.search.open` was received and `breadsearch` was spawned. This is the command confirmation, not proof the overlay mapped — the spawned process is the same PID-file toggle as a keybind. |
| `bread.search.open.failed` | `{ "error": "<message>" }` | `bread.command.search.open` was received but this binary could not be started. |

## Commands honored (`bread.command.search.*`)

These are only received while `breadsearch listen` is running. Publishing a
command with no subscriber is a silent no-op — that is the documented
bread convention, not a breadsearch bug.

| Verb | Data | Effect |
|------|------|--------|
| `open` | none | Same as running `breadsearch` (PID-file toggle: show the overlay, or dismiss it if it is already up). Emits `bread.search.open.done` / `.failed`. |

```lua
bread.spawn(function()
    bread.emit("bread.command.search.open")
    bread.wait("bread.search.open.done", { timeout = 5000 })
end)
```

### Not implemented: extra verbs

There is no `query` / `close` / `reindex` command verb. breadmill already
has its own query socket; inventing a bus query plane would be a new
product surface. If/when that exists, add the corresponding
`bread.command.search.*` verb at the same time, not stubbed as a no-op
ahead of it.

## Fail-safe behavior

- If breadd isn't installed or isn't running, `emit` is a silent no-op
  (`BreadClient::emit` never blocks or errors the caller) and the
  command subscription simply never receives anything — breadsearch's
  overlay and breadmill's indexing/query path are entirely unaffected.
- If breadd restarts, the command subscription reconnects automatically
  (`BreadClient::subscribe`'s background thread has its own backoff
  loop); no restart of `breadsearch listen` is needed.
- If `breadsearch listen` is not running, commands are a graceful no-op at
  the bus (no subscriber). The overlay CLI still works.
