use bread_theme::{hex_to_rgba, ink_on, load_palette, Palette};
use breadsearch_shared::{Hit, Request, Response};
use std::{
    cell::RefCell,
    env, fs,
    path::PathBuf,
    process::Command,
    rc::Rc,
    sync::mpsc,
};

use gtk4::{
    glib,
    pango::EllipsizeMode,
    prelude::*,
    Application, ApplicationWindow, Box as GBox, CssProvider, EventControllerKey, Image, Label,
    ListBox, Orientation, PolicyType, ScrolledWindow, SearchEntry, SelectionMode,
};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

// ---- Theming ----------------------------------------------------------------

fn build_css(p: &Palette) -> String {
    let bg_panel = hex_to_rgba(&p.background, 0.60);
    format!(
        "window {{ background-color: transparent; }}\
         .launcher-bg {{ background-color: {bg_panel}; color: {on_bg}; border-radius: 8px;\
             box-shadow: 0 8px 32px rgba(0,0,0,0.6); }}\
         searchentry {{ background-color: {surface}; color: {on_surface}; caret-color: {accent};\
             border: none; outline: none; box-shadow: none;\
             padding: 12px 16px; border-radius: 6px 6px 0 0; }}\
         listbox {{ background-color: transparent; padding: 4px; }}\
         row {{ padding: 8px 12px; color: {on_bg}; background-color: transparent;\
             border-radius: 6px; }}\
         row:hover {{ background-color: {surface}; color: {on_surface}; }}\
         row:selected {{ background-color: {surface}; color: {on_surface}; }}\
         .hit-title {{ font-size: 14px; }}\
         .hit-muted {{ opacity: 0.6; font-size: 12px; }}\
         .hit-snippet {{ opacity: 0.75; font-size: 11px; font-style: italic; }}\
         .hit-score {{ opacity: 0.5; font-size: 11px; }}\
         image {{ margin-right: 8px; }}",
        bg_panel   = bg_panel,
        surface    = p.color0,
        accent     = p.color4,
        on_bg      = ink_on(&p.background),
        on_surface = ink_on(&p.color0),
    )
}

// ---- PID file toggle --------------------------------------------------------

fn pid_file() -> PathBuf {
    env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join("breadsearch.pid")
}

fn is_breadsearch_pid(pid: u32) -> bool {
    fs::read_to_string(format!("/proc/{}/comm", pid))
        .map(|s| s.trim() == "breadsearch")
        .unwrap_or(false)
}

fn toggle_or_continue() -> bool {
    let pf = pid_file();
    if let Ok(content) = fs::read_to_string(&pf) {
        if let Ok(pid) = content.trim().parse::<u32>() {
            if is_breadsearch_pid(pid) {
                let _ = Command::new("kill").arg(pid.to_string()).status();
                return false;
            }
        }
    }
    let _ = fs::write(&pf, std::process::id().to_string());
    true
}

fn cleanup_pid() {
    let _ = fs::remove_file(pid_file());
}

// ---- Row builder ------------------------------------------------------------

fn make_hit_row(hit: &Hit) -> gtk4::ListBoxRow {
    let row = gtk4::ListBoxRow::new();
    let vbox = GBox::new(Orientation::Vertical, 2);
    vbox.set_margin_start(6);
    vbox.set_margin_end(6);
    vbox.set_margin_top(4);
    vbox.set_margin_bottom(4);

    let hbox = GBox::new(Orientation::Horizontal, 0);

    let (content_type, _) =
        gtk4::gio::functions::content_type_guess(Some(&hit.path), None::<&[u8]>);
    let gicon = gtk4::gio::content_type_get_icon(&content_type);
    let img = Image::from_gicon(&gicon);
    img.set_pixel_size(24);
    hbox.append(&img);

    let title = Label::new(Some(&hit.title));
    title.add_css_class("hit-title");
    title.set_xalign(0.0);
    title.set_hexpand(true);
    title.set_ellipsize(EllipsizeMode::End);
    hbox.append(&title);

    let score_lbl = Label::new(Some(&format!("{:.0}%", hit.score * 100.0)));
    score_lbl.add_css_class("hit-score");
    score_lbl.set_xalign(1.0);
    hbox.append(&score_lbl);

    vbox.append(&hbox);

    let path_lbl = Label::new(Some(&hit.path));
    path_lbl.add_css_class("hit-muted");
    path_lbl.set_xalign(0.0);
    path_lbl.set_ellipsize(EllipsizeMode::Start);
    vbox.append(&path_lbl);

    let snippet_text = hit.snippet.lines().next().unwrap_or("").trim().to_string();
    if !snippet_text.is_empty() {
        let snippet = Label::new(Some(&snippet_text));
        snippet.add_css_class("hit-snippet");
        snippet.set_xalign(0.0);
        snippet.set_ellipsize(EllipsizeMode::End);
        vbox.append(&snippet);
    }

    row.set_child(Some(&vbox));
    unsafe { row.set_data("hit_path", hit.path.clone()) };
    row
}

fn row_path(row: &gtk4::ListBoxRow) -> Option<String> {
    unsafe { row.data::<String>("hit_path").map(|p| p.as_ref().clone()) }
}

// ---- List population --------------------------------------------------------

fn clear_list(list: &ListBox) {
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
}

fn info_row(text: &str) -> gtk4::ListBoxRow {
    let row = gtk4::ListBoxRow::new();
    let lbl = Label::new(Some(text));
    lbl.add_css_class("hit-muted");
    lbl.set_margin_top(12);
    lbl.set_margin_bottom(12);
    row.set_child(Some(&lbl));
    row.set_activatable(false);
    row.set_selectable(false);
    row
}

fn populate_list(list: &ListBox, result: std::io::Result<Response>) {
    clear_list(list);
    match result {
        Ok(Response::Hits { hits }) => {
            if hits.is_empty() {
                list.append(&info_row("No results"));
            } else {
                for hit in &hits {
                    list.append(&make_hit_row(hit));
                }
                if let Some(first) = list.row_at_index(0) {
                    list.select_row(Some(&first));
                }
            }
        }
        Ok(Response::Error { message }) => {
            list.append(&info_row(&format!("Error: {}", message)));
        }
        Err(e) => {
            list.append(&info_row(&format!("breadmill not running: {}", e)));
        }
        _ => {}
    }
}

// ---- Actions ----------------------------------------------------------------

fn open_file(path: &str) {
    let _ = Command::new("xdg-open")
        .arg(path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

fn open_folder(path: &str) {
    let parent = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let _ = Command::new("xdg-open")
        .arg(&parent)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

// ---- UI ---------------------------------------------------------------------

fn run_ui() {
    let app = Application::builder()
        .application_id("com.breadway.breadsearch")
        .build();

    let debounce_id: Rc<RefCell<Option<glib::SourceId>>> = Rc::new(RefCell::new(None));

    app.connect_activate(move |app| {
        bread_theme::gtk::apply_shared();
        bread_theme::gtk::apply_app_css(|| build_css(&load_palette()));

        {
            let user_css_path = breadsearch_shared::config_dir().join("style.css");
            let user_cell: RefCell<Option<CssProvider>> = RefCell::new(None);
            bread_theme::gtk::apply_user_css(&user_css_path, &user_cell);
        }

        let window = ApplicationWindow::builder().application(app).build();
        window.init_layer_shell();
        window.set_namespace(Some("breadsearch"));
        window.set_layer(Layer::Overlay);
        window.set_keyboard_mode(KeyboardMode::Exclusive);
        for edge in [Edge::Top, Edge::Bottom, Edge::Left, Edge::Right] {
            window.set_anchor(edge, true);
        }
        window.set_exclusive_zone(0);

        let close_all: Rc<dyn Fn()> = Rc::new({
            let w = window.clone();
            move || {
                cleanup_pid();
                w.close();
            }
        });

        let vbox = GBox::new(Orientation::Vertical, 0);
        vbox.add_css_class("launcher-bg");
        vbox.set_halign(gtk4::Align::Center);
        vbox.set_valign(gtk4::Align::Start);
        vbox.set_margin_top(120);
        vbox.set_size_request(640, -1);

        let search = SearchEntry::new();
        search.set_placeholder_text(Some("breadsearch — find anything by meaning"));
        vbox.append(&search);

        let scroll = ScrolledWindow::new();
        scroll.set_policy(PolicyType::Never, PolicyType::Automatic);
        scroll.set_max_content_height(520);
        scroll.set_propagate_natural_height(true);

        let list = ListBox::new();
        list.set_selection_mode(SelectionMode::Browse);
        list.append(&info_row("Type to search across your documents…"));

        scroll.set_child(Some(&list));
        vbox.append(&scroll);
        window.set_child(Some(&vbox));

        // Search with 150ms debounce + off-UI-thread query
        let list_s = list.clone();
        let debounce = Rc::clone(&debounce_id);

        search.connect_changed(move |entry| {
            let query = entry.text().to_string();

            if let Some(id) = debounce.borrow_mut().take() {
                id.remove();
            }

            if query.is_empty() {
                clear_list(&list_s);
                list_s.append(&info_row("Type to search across your documents…"));
                return;
            }

            let list_clone = list_s.clone();
            let debounce_clone = Rc::clone(&debounce);

            let id = glib::timeout_add_local(std::time::Duration::from_millis(150), move || {
                debounce_clone.borrow_mut().take();

                let q = query.clone();
                let (tx, rx) = mpsc::sync_channel::<std::io::Result<Response>>(1);

                std::thread::spawn(move || {
                    let req = Request::Query { query: q, limit: 10 };
                    let _ = tx.send(breadsearch_shared::send_request(&req));
                });

                // Poll via idle_add_local until the thread delivers its result.
                // Unix socket round-trips are sub-millisecond so this fires once.
                let rx = Rc::new(rx);
                let list_t = list_clone.clone();

                glib::idle_add_local(move || {
                    match rx.try_recv() {
                        Ok(result) => {
                            populate_list(&list_t, result);
                            glib::ControlFlow::Break
                        }
                        Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                        Err(mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
                    }
                });

                glib::ControlFlow::Break
            });

            *debounce.borrow_mut() = Some(id);
        });

        // Keyboard handling
        let key_ctrl = EventControllerKey::new();
        key_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        let close_k = Rc::clone(&close_all);
        let list_k = list.clone();

        key_ctrl.connect_key_pressed(move |_, key, _, mods| {
            use gtk4::gdk::Key;
            match key {
                Key::Escape => {
                    close_k();
                    glib::Propagation::Stop
                }
                Key::Return | Key::KP_Enter => {
                    if let Some(row) = list_k.selected_row() {
                        if let Some(path) = row_path(&row) {
                            if mods.contains(gtk4::gdk::ModifierType::CONTROL_MASK) {
                                open_folder(&path);
                            } else {
                                open_file(&path);
                            }
                            close_k();
                        }
                    }
                    glib::Propagation::Stop
                }
                Key::Down => {
                    let cur = list_k.selected_row().map(|r| r.index()).unwrap_or(-1);
                    let mut i = cur + 1;
                    loop {
                        match list_k.row_at_index(i) {
                            Some(r) if r.is_selectable() => {
                                list_k.select_row(Some(&r));
                                break;
                            }
                            Some(_) => i += 1,
                            None => break,
                        }
                    }
                    glib::Propagation::Stop
                }
                Key::Up => {
                    let cur = list_k.selected_row().map(|r| r.index()).unwrap_or(0);
                    let mut i = cur - 1;
                    loop {
                        if i < 0 {
                            break;
                        }
                        match list_k.row_at_index(i) {
                            Some(r) if r.is_selectable() => {
                                list_k.select_row(Some(&r));
                                break;
                            }
                            Some(_) => i -= 1,
                            None => break,
                        }
                    }
                    glib::Propagation::Stop
                }
                _ => glib::Propagation::Proceed,
            }
        });
        window.add_controller(key_ctrl);

        // Row click opens file
        let close_a = Rc::clone(&close_all);
        list.connect_row_activated(move |_, row| {
            if let Some(path) = row_path(row) {
                open_file(&path);
                close_a();
            }
        });

        // Click outside launcher panel → close
        let close_outside = Rc::clone(&close_all);
        let vbox_ref = vbox.clone();
        let win_ref = window.clone();
        let outside_click = gtk4::GestureClick::new();
        outside_click.connect_pressed(move |_, _, x, y| {
            if let Some(b) = vbox_ref.compute_bounds(&win_ref) {
                if x < b.x() as f64
                    || x > (b.x() + b.width()) as f64
                    || y < b.y() as f64
                    || y > (b.y() + b.height()) as f64
                {
                    close_outside();
                }
            }
        });
        window.add_controller(outside_click);

        window.connect_destroy(|_| cleanup_pid());
        window.present();
        search.grab_focus();
    });

    app.run();
}

// ---- Main -------------------------------------------------------------------

fn main() {
    if !toggle_or_continue() {
        return;
    }
    run_ui();
}
