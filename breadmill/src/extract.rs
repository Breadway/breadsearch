use std::{
    fs, io::Read, path::Path,
};

pub type ExtractResult = Result<String, String>;

/// Extract plain text from a file based on its extension.
/// Returns Err on hard failures; Err with message if format unsupported.
pub fn extract(path: &Path) -> ExtractResult {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "md" | "txt" | "org" => read_text(path),
        "pdf" => extract_pdf(path),
        "docx" => extract_docx(path),
        "odt" => extract_odt(path),
        other => Err(format!("unsupported extension: {}", other)),
    }
}

fn read_text(path: &Path) -> ExtractResult {
    fs::read_to_string(path).map_err(|e| e.to_string())
}

fn extract_pdf(path: &Path) -> ExtractResult {
    // pdf-extract panics on some malformed PDFs; catch_unwind prevents indexer thread death.
    let path = path.to_path_buf();
    match std::panic::catch_unwind(|| pdf_extract::extract_text(&path)) {
        Ok(result) => result.map_err(|e| e.to_string()),
        Err(_) => Err("pdf-extract panicked on malformed content stream".into()),
    }
}

fn extract_docx(path: &Path) -> ExtractResult {
    extract_office_xml(path, "word/document.xml", "w:t")
}

fn extract_odt(path: &Path) -> ExtractResult {
    extract_office_xml(path, "content.xml", "text:p")
}

/// Open a zip-based office format and concatenate text from the named XML entry.
/// We grab all Text events as a best-effort extraction.
fn extract_office_xml(path: &Path, xml_entry: &str, _tag_hint: &str) -> ExtractResult {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

    let mut xml_bytes = Vec::new();
    archive
        .by_name(xml_entry)
        .map_err(|e| format!("entry '{}' not found: {}", xml_entry, e))?
        .read_to_end(&mut xml_bytes)
        .map_err(|e| e.to_string())?;

    let xml_str = String::from_utf8_lossy(&xml_bytes);
    let mut reader = quick_xml::Reader::from_str(&xml_str);
    reader.config_mut().trim_text(true);

    let mut text = String::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Text(e)) => {
                if let Ok(s) = e.decode() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&s);
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => return Err(e.to_string()),
            _ => {}
        }
        buf.clear();
    }

    Ok(text)
}
