pub struct Chunk {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Split `text` into overlapping word-based windows, then enforce `max_chunk_chars`.
///
/// Any word-window that exceeds `max_chunk_chars` characters is split further at
/// character boundaries so that no chunk passed to the embedder is pathologically
/// large (e.g. minified JSON where a single "word" is hundreds of KB).
///
/// Set `max_chunk_chars = 0` to skip the character cap.
pub fn chunk_text(text: &str, words_per_chunk: usize, overlap_words: usize, max_chunk_chars: usize) -> Vec<Chunk> {
    let word_chunks = chunk_by_words(text, words_per_chunk, overlap_words);

    if max_chunk_chars == 0 {
        return word_chunks;
    }

    let mut result = Vec::new();
    for chunk in word_chunks {
        if chunk.text.len() <= max_chunk_chars {
            result.push(chunk);
        } else {
            result.extend(split_by_chars(chunk, max_chunk_chars));
        }
    }
    result
}

fn chunk_by_words(text: &str, words_per_chunk: usize, overlap_words: usize) -> Vec<Chunk> {
    let mut positions: Vec<(usize, usize)> = Vec::new();
    let mut in_word = false;
    let mut word_start = 0;

    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if in_word {
                positions.push((word_start, i));
                in_word = false;
            }
        } else if !in_word {
            word_start = i;
            in_word = true;
        }
    }
    if in_word {
        positions.push((word_start, text.len()));
    }

    if positions.is_empty() {
        return Vec::new();
    }

    let step = words_per_chunk.saturating_sub(overlap_words).max(1);
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < positions.len() {
        let last = (i + words_per_chunk - 1).min(positions.len() - 1);
        let start = positions[i].0;
        let end = positions[last].1;

        chunks.push(Chunk {
            text: text[start..end].to_string(),
            start,
            end,
        });

        if last == positions.len() - 1 {
            break;
        }
        i += step;
    }

    chunks
}

fn split_by_chars(chunk: Chunk, max_chars: usize) -> Vec<Chunk> {
    let text = &chunk.text;
    let mut result = Vec::new();
    let mut seg_start = 0usize;
    let mut count = 0usize;

    for (byte_idx, _) in text.char_indices() {
        if count > 0 && count % max_chars == 0 {
            result.push(Chunk {
                text: text[seg_start..byte_idx].to_string(),
                start: chunk.start + seg_start,
                end: chunk.start + byte_idx,
            });
            seg_start = byte_idx;
        }
        count += 1;
    }
    if seg_start < text.len() {
        result.push(Chunk {
            text: text[seg_start..].to_string(),
            start: chunk.start + seg_start,
            end: chunk.end,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_chunk() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 4, 1, 0);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(!c.text.is_empty());
            assert!(c.start <= c.end);
            assert_eq!(&text[c.start..c.end], c.text);
        }
    }

    #[test]
    fn char_cap_splits_large_chunks() {
        // Simulate a "word" that is 200 chars long — exceeds cap of 50.
        let text = "a".repeat(200);
        let chunks = chunk_text(&text, 1, 0, 50);
        assert_eq!(chunks.len(), 4);
        for c in &chunks {
            assert!(c.text.len() <= 50);
            assert_eq!(&text[c.start..c.end], c.text);
        }
    }

    #[test]
    fn char_cap_disabled() {
        let text = "a".repeat(200);
        let chunks = chunk_text(&text, 1, 0, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text.len(), 200);
    }
}
