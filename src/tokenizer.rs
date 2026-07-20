//! Lightweight tokenizer decoder.
//!
//! Loads a HuggingFace `tokenizer.json` file to extract the vocabulary
//! for token ID → text decoding.  Supports byte-level BPE tokenizers
//! (GPT-2, LLaMA family) and simple word-level vocabularies.
//!
//! Encoding (text → ids) is NOT implemented — the model's own
//! tokenizer (Python HuggingFace or future Rust tokenizer) handles
//! that at request time.  This module only decodes.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

// ── Tokenizer JSON structure (subset) ────────────────────────────────────────

#[derive(Deserialize)]
struct TokenizerJson {
    model: Option<ModelSection>,
    #[serde(rename = "added_tokens")]
    added_tokens: Option<Vec<AddedToken>>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(rename = "type")]
    _type: Option<String>,
    vocab: Option<HashMap<String, u32>>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

/// A lightweight token ID → text decoder backed by a HuggingFace
/// `tokenizer.json` vocabulary.
pub struct Tokenizer {
    /// id → token string
    id_to_token: Vec<Option<String>>,
    /// Whether this is a byte-level BPE tokenizer (GPT-2 style)
    is_byte_level: bool,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file path.
    /// Also accepts a directory containing `tokenizer.json`.
    pub fn load(path: &str) -> Result<Self, String> {
        let p = Path::new(path);
        let file_path = if p.is_dir() {
            p.join("tokenizer.json")
        } else {
            p.to_path_buf()
        };

        let data = fs::read_to_string(&file_path)
            .map_err(|e| format!("cannot read {:?}: {e}", file_path))?;
        let parsed: TokenizerJson =
            serde_json::from_str(&data).map_err(|e| format!("invalid tokenizer.json: {e}"))?;

        let vocab = parsed
            .model
            .as_ref()
            .and_then(|m| m.vocab.as_ref())
            .ok_or_else(|| "tokenizer.json missing model.vocab".to_string())?;

        // Determine if byte-level BPE (GPT-2 family uses byte-level)
        let model_type = parsed
            .model
            .as_ref()
            .and_then(|m| m._type.as_deref())
            .unwrap_or("");
        let is_byte_level = model_type == "BPE";
        let is_gpt2 = vocab.values().any(|&id| id == 0) && vocab.contains_key("!");

        // Build reverse mapping
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token: Vec<Option<String>> = vec![None; max_id + 1];
        for (token_str, &id) in vocab {
            let idx = id as usize;
            if idx < id_to_token.len() {
                id_to_token[idx] = Some(token_str.clone());
            }
        }

        // Add any added_tokens that might be beyond the vocab range
        if let Some(ref added) = parsed.added_tokens {
            for t in added {
                let idx = t.id as usize;
                if idx >= id_to_token.len() {
                    id_to_token.resize(idx + 1, None);
                }
                id_to_token[idx] = Some(t.content.clone());
            }
        }

        Ok(Self {
            id_to_token,
            is_byte_level: is_byte_level || is_gpt2,
        })
    }

    /// Decode a single token (incremental — called per decode step).
    pub fn decode_single(&self, token_id: i64) -> Option<String> {
        if token_id < 0 {
            return None;
        }
        let id = token_id as usize;
        self.id_to_token
            .get(id)
            .and_then(|opt| opt.as_ref())
            .map(|s| self.token_to_text(s))
    }

    /// Decode a sequence of token IDs.
    pub fn decode(&self, token_ids: &[i64]) -> String {
        if self.is_byte_level {
            self.decode_byte_level(token_ids)
        } else {
            self.decode_simple(token_ids)
        }
    }

    /// Simple concatenation of token strings (word-level tokenizers).
    fn decode_simple(&self, token_ids: &[i64]) -> String {
        let mut parts = Vec::new();
        for &id in token_ids {
            if let Some(s) = self.decode_single(id) {
                parts.push(s);
            }
        }
        // Join with space if token doesn't start with punctuation
        let mut out = String::new();
        for part in parts {
            if !out.is_empty() && !part.starts_with(|c: char| c.is_ascii_punctuation()) {
                out.push(' ');
            }
            out.push_str(&part);
        }
        out
    }

    /// Byte-level BPE decode: convert token bytes to UTF-8 text.
    /// GPT-2 / LLaMA style: token strings are byte representations.
    fn decode_byte_level(&self, token_ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if let Some(raw) = self.decode_single_raw(id) {
                bytes.extend_from_slice(&raw);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Get the raw byte representation of a token.
    /// For byte-level BPE tokens, interprets the token string as UTF-8
    /// where each Unicode codepoint represents a byte value (0-255).
    fn decode_single_raw(&self, token_id: i64) -> Option<Vec<u8>> {
        if token_id < 0 {
            return None;
        }
        let id = token_id as usize;
        let token_str = self.id_to_token.get(id).and_then(|opt| opt.as_ref())?;
        Some(byte_level_decode(token_str))
    }

    /// Convert a byte-level token string to bytes.
    /// Characters 0-255 map directly to bytes; "Ġ" (U+0120) maps to space.
    fn token_to_text(&self, token: &str) -> String {
        if !self.is_byte_level {
            return token.to_string();
        }
        let bytes = byte_level_decode(token);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

/// Decode a byte-level BPE token string to raw bytes.
///
/// GPT-2 tokenizer encodes bytes 0-255 as characters (including
/// unprintable ones), plus uses 'Ġ' (U+0120) to represent spaces.
fn byte_level_decode(token: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(token.len());
    for ch in token.chars() {
        let code = ch as u32;
        if code == 0x0120 {
            // 'Ġ' (U+0120) → space (byte 32)
            bytes.push(32u8);
        } else if code < 256 {
            // Bytes 0–255 map directly to single-byte values.
            // This is the common case for GPT-2 / LLaMA byte-level BPE.
            bytes.push(code as u8);
        } else {
            // Multi-byte Unicode character (code > 255).
            // For strictly byte-level BPE this shouldn't appear in the
            // vocabulary, but handle it gracefully by emitting the full
            // UTF-8 encoding rather than silently truncating to one byte.
            let mut buf = [0u8; 4];
            let len = ch.encode_utf8(&mut buf).len();
            bytes.extend_from_slice(&buf[..len]);
        }
    }
    bytes
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_decode_space() {
        // 'Ġ' (U+0120) is a space prefix in GPT-2 BPE tokens
        let bytes = byte_level_decode("Ġhello");
        assert_eq!(bytes, vec![32, 104, 101, 108, 108, 111]); // ' hello'
    }

    #[test]
    fn byte_level_decode_ascii() {
        let bytes = byte_level_decode("Hello");
        assert_eq!(bytes, vec![72, 101, 108, 108, 111]); // 'Hello'
    }

    #[test]
    fn decode_single_negative() {
        let t = Tokenizer {
            id_to_token: vec![Some("test".into())],
            is_byte_level: false,
        };
        assert_eq!(t.decode_single(-1), None);
        assert_eq!(t.decode_single(0), Some("test".into()));
        assert_eq!(t.decode_single(1), None); // out of range
    }

    #[test]
    fn decode_simple_word_level() {
        let t = Tokenizer {
            id_to_token: vec![Some("Hello".into()), Some("world".into()), Some("!".into())],
            is_byte_level: false,
        };
        assert_eq!(t.decode(&[0, 1, 2]), "Hello world!");
    }

    #[test]
    fn decode_byte_level() {
        // Build a minimal byte-level tokenizer
        // 'H'=72, 'e'=101, 'l'=108, 'o'=111, 'Ġ'=space
        let mut id_to_token: Vec<Option<String>> = vec![None; 256];
        // Map byte 'H' → token "H"
        for b in 0u8..=255u8 {
            id_to_token[b as usize] = Some(String::from_utf8(vec![b]).unwrap_or_default());
        }
        let t = Tokenizer {
            id_to_token,
            is_byte_level: true,
        };
        // "Ġworld" is space (32) + 'w', 'o', 'r', 'l', 'd'
        let tokens = &[72_i64, 101, 108, 108, 111]; // H, e, l, l, o
        assert_eq!(t.decode(tokens), "Hello");
    }

    #[test]
    fn byte_level_decode_multi_byte_unicode() {
        // Chinese character '中' (U+4E2D) encoded as UTF-8: E4 B8 AD
        // If it somehow appears in a byte-level vocab, it should be preserved
        // as the full UTF-8 byte sequence, not truncated to a single byte.
        let bytes = byte_level_decode("中");
        assert_eq!(
            bytes,
            vec![0xE4, 0xB8, 0xAD],
            "multi-byte char should emit full UTF-8 sequence"
        );

        // Emoji '🚀' (U+1F680) — 4-byte UTF-8: F0 9F 9A 80
        let bytes = byte_level_decode("🚀");
        assert_eq!(
            bytes,
            vec![0xF0, 0x9F, 0x9A, 0x80],
            "4-byte char should emit full UTF-8 sequence"
        );
    }
}
