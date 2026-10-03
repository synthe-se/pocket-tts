//! Turning a text prompt into what the model reads, mirroring upstream's
//! `pocket_tts/models/text_chunking.py`: the per-config text rules
//! (`prepare_text_prompt`) and the token-boundary helpers the splitter in
//! `tts_model.rs` builds on.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::config::Config;

/// The per-config text rules applied to every prompt before tokenization.
#[derive(Debug, Clone)]
pub struct TextRules {
    /// Prepend 8 spaces to very short inputs (< 5 words). The English
    /// checkpoints rely on this; the multilingual ones do not.
    pub pad_with_spaces_for_short_inputs: bool,
    /// Replace `;` with `,` (multilingual checkpoints).
    pub remove_semicolons: bool,
    /// End the prompt with sentence-final punctuation (#296).
    pub append_terminal_punctuation: bool,
    /// Upper-case the first letter; off for phoneme models.
    pub capitalize_first_letter: bool,
    /// Per-character rewrites ("" deletes) for characters the training text
    /// never contained.
    pub replace_characters: HashMap<char, String>,
}

impl Default for TextRules {
    /// Upstream's `prepare_text_prompt` defaults.
    fn default() -> Self {
        Self {
            pad_with_spaces_for_short_inputs: false,
            remove_semicolons: false,
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            replace_characters: HashMap::new(),
        }
    }
}

impl TextRules {
    /// The rules a config asks for. Fails on a `replace_characters` key that
    /// is not a single character, which Python's `str.maketrans` refuses too.
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let mut replace_characters = HashMap::new();
        for (from, to) in &config.replace_characters {
            let mut chars = from.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => {
                    replace_characters.insert(c, to.clone());
                }
                _ => {
                    anyhow::bail!("replace_characters keys must be single characters, got {from:?}")
                }
            }
        }
        Ok(Self {
            pad_with_spaces_for_short_inputs: config.pad_with_spaces_for_short_inputs,
            remove_semicolons: config.remove_semicolons,
            append_terminal_punctuation: config.append_terminal_punctuation,
            capitalize_first_letter: config.capitalize_first_letter,
            replace_characters,
        })
    }
}

/// Deleted quotes leave `"Hi?", she said` as `Hi?, she said`, which reads as
/// a sentence end followed by a stray comma; keep the sentence mark only.
static STRAY_MARK_AFTER_SENTENCE_END: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"([.!?\x{2026}])\s*[,;:]").expect("valid regex"));

/// Prepare text for generation (upstream `prepare_text_prompt`), after
/// stripping this port's explicit `[pause:...]` markers.
pub fn prepare_text_prompt(text: &str, rules: &TextRules) -> String {
    let text = crate::pause::strip_pause_markers(text);
    let mut text = text.trim().to_string();

    if !rules.replace_characters.is_empty() {
        let mut replaced = String::with_capacity(text.len());
        for c in text.chars() {
            match rules.replace_characters.get(&c) {
                Some(to) => replaced.push_str(to),
                None => replaced.push(c),
            }
        }
        let collapsed = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
        text = STRAY_MARK_AFTER_SENTENCE_END
            .replace_all(&collapsed, "$1")
            .into_owned();
    }

    if text.is_empty() {
        // Upstream raises; the port keeps generating a minimal prompt.
        return ".".to_string();
    }

    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    if rules.remove_semicolons {
        text = text.replace(';', ",");
    }

    let word_count = text.split_whitespace().count();

    // Only meaningful for orthographies that have case.
    if rules.capitalize_first_letter
        && let Some(first) = text.chars().next()
        && !first.is_uppercase()
    {
        text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
    }

    if rules.append_terminal_punctuation {
        text = ensure_terminal_punctuation(&text);
    }

    // The model does not perform well with very few tokens, so short inputs
    // get leading spaces to raise the token count (English checkpoints).
    if rules.pad_with_spaces_for_short_inputs && word_count < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }

    text
}

const TERMINAL_PUNCTUATION: &[char] = &['.', '!', '?', '\u{2026}'];
const WEAK_PUNCTUATION: &[char] = &[',', ';', ':', '-', '\u{2013}', '\u{2014}'];
const CLOSERS: &[char] = &['"', '\'', '\u{201d}', '\u{2019}', ')', ']', '\u{bb}'];

/// Make sure the prompt ends with sentence-final punctuation (upstream
/// `_ensure_terminal_punctuation`). The model is trained on sentences ending
/// with a period, question or exclamation mark; without one the last word is
/// often mispronounced or repeated. Text already ending with one (possibly
/// followed by closing quotes or brackets) is left alone, a trailing comma,
/// colon or dash becomes a period, and anything else gets a period appended
/// after the closing quote or bracket.
pub fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c: char| CLOSERS.contains(&c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(c) if TERMINAL_PUNCTUATION.contains(&c) => text.to_string(),
        Some(c) if WEAK_PUNCTUATION.contains(&c) => {
            let stem = core.trim_end_matches(|c: char| WEAK_PUNCTUATION.contains(&c) || c == ' ');
            format!("{stem}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// Find token indices where text should be split on boundary tokens
/// (upstream `_find_boundary_indices`): each consecutive pair delimits one
/// segment, the first is always 0 and the last `tokens.len()`. `skip` vetoes
/// a split right before a given index (the decimal-period check).
pub fn find_boundary_indices(
    tokens: &[u32],
    boundary_tokens: &[u32],
    mut skip: impl FnMut(usize) -> bool,
) -> Vec<usize> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, token) in tokens.iter().enumerate() {
        if boundary_tokens.contains(token) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary && !skip(idx) {
                indices.push(idx);
            }
            previous_was_boundary = false;
        }
    }
    indices.push(tokens.len());
    indices
}

/// Whether a segment starting right after `prefix` (decoded tokens before
/// the split) and reading `suffix` sits inside a decimal number like "3.5"
/// (upstream `_is_decimal_period_boundary`, #217).
pub fn is_decimal_period_boundary(prefix: &str, suffix: &str) -> bool {
    let mut tail = prefix.chars().rev();
    matches!(
        (tail.next(), tail.next(), suffix.chars().next()),
        (Some('.'), Some(d), Some(s)) if d.is_numeric() && s.is_numeric()
    )
}

/// Frames to keep generating after EOS when neither the caller nor the
/// config says: upstream's word-count guess plus its fixed 2.
pub fn estimate_frames_after_eos(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    if word_count <= 4 {
        3 + 2 // prepare_text_prompt guess + 2
    } else {
        1 + 2 // prepare_text_prompt guess + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(pad: bool, semicolons: bool) -> TextRules {
        TextRules {
            pad_with_spaces_for_short_inputs: pad,
            remove_semicolons: semicolons,
            ..TextRules::default()
        }
    }

    #[test]
    fn test_prepare_text_prompt() {
        // Short texts (<5 words) get 8 spaces prepended (padding enabled)
        assert_eq!(
            prepare_text_prompt("hello world", &rules(true, false)),
            "        Hello world."
        );
        assert_eq!(
            prepare_text_prompt("Hello world.", &rules(true, false)),
            "        Hello world."
        );
        assert_eq!(
            prepare_text_prompt("  hello  ", &rules(true, false)),
            "        Hello."
        );
        // Long texts don't get spaces
        assert_eq!(
            prepare_text_prompt("one two three four five", &rules(true, false)),
            "One two three four five."
        );
        // Padding disabled: short text is not padded
        assert_eq!(
            prepare_text_prompt("hello world", &rules(false, false)),
            "Hello world."
        );
        // remove_semicolons: ';' becomes ','
        assert_eq!(
            prepare_text_prompt("one two; three four five", &rules(false, true)),
            "One two, three four five."
        );
    }

    #[test]
    fn test_prepare_text_prompt_strips_pause_markers() {
        let result = prepare_text_prompt("Hello [pause:500ms] world", &rules(true, false));
        assert!(!result.contains("[pause:"));
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_prepare_text_prompt_handles_multiple_pauses() {
        let result = prepare_text_prompt(
            "One [pause:100ms] two [pause:1s] three",
            &rules(true, false),
        );
        assert!(!result.contains("[pause:"));
        assert!(result.contains("One"));
        assert!(result.contains("two"));
        assert!(result.contains("three"));
    }

    #[test]
    fn test_capitalize_first_letter_can_be_disabled() {
        let r = TextRules {
            capitalize_first_letter: false,
            ..TextRules::default()
        };
        assert_eq!(prepare_text_prompt("salAm dunya", &r), "salAm dunya.");
    }

    #[test]
    fn test_append_terminal_punctuation_can_be_disabled() {
        let r = TextRules {
            append_terminal_punctuation: false,
            ..TextRules::default()
        };
        assert_eq!(prepare_text_prompt("namaste duniya", &r), "Namaste duniya");
    }

    #[test]
    fn test_replace_characters() {
        // The French config's table, in part.
        let mut map = HashMap::new();
        for c in ['"', '\u{201c}', '\u{201d}', '\u{ab}', '\u{bb}', '(', ')'] {
            map.insert(c, String::new());
        }
        map.insert('\u{2019}', "'".to_string());
        map.insert(':', ",".to_string());
        let r = TextRules {
            replace_characters: map,
            ..TextRules::default()
        };
        assert_eq!(
            prepare_text_prompt("\u{ab} Bonjour ! \u{bb} dit-elle", &r),
            "Bonjour ! dit-elle."
        );
        assert_eq!(
            prepare_text_prompt("\"Hi?\", she said", &r),
            "Hi? she said."
        );
        assert_eq!(
            prepare_text_prompt("l\u{2019}heure : midi (environ)", &r),
            "L'heure , midi environ."
        );
        // Deleting everything leaves the minimal prompt.
        assert_eq!(prepare_text_prompt("\"\"", &r), ".");
    }

    #[test]
    fn test_terminal_punctuation_is_added_when_missing() {
        // Upstream tests/test_split_sentences.py cases.
        let cases = [
            ("hello world", "Hello world."),
            ("it costs 42", "It costs 42."),
            ("hello world!", "Hello world!"),
            ("wait for it...", "Wait for it..."),
            ("hello world,", "Hello world."),
            ("hello world:", "Hello world."),
            ("hello world -", "Hello world."),
            ("he said \"go home\"", "He said \"go home\"."),
            ("she whispered 'run'", "She whispered 'run'."),
            ("he said \"go home.\"", "He said \"go home.\""),
            ("he said \"go home,\"", "He said \"go home.\""),
            ("is it (really)?", "Is it (really)?"),
            ("see the note (below)", "See the note (below)."),
            ("up by 50%", "Up by 50%."),
        ];
        for (text, expected) in cases {
            assert_eq!(prepare_text_prompt(text, &rules(false, false)), expected);
        }
    }

    #[test]
    fn test_replace_characters_upstream_cases() {
        let drop: HashMap<char, String> = "\"\u{a1}\u{bf}\u{ab}\u{bb}"
            .chars()
            .map(|c| (c, String::new()))
            .collect();
        let with = |map: HashMap<char, String>| TextRules {
            replace_characters: map,
            ..TextRules::default()
        };
        assert_eq!(
            prepare_text_prompt(
                "\"\u{a1}Venid a m\u{ed}, hombres!\" Alz\u{f3} la voz.",
                &with(drop.clone())
            ),
            "Venid a m\u{ed}, hombres! Alz\u{f3} la voz."
        );
        let mut fr = drop.clone();
        fr.insert('\u{2019}', "'".to_string());
        assert_eq!(
            prepare_text_prompt("il a dit \u{ab} l\u{2019}homme \u{bb}", &with(fr)),
            "Il a dit l'homme."
        );
        assert_eq!(
            prepare_text_prompt("\"Vieni stasera?\", chiese.", &with(drop)),
            "Vieni stasera? chiese."
        );
        // Empty by default: configs that don't set it are unchanged.
        assert_eq!(
            prepare_text_prompt("\"Yes,\" she said.", &TextRules::default()),
            "\"Yes,\" she said."
        );
    }

    #[test]
    fn test_find_boundary_indices() {
        // tokens: a . b . . c   boundaries: '.' = 9
        let tokens = [1, 9, 2, 9, 9, 3];
        assert_eq!(
            find_boundary_indices(&tokens, &[9], |_| false),
            vec![0, 2, 5, 6]
        );
        assert_eq!(
            find_boundary_indices(&tokens, &[9], |i| i == 2),
            vec![0, 5, 6]
        );
    }

    #[test]
    fn test_is_decimal_period_boundary() {
        assert!(is_decimal_period_boundary("It costs 3.", "5 euros."));
        assert!(!is_decimal_period_boundary("It costs 3.", " Five more."));
        assert!(!is_decimal_period_boundary("End.", "5 euros."));
        assert!(!is_decimal_period_boundary(".", "5"));
    }

    #[test]
    fn test_estimate_frames_after_eos() {
        assert_eq!(estimate_frames_after_eos("Hello world"), 5);
        assert_eq!(estimate_frames_after_eos("One two three four five"), 3);
    }
}
