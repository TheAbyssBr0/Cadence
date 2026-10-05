//! `dev mcq` session wiring (§2.1 / §7.1): pure helpers behind the interactive
//! loop in `main`.
//!
//! The loop itself lives in `main` (terminal I/O); everything testable lives
//! here: answer parsing, per-question shuffle seeds, `McqItem` ↔
//! `ValidatedMcq` conversion, `NewMcqItem` serialization, dev-chapter
//! find-or-create (so `--persist --seed S` resumes), and misconception text.

use crate::domain::Chapter;
use crate::domain::ChapterStatus;
use crate::engines::UnitText;
use crate::error::{Error, Result};
use crate::mcq::{DISPLAYED_OPTION_COUNT, IDK_LABEL, McqPhase, ValidatedMcq};
use crate::store::{NewBook, NewChapter, NewMcqItem, Store};

/// Questions generated per `dev mcq` run (§7.1 minimum of the 8–12 range).
pub const DEV_MCQ_COUNT: usize = 8;
/// Completion-token cap for MCQ generation calls (full 8-Q JSON must fit).
pub const DEV_MCQ_MAX_TOKENS: u32 = 8_000;
/// Attempt number used by every `dev mcq` session (isolated harness, §2.1).
pub const DEV_ATTEMPT_NO: i64 = 1;

/// Parse an interactive answer (`A`–`E`, case-insensitive, trimmed).
/// Returns the displayed index (0–4, where 4 is `"I don't know"`). Derived
/// from [`option_label`] over [`DISPLAYED_OPTION_COUNT`] so the accepted
/// letters can never drift from the rendered labels.
#[must_use]
pub fn parse_answer(input: &str) -> Option<usize> {
    let normalized = input.trim().to_lowercase();
    (0..DISPLAYED_OPTION_COUNT).find(|index| option_label(*index).eq_ignore_ascii_case(&normalized))
}

/// Per-question shuffle seed: the session base (from `--seed` or the clock)
/// plus the zero-based display position. Deterministic across resume runs
/// with the same seed because item order is stable (insertion order).
#[must_use]
pub fn shuffle_seed_for(base_seed: u64, position: usize) -> u64 {
    let position_u64 = u64::try_from(position).unwrap_or(0);
    base_seed.wrapping_add(position_u64)
}

/// Render labels: `A`–`D` plus hardcoded `E = "I don't know"`.
#[must_use]
pub const fn option_label(displayed_index: usize) -> &'static str {
    match displayed_index {
        0 => "A",
        1 => "B",
        2 => "C",
        3 => "D",
        _ => "E",
    }
}

/// Serialize validated LLM output into storable rows (pre-shuffle indices).
///
/// # Errors
///
/// Returns [`Error::Io`] when option/source-ref serialization fails
/// (practically unreachable: strings always serialize).
pub fn to_new_items(
    chapter_id: i64,
    phase: McqPhase,
    validated: &[ValidatedMcq],
) -> Result<Vec<NewMcqItem>> {
    to_new_items_for(chapter_id, phase, validated, DEV_ATTEMPT_NO)
}

/// Serialize validated LLM output for an explicit attempt number
/// (production chapters store one set per `attempt_no`, §4.1).
///
/// # Errors
///
/// Returns [`Error::Io`] when option/source-ref serialization fails
/// (practically unreachable: strings always serialize).
pub fn to_new_items_for(
    chapter_id: i64,
    phase: McqPhase,
    validated: &[ValidatedMcq],
    attempt_no: i64,
) -> Result<Vec<NewMcqItem>> {
    let mut out = Vec::with_capacity(validated.len());
    for item in validated {
        let options_json =
            serde_json::to_string(&item.options).map_err(|e| Error::Io(e.to_string()))?;
        let source_refs = serde_json::to_string(&serde_json::json!({
            "pages": item.source_refs.pages,
            "sections": item.source_refs.sections,
        }))
        .map_err(|e| Error::Io(e.to_string()))?;
        let (Some(correct), Some(trap)) = (
            i64::try_from(item.correct_index).ok(),
            i64::try_from(item.trap_index).ok(),
        ) else {
            return Err(Error::Io("MCQ index overflow".to_string()));
        };
        out.push(NewMcqItem {
            chapter_id,
            phase: phase.as_str().to_string(),
            question_text: item.question.clone(),
            options_json,
            correct_index: correct,
            trap_index: trap,
            explanation_text: item.explanation.clone(),
            source_refs,
            topic: item.topic.clone(),
            attempt_no,
        });
    }
    Ok(out)
}

/// Borrowed view of one stored MCQ row for [`validated_from_row`]: the seven
/// display fields travel together, so they take one parameter, not seven.
pub struct McqRowView<'a> {
    /// Question stem.
    pub question: &'a str,
    /// JSON array of pre-shuffle option strings.
    pub options_json: &'a str,
    /// Correct-option index pre-shuffle.
    pub correct_index: i64,
    /// Trap-option index pre-shuffle.
    pub trap_index: i64,
    /// 2–5 sentence explanation.
    pub explanation: &'a str,
    /// Topic label.
    pub topic: &'a str,
    /// JSON source references (`pages`, `sections`).
    pub source_refs: &'a str,
}

/// Rebuild a validated item from its stored row so the display shuffle can be
/// recomputed deterministically on resume.
///
/// # Errors
///
/// Returns [`Error::Store`] when the row's JSON or indices are corrupt.
pub fn validated_from_row(row: &McqRowView<'_>, unit: &UnitText) -> Result<ValidatedMcq> {
    let options: Vec<String> =
        serde_json::from_str(row.options_json).map_err(|e| Error::Store(e.to_string()))?;
    let refs: serde_json::Value =
        serde_json::from_str(row.source_refs).map_err(|e| Error::Store(e.to_string()))?;
    let pages: Vec<i64> = refs
        .get("pages")
        .and_then(serde_json::Value::as_array)
        .map_or_else(Vec::new, |arr| {
            arr.iter().filter_map(serde_json::Value::as_i64).collect()
        });
    let sections: Vec<String> = refs
        .get("sections")
        .and_then(serde_json::Value::as_array)
        .map_or_else(Vec::new, |arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        });
    let (Some(correct), Some(trap)) = (
        usize::try_from(row.correct_index).ok(),
        usize::try_from(row.trap_index).ok(),
    ) else {
        return Err(Error::Store("stored MCQ index out of range".to_string()));
    };
    let item = ValidatedMcq {
        question: row.question.to_string(),
        options,
        correct_index: correct,
        trap_index: trap,
        explanation: row.explanation.to_string(),
        topic: row.topic.to_string(),
        source_refs: crate::mcq::SourceRefs { pages, sections },
    };
    // Reuse the shape gate (minus count): a corrupt row fails loudly rather
    // than displaying garbage.
    let _ = crate::mcq::classify_pages(&item.source_refs.pages, unit);
    Ok(item)
}

/// Find-or-create the dev chapter so `--persist --seed S` resumes:
/// reuse book 1 when its hash matches this PDF, else create a fresh book;
/// reuse the first chapter with matching pages when present, else create one.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
pub fn ensure_dev_chapter(
    store: &mut dyn Store,
    pdf_hash: &str,
    pdf_name: &str,
    unit: &UnitText,
) -> Result<Chapter> {
    if let Ok(book) = store.get_book(1)
        && book.file_hash == pdf_hash
    {
        let chapters = store.list_chapters(book.id)?;
        let mut first = None;
        let mut chapter_count: usize = 0;
        for chapter in chapters {
            chapter_count = chapter_count.saturating_add(1);
            if chapter.start_page == unit.page_start && chapter.end_page == unit.page_end {
                return Ok(chapter);
            }
            if first.is_none() {
                first = Some(chapter);
            }
        }
        if let Some(chapter) = first {
            return Ok(chapter);
        }
        return store.create_chapter(&NewChapter {
            book_id: book.id,
            index_in_book: i64::try_from(chapter_count).unwrap_or(0),
            level: 1,
            title: unit.heading.clone(),
            start_page: unit.page_start,
            end_page: unit.page_end,
            file_path: format!("dev:{pdf_name}"),
            status: ChapterStatus::PretestReady,
        });
    }
    let book = store.create_book(
        &NewBook {
            title: format!("dev-mcq:{pdf_name}"),
            filepath: pdf_name.to_string(),
            file_hash: pdf_hash.to_string(),
            start_page: unit.page_start,
        },
        "2026-09-20",
    )?;
    store.create_chapter(&NewChapter {
        book_id: book.id,
        index_in_book: 0,
        level: 1,
        title: unit.heading.clone(),
        start_page: unit.page_start,
        end_page: unit.page_end,
        file_path: format!("dev:{pdf_name}"),
        status: ChapterStatus::PretestReady,
    })
}

/// Build misconception concept/description/evidence for a wrong retest answer.
/// Callers enforce [`crate::mcq::should_log_misconception`]; this only formats
/// text. Wrong non-trap answers log the *selected* answer as evidence, never
/// the designated trap's claim.
#[must_use]
pub fn misconception_texts(
    topic: &str,
    question: &str,
    selected_text: &str,
    correct_text: &str,
    selected_trap: bool,
) -> (String, String, String) {
    let concept = topic.to_string();
    let description = format!("Retest misconception on '{topic}': {question}");
    let kind = if selected_trap {
        "trap selected"
    } else {
        "non-trap wrong answer"
    };
    let evidence =
        format!("Selected '{selected_text}' ({kind}) instead of '{correct_text}' for: {question}");
    (concept, description, evidence)
}

/// Whether the harness has recorded any answer for this item (resume gate).
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
pub fn is_answered(store: &dyn Store, mcq_item_id: i64) -> Result<bool> {
    Ok(!store.list_mcq_responses(mcq_item_id)?.is_empty())
}

/// Label for the hardcoded fifth option.
pub const fn idk_label() -> &'static str {
    IDK_LABEL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcq::IDK_INDEX;

    fn unit() -> UnitText {
        UnitText {
            text: "Pointers hold addresses.".to_string(),
            page_start: 10,
            page_end: 20,
            heading: "Pointers".to_string(),
        }
    }

    #[test]
    fn answers_parse_case_insensitively() {
        assert_eq!(parse_answer("A"), Some(0));
        assert_eq!(parse_answer("  d "), Some(3));
        assert_eq!(parse_answer("e"), Some(4));
        assert_eq!(parse_answer("E"), Some(4));
        assert_eq!(parse_answer("F"), None);
        assert_eq!(parse_answer(""), None);
        assert_eq!(parse_answer("ab"), None);
    }

    #[test]
    fn shuffle_seeds_are_stable_and_position_sensitive() {
        assert_eq!(shuffle_seed_for(42, 0), 42);
        assert_eq!(shuffle_seed_for(42, 1), 43);
        assert_ne!(shuffle_seed_for(42, 0), shuffle_seed_for(43, 0));
    }

    #[test]
    fn option_labels_cover_display() {
        assert_eq!(
            (0..DISPLAYED_OPTION_COUNT)
                .map(option_label)
                .collect::<Vec<_>>(),
            vec!["A", "B", "C", "D", "E"]
        );
        assert_eq!(idk_label(), "I don't know");
        assert_eq!(IDK_INDEX, DISPLAYED_OPTION_COUNT.saturating_sub(1));
    }

    #[test]
    fn round_trip_row_serialization() {
        let validated = vec![ValidatedMcq {
            question: "What does &x yield?".to_string(),
            options: vec![
                "The address of x".to_string(),
                "The value of x".to_string(),
                "A null pointer".to_string(),
                "A dangling reference".to_string(),
            ],
            correct_index: 0,
            trap_index: 1,
            explanation: "The & operator takes the address of its operand, plainly.".to_string(),
            topic: "addresses".to_string(),
            source_refs: crate::mcq::SourceRefs {
                pages: vec![12],
                sections: vec!["Addresses".to_string()],
            },
        }];
        let phase = McqPhase::Pretest;
        let rows = to_new_items(7, phase, &validated).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].phase, "pretest");
        assert_eq!(rows[0].attempt_no, DEV_ATTEMPT_NO);
        let back = validated_from_row(
            &McqRowView {
                question: &rows[0].question_text,
                options_json: &rows[0].options_json,
                correct_index: rows[0].correct_index,
                trap_index: rows[0].trap_index,
                explanation: &rows[0].explanation_text,
                topic: &rows[0].topic,
                source_refs: &rows[0].source_refs,
            },
            &unit(),
        )
        .unwrap();
        assert_eq!(back, validated[0]);
    }

    #[test]
    fn items_for_carries_explicit_attempt() {
        let validated = vec![ValidatedMcq {
            question: "What does &x yield?".to_string(),
            options: vec![
                "The address of x".to_string(),
                "The value of x".to_string(),
                "A null pointer".to_string(),
                "A dangling reference".to_string(),
            ],
            correct_index: 0,
            trap_index: 1,
            explanation: "The & operator takes the address of its operand, plainly.".to_string(),
            topic: "addresses".to_string(),
            source_refs: crate::mcq::SourceRefs {
                pages: vec![12],
                sections: vec!["Addresses".to_string()],
            },
        }];
        let rows = to_new_items_for(7, McqPhase::Pretest, &validated, 3).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].attempt_no, 3);
        assert_eq!(rows[0].phase, "pretest");
    }

    #[test]
    fn corrupt_row_fails_loudly() {
        let err = validated_from_row(
            &McqRowView {
                question: "q",
                options_json: "not-json",
                correct_index: 0,
                trap_index: 1,
                explanation: "explanation with enough length here",
                topic: "t",
                source_refs: "{}",
            },
            &unit(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn answered_tracks_recorded_responses() {
        let mut store = crate::store::MemoryStore::new();
        let chapter = ensure_dev_chapter(&mut store, "hash1", "ch.pdf", &unit()).unwrap();
        let saved = store
            .save_mcq_items(&[crate::store::NewMcqItem {
                chapter_id: chapter.id,
                phase: "pretest".to_string(),
                question_text: "What does &x yield?".to_string(),
                options_json: "[\"a\", \"b\", \"c\", \"d\"]".to_string(),
                correct_index: 0,
                trap_index: 1,
                explanation_text: "It yields the address.".to_string(),
                source_refs: "{}".to_string(),
                topic: "addresses".to_string(),
                attempt_no: 1,
            }])
            .unwrap();
        assert!(!is_answered(&store, saved[0].id).unwrap());
        store
            .record_mcq_response(saved[0].id, 0, true, false, "2026-09-20", 1)
            .unwrap();
        assert!(is_answered(&store, saved[0].id).unwrap());
        assert!(!is_answered(&store, saved[0].id + 99).unwrap());
    }

    #[test]
    fn dev_chapter_reuses_matching_pages() {
        let mut store = crate::store::MemoryStore::new();
        let u = unit();
        let first = ensure_dev_chapter(&mut store, "hash1", "ch.pdf", &u).unwrap();
        let second = ensure_dev_chapter(&mut store, "hash1", "ch.pdf", &u).unwrap();
        assert_eq!(first.id, second.id);
        // A different unit under the same PDF reuses the first chapter rather
        // than fragmenting the dev DB (single-chapter fixture).
        let mut other = u.clone();
        other.page_start = 21;
        other.page_end = 30;
        let third = ensure_dev_chapter(&mut store, "hash1", "ch.pdf", &other).unwrap();
        assert_eq!(third.id, first.id);
        // A different PDF starts a fresh book.
        let fourth = ensure_dev_chapter(&mut store, "hash2", "other.pdf", &u).unwrap();
        assert_ne!(fourth.id, first.id);
    }

    #[test]
    fn misconception_evidence_names_selected_answer() {
        let (concept, description, evidence) = misconception_texts(
            "addresses",
            "What does &x yield?",
            "The value of x",
            "The address of x",
            false,
        );
        assert_eq!(concept, "addresses");
        assert!(description.contains("What does &x yield?"));
        assert!(evidence.contains("The value of x"));
        assert!(evidence.contains("non-trap"));
        let (_, _, trap_evidence) = misconception_texts(
            "addresses",
            "What does &x yield?",
            "The value of x",
            "The address of x",
            true,
        );
        assert!(trap_evidence.contains("trap selected"));
    }

    #[test]
    fn generation_isolates_params_identity() {
        let plain = crate::mcq::mcq_params_json(DEV_MCQ_COUNT, McqPhase::Pretest);
        let gen_a = crate::mcq::mcq_params_json_with_generation(
            DEV_MCQ_COUNT,
            McqPhase::Pretest,
            Some("exp-a"),
        );
        let gen_b = crate::mcq::mcq_params_json_with_generation(
            DEV_MCQ_COUNT,
            McqPhase::Pretest,
            Some("exp-b"),
        );
        assert_ne!(plain, gen_a);
        assert_ne!(gen_a, gen_b);
        assert!(gen_a.contains("exp-a"));
        assert!(plain.contains("\"max_tokens\":8000"));
    }
}
