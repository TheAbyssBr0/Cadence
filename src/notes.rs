//! Post-grading chapter-notes synthesis (§11): prompts, validation, identity.
//!
//! After grading, one synthesis call produces organic markdown notes that read
//! like a strong student's own chapter notes — content-derived `## ` sections,
//! foundational ideas first, never a report about the study process. The
//! misconception and grading evidence steers *emphasis only*: stumbled-on
//! concepts get extra space with the correction stated crisply as the obvious
//! thing to write down; demonstrated topics get a confident line or two. The
//! notes never mention mistakes, answers, scores, questions, takeaway numbers,
//! or lifecycle statuses — the validator rejects such meta-language so it
//! fast-retries with direction instead of persisting.
//!
//! The engine is pure: callers map store rows onto [`MisconceptionItem`] and
//! [`GradeSummary`] (the engine never touches the store), and persist the
//! validated result with [`to_new_note`]. Like every other generation stage,
//! notes output is validated before it is cached. Content binding (does the
//! synthesis actually reflect the grades?) stays in the prompt: misconception
//! concepts are paraphrased freely by the model, so verbatim matching would
//! false-reject good notes.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

use crate::engines::UnitText;
use crate::error::{Error, Result};
use crate::store::NewNote;

/// Completion-token cap for notes synthesis (one six-section markdown doc).
pub const NOTES_MAX_TOKENS: u32 = 8_000;
/// Minimum accepted markdown length: an organic synthesis below this is a
/// stub or a truncation, never a storable artifact.
pub const MIN_NOTES_CHARS: usize = 200;
/// Minimum content-derived `## ` sections: thinner notes are an outline, not
/// notes someone could re-read and learn from.
pub const MIN_NOTES_SECTIONS: usize = 3;

/// Meta-language that must never appear in notes (case-insensitive): words
/// about the study process rather than the chapter. The validator rejects
/// them so the fast-retry loop repairs toward the organic voice instead of
/// persisting a grading report.
const BANNED_PHRASES: [&str; 16] = [
    "misconception",
    "takeaway",
    "active vs",
    "core mental models",
    "most important lessons",
    "suggested mental models",
    "mistakes from assignment",
    "things demonstrated",
    "your assignment",
    "you answered",
    "you wrote",
    "your answer",
    "model solution",
    "partially_correct",
    "correct_but_brief",
    "question_defective",
];

/// One tracked misconception feeding the synthesis (§12 input): concept plus
/// the evidence behind it and its lifecycle status. `status` is `ACTIVE` for
/// open rows and `RESOLVED` for cleared ones — the prompt puts each side of
/// the §11 item-3 section accordingly, and explicitly notes assignment-time
/// resolutions as cleared, never still emphasized as unresolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisconceptionItem {
    /// Short concept label.
    pub concept: String,
    /// User's wrong answer / explanation behind it.
    pub evidence: String,
    /// Lifecycle status (`ACTIVE` | `RESOLVED`, ...).
    pub status: String,
}

/// One graded answer feeding the synthesis (§7.3/§10 input): the frozen
/// verdict plus the feedback that names what went wrong or right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradeSummary {
    /// Joined question parts.
    pub question: String,
    /// Points awarded.
    pub score: i64,
    /// Rubric total.
    pub max_score: i64,
    /// §10 verdict label.
    pub classification: String,
    /// Technical feedback for the student.
    pub feedback: String,
}

/// Validated chapter notes, ready to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedNotes {
    /// Full personalized markdown (§11, six sections).
    pub markdown: String,
}

/// Build the notes synthesis prompt: chapter text plus an emphasis guide
/// distilled from the misconception and grading evidence. The voice is a
/// strong student's own re-readable notes — content-derived `## ` headings,
/// foundational ideas first, corrections woven in as the natural thing to
/// write down. The evidence below is steering only: never mention, quote, or
/// allude to it.
#[must_use]
pub fn build_notes_prompt(
    unit: &UnitText,
    misconceptions: &[MisconceptionItem],
    grades: &[GradeSummary],
) -> String {
    let mut prompt = format!(
        "You are writing study notes for a chapter, in the voice of a strong student writing notes they will actually re-read — not a report about studying. Base every claim strictly on the chapter text below — never introduce facts not present in or directly inferable from it.\nWrite the notes as markdown starting with `# {heading} — Notes`, followed by 3-6 `## ` sections with content-derived titles (name them after the chapter's ideas, e.g. `## Truth values`, never after the study process). Prioritize foundational, high-leverage concepts — do not treat all facts equally; keep every section substantive (the whole document is substantial, never a stub).\nVoice rules (all hard requirements): write about the chapter, never about the student, the assignment, or any grading. Never use the words misconception, takeaway, assignment, grade, or score; never cite takeaway numbers, question numbers, verdict labels, or lifecycle statuses — state each rule directly in your own words (write `Prefer \\`if (i)\\` over \\`if (i != 0)\\``, never `Takeaway 3.3 says ...`). Never quote or paraphrase the emphasis guide below; never write `you`, `your answer`, or `evidence:`. A reader must not be able to tell any assessment happened.\nRespond with a single JSON object and nothing else: {{\"notes_markdown\": \"# {heading} — Notes\\n\\n## ...\\n...\"}}. Emit raw JSON only (no markdown fences, no commentary); escape newlines inside strings as \\n.\n\n--- CHAPTER TEXT (heading: {heading}, pages {start}-{end}) ---\n{text}\n--- END CHAPTER TEXT ---",
        heading = unit.heading,
        start = unit.page_start,
        end = unit.page_end,
        text = unit.text,
    );
    if misconceptions.is_empty() && grades.is_empty() {
        prompt.push_str("\n\nEMPHASIS GUIDE: none — no assessment evidence for this chapter. Write straight chapter notes with even, content-driven emphasis.");
    } else {
        prompt.push_str("\n\nEMPHASIS GUIDE (steering only — never mention, quote, or allude to any of this; it must be invisible in the notes):\nLinger on these topics — give each extra space and state the correction crisply as if it were the obvious thing to write down:\n");
        for item in misconceptions {
            let _ = writeln!(prompt, "- {} ({})", item.concept, item.status);
        }
        for grade in grades {
            let _ = writeln!(
                prompt,
                "- [{} {}/{}] {} — {}",
                grade.classification, grade.score, grade.max_score, grade.question, grade.feedback
            );
        }
        prompt.push_str("Cover everything else at natural weight: topics handled well get a confident line or two, not whole sections.");
    }
    prompt
}

/// Canonical params JSON for notes calls (token cap + operation identity +
/// response-schema tag).
#[must_use]
pub fn notes_params_json() -> String {
    let tag = crate::llm::schema_tag(&notes_response_schema());
    format!("{{\"max_tokens\":{NOTES_MAX_TOKENS},\"operation\":\"notes\",\"rf\":\"{tag}\"}}")
}

/// JSON Schema for the notes response (constrained decoding): a single
/// `notes_markdown` string. Section presence and substance stay in the
/// validator: schemas enforce shape, never judgment.
#[must_use]
pub fn notes_response_schema() -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "notes_markdown": {"type": "string"},
        },
        "required": ["notes_markdown"],
        "additionalProperties": false,
    })
    .to_string()
}

/// Hex SHA-256 over the (unit text, misconceptions, grades) triple: the
/// identity half that keeps notes cache entries apart when any input changes.
/// The cache identity never sees prompt text (§16), so every varying input
/// must be bound here.
#[must_use]
pub fn notes_source_hash_for(
    unit_text: &str,
    misconceptions: &[MisconceptionItem],
    grades: &[GradeSummary],
) -> String {
    let mut mis_repr = String::new();
    for item in misconceptions {
        mis_repr.push_str(&item.concept);
        mis_repr.push('\0');
        mis_repr.push_str(&item.evidence);
        mis_repr.push('\0');
        mis_repr.push_str(&item.status);
        mis_repr.push('\0');
    }
    let mut grades_repr = String::new();
    for grade in grades {
        grades_repr.push_str(&grade.question);
        grades_repr.push('\0');
        grades_repr.push_str(&grade.score.to_string());
        grades_repr.push('\0');
        grades_repr.push_str(&grade.max_score.to_string());
        grades_repr.push('\0');
        grades_repr.push_str(&grade.classification);
        grades_repr.push('\0');
        grades_repr.push_str(&grade.feedback);
        grades_repr.push('\0');
    }
    let mut hasher = Sha256::new();
    for part in [unit_text, &mis_repr, &grades_repr] {
        hasher.update(part.as_bytes());
        hasher.update([0_u8]);
    }
    hex::encode(hasher.finalize())
}

/// Validate one notes response: raw JSON object with a substantial organic
/// `notes_markdown` — at least [`MIN_NOTES_SECTIONS`] content-derived `## `
/// headings, no study-process meta-language.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect (fed back into the
/// fast-retry repair loop by `complete_cached`).
pub fn validate_notes(text: &str) -> Result<ValidatedNotes> {
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::LlmFatal(format!("notes is not valid JSON: {e}")))?;
    let markdown = parsed
        .get("notes_markdown")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal("notes missing 'notes_markdown'".to_string()))?;
    if markdown.chars().count() < MIN_NOTES_CHARS {
        return Err(Error::LlmFatal(format!(
            "notes is a stub ({} chars, need {MIN_NOTES_CHARS}+)",
            markdown.chars().count()
        )));
    }
    let sections = markdown
        .lines()
        .filter(|line| line.trim_start().starts_with("## "))
        .count();
    if sections < MIN_NOTES_SECTIONS {
        return Err(Error::LlmFatal(format!(
            "notes has {sections} `## ` section(s), need at least {MIN_NOTES_SECTIONS} content-derived sections"
        )));
    }
    // Organic voice: notes about the study process read as a grading report,
    // not re-readable notes. Banned phrases plus question/score references
    // (`(Q1)`, `5/10`) are the tells.
    let lowered = markdown.to_ascii_lowercase();
    let mut violations = Vec::new();
    for banned in BANNED_PHRASES {
        if lowered.contains(banned) {
            violations.push((*banned).to_string());
        }
    }
    for q in 1..=9 {
        if lowered.contains(&format!("(q{q}")) {
            violations.push(format!("question reference (q{q})"));
        }
    }
    if lowered.contains("/10") {
        violations.push("score reference (/10)".to_string());
    }
    if !violations.is_empty() {
        return Err(Error::LlmFatal(format!(
            "notes reads as a grading report, not chapter notes — remove the study-process meta-language ({}) and state each rule directly",
            violations.join(", ")
        )));
    }
    Ok(ValidatedNotes {
        markdown: markdown.to_string(),
    })
}

/// Serialize validated notes into their store row (§17).
#[must_use]
pub fn to_new_note(
    chapter_id: i64,
    attempt_no: i64,
    validated: &ValidatedNotes,
    generated_at: &str,
) -> NewNote {
    NewNote {
        chapter_id,
        content_markdown: validated.markdown.clone(),
        generated_at: generated_at.to_string(),
        attempt_no,
    }
}

/// Parse a `--misconceptions` file for `dev notes`: a JSON array of
/// `{"concept": ..., "evidence"?: ..., "status"?: ...}` objects. Status
/// defaults to `ACTIVE`; evidence defaults to empty.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the file is not an array of
/// concept-carrying objects.
pub fn parse_misconceptions_file(text: &str) -> Result<Vec<MisconceptionItem>> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::InvalidInput(format!("misconceptions file is not valid JSON: {e}")))?;
    let raw = value.as_array().ok_or_else(|| {
        Error::InvalidInput("misconceptions file must be a JSON array".to_string())
    })?;
    let mut out = Vec::with_capacity(raw.len());
    for (index, entry) in raw.iter().enumerate() {
        let concept = entry
            .get("concept")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::InvalidInput(format!("misconceptions[{index}] needs a concept"))
            })?;
        let evidence = entry
            .get("evidence")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let status = entry
            .get("status")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("ACTIVE");
        out.push(MisconceptionItem {
            concept: concept.to_string(),
            evidence: evidence.to_string(),
            status: status.to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit() -> UnitText {
        UnitText {
            text: "Pointers hold addresses. The & operator takes an address. Dereference with *."
                .to_string(),
            page_start: 10,
            page_end: 20,
            heading: "Pointers".to_string(),
        }
    }

    fn misconceptions() -> Vec<MisconceptionItem> {
        vec![
            MisconceptionItem {
                concept: "address-of yields value".to_string(),
                evidence: "answered that &x is the value".to_string(),
                status: "ACTIVE".to_string(),
            },
            MisconceptionItem {
                concept: "deref is free".to_string(),
                evidence: "skipped null checks".to_string(),
                status: "RESOLVED".to_string(),
            },
        ]
    }

    fn grades() -> Vec<GradeSummary> {
        vec![GradeSummary {
            question: "a) Explain what &x yields.".to_string(),
            score: 4,
            max_score: 5,
            classification: "CORRECT_BUT_BRIEF".to_string(),
            feedback: "Correct core model; say more next time.".to_string(),
        }]
    }

    fn good_markdown() -> String {
        String::from(
            "# Pointers — Notes\n\n## Addresses and dereference\nBody text with enough substance to read as real notes on the chapter material, written by a student for re-reading. Foundational ideas first.\n\n## Null checks without comparison\nPrefer `if (p)` over `if (p != NULL)`: a null pointer is false, any live address is true, so the comparison adds noise.\n\n## Aliasing traps\nTwo pointers can name the same cell; writes through one are visible through the other. Reason about cells, not names.\n\n",
        )
    }

    fn good_response() -> String {
        serde_json::json!({"notes_markdown": good_markdown()}).to_string()
    }

    #[test]
    fn prompt_carries_evidence_and_headings() {
        let prompt = build_notes_prompt(&unit(), &misconceptions(), &grades());
        assert!(prompt.contains("# Pointers — Notes"));
        assert!(prompt.contains("content-derived titles"));
        assert!(prompt.contains("address-of yields value"));
        assert!(prompt.contains("EMPHASIS GUIDE"));
        assert!(prompt.contains("steering only"));
        assert!(prompt.contains("CORRECT_BUT_BRIEF"));
        assert!(prompt.contains("Correct core model"));
        assert!(prompt.contains("Pointers"));
        // The student's wrong-answer text is steering input, never quoted
        // into the prompt where the model could echo it back.
        assert!(!prompt.contains("answered that &x is the value"));
    }

    #[test]
    fn prompt_handles_empty_inputs() {
        let prompt = build_notes_prompt(&unit(), &[], &[]);
        assert!(prompt.contains("no assessment evidence"));
        assert!(prompt.contains("content-derived titles"));
    }

    #[test]
    fn prompt_guides_on_partial_evidence() {
        // Either evidence kind alone steers emphasis (the "none" branch
        // needs both empty).
        let prompt = build_notes_prompt(&unit(), &misconceptions(), &[]);
        assert!(prompt.contains("Linger on these topics"));
        assert!(!prompt.contains("no assessment evidence"));
        let prompt = build_notes_prompt(&unit(), &[], &grades());
        assert!(prompt.contains("Linger on these topics"));
        assert!(!prompt.contains("no assessment evidence"));
    }

    #[test]
    fn prompt_forbids_meta_language() {
        let prompt = build_notes_prompt(&unit(), &misconceptions(), &grades());
        assert!(prompt.contains("never cite takeaway numbers"));
        assert!(prompt.contains("must not be able to tell any assessment happened"));
    }

    #[test]
    fn schema_is_closed_shape() {
        let schema: serde_json::Value = serde_json::from_str(&notes_response_schema()).unwrap();
        assert_eq!(schema["required"], serde_json::json!(["notes_markdown"]));
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn params_bind_operation_and_schema() {
        let params = notes_params_json();
        assert!(params.contains("\"operation\":\"notes\""));
        assert!(params.contains(&NOTES_MAX_TOKENS.to_string()));
        assert!(params.contains(&crate::llm::schema_tag(&notes_response_schema())));
    }

    #[test]
    fn valid_notes_pass() {
        let validated = validate_notes(&good_response()).unwrap();
        assert!(
            validated
                .markdown
                .contains("## Null checks without comparison")
        );
        assert!(validated.markdown.chars().count() >= MIN_NOTES_CHARS);
    }

    #[test]
    fn too_few_sections_fails() {
        let doc = "# Pointers — Notes\n\n## Only section\nSubstantive body text for this section with enough words to pass the length floor on its own merit. ".to_string()
            + &"Padding to clear the stub floor. ".repeat(20);
        let response = serde_json::json!({"notes_markdown": doc}).to_string();
        let err = validate_notes(&response).unwrap_err();
        assert!(err.to_string().contains("section(s)"));
    }

    #[test]
    fn meta_language_fails_with_repair_direction() {
        // A grading-report-shaped draft: banned phrases, a question
        // reference, and a score — padded past the length floor so the
        // failure is the voice, not the size.
        let mut doc = String::from(
            "# Chapter notes\n\n## Truth values\nGenuine chapter content about truth values with enough substance to stand alone. ",
        );
        doc.push_str("\n\n## Misconceptions\nYour assignment Q1 (q1) showed a misconception worth 5/10. Takeaway 3.3 covers it. ");
        doc.push_str("\n\n## Things Demonstrated\nYou answered well; the model solution agrees. ");
        doc.push_str(&"Padding to clear the stub floor. ".repeat(20));
        let response = serde_json::json!({"notes_markdown": doc}).to_string();
        let err = validate_notes(&response).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("grading report"), "got: {message}");
        assert!(message.contains("misconception"), "got: {message}");
    }

    #[test]
    fn wrong_heading_level_fails() {
        let mut doc = String::from("# Pointers — Notes\n\n");
        for title in ["Addresses", "Null checks", "Aliasing"] {
            doc.push_str("### ");
            doc.push_str(title);
            doc.push_str("\nSubstantive body text for this section with enough words to read as real notes.\n\n");
        }
        doc.push_str(&"Padding to clear the stub floor. ".repeat(20));
        let response = serde_json::json!({"notes_markdown": doc}).to_string();
        assert!(validate_notes(&response).is_err());
    }

    #[test]
    fn boundary_length_accepted() {
        // Exactly MIN_NOTES_CHARS with enough sections is substantial.
        let header = "## Alpha\n## Beta\n## Gamma\n";
        let filler: String = "Chapter substance. "
            .chars()
            .cycle()
            .take(MIN_NOTES_CHARS - header.chars().count())
            .collect();
        let markdown = format!("{header}{filler}");
        assert_eq!(markdown.chars().count(), MIN_NOTES_CHARS);
        let response = serde_json::json!({"notes_markdown": markdown}).to_string();
        assert!(validate_notes(&response).is_ok());
    }

    #[test]
    fn stub_and_shape_failures() {
        assert!(validate_notes("not json").is_err());
        assert!(validate_notes(r#"{"other": 1}"#).is_err());
        assert!(validate_notes(r#"{"notes_markdown": "  "}"#).is_err());
        let short =
            serde_json::json!({"notes_markdown": "## Core Mental Models\ntiny"}).to_string();
        let err = validate_notes(&short).unwrap_err();
        assert!(err.to_string().contains("stub"));
    }

    #[test]
    fn source_hash_binds_every_input() {
        let base = notes_source_hash_for(unit().text.as_str(), &misconceptions(), &grades());
        assert_eq!(
            base,
            notes_source_hash_for(unit().text.as_str(), &misconceptions(), &grades())
        );
        assert_ne!(
            base,
            notes_source_hash_for("other text", &misconceptions(), &grades())
        );
        let mut changed_mis = misconceptions();
        changed_mis[0].status = "RESOLVED".to_string();
        assert_ne!(
            base,
            notes_source_hash_for(unit().text.as_str(), &changed_mis, &grades())
        );
        let mut changed_grades = grades();
        changed_grades[0].score = 1;
        assert_ne!(
            base,
            notes_source_hash_for(unit().text.as_str(), &misconceptions(), &changed_grades)
        );
        assert_ne!(
            base,
            notes_source_hash_for(unit().text.as_str(), &[], &grades())
        );
    }

    #[test]
    fn to_new_note_maps_fields() {
        let validated = validate_notes(&good_response()).unwrap();
        let row = to_new_note(9, 2, &validated, "2026-09-25");
        assert_eq!(row.chapter_id, 9);
        assert_eq!(row.attempt_no, 2);
        assert_eq!(row.generated_at, "2026-09-25");
        assert_eq!(row.content_markdown, validated.markdown);
    }

    #[test]
    fn misconceptions_file_parsing() {
        let items = parse_misconceptions_file(
            r#"[{"concept": "aliasing", "evidence": "missed it", "status": "RESOLVED"}, {"concept": "scales"}]"#,
        )
        .unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].status, "RESOLVED");
        assert_eq!(items[0].evidence, "missed it");
        assert_eq!(items[1].status, "ACTIVE");
        assert_eq!(items[1].evidence, String::new());
        assert!(parse_misconceptions_file(r#"{"concept": "x"}"#).is_err());
        assert!(parse_misconceptions_file(r#"[{"evidence": "no concept"}]"#).is_err());
        assert!(parse_misconceptions_file("broken").is_err());
        assert_eq!(parse_misconceptions_file("[]").unwrap().len(), 0);
    }
}
