//! MCQ engine (§7.1 / §8.1): prompts, validation, shuffle, source gating.
//!
//! Pipeline: concepts → draft → verify → validate. The LLM drafts question
//! JSON; this module validates shape (4 options, 1 correct, 1 trap,
//! explanation, source refs), gates source membership (every cited page must
//! lie inside the unit's one-based range), and shuffles the 4 generated
//! options app-side before display. `E = "I don't know"` is hardcoded at
//! render time and never accepted from the model.
//!
//! Engines never touch the network: callers run the prompt through
//! [`crate::llm::complete_cached`] with [`validate_mcq_set`] as the
//! validator (malformed responses fast-retry there, never cached).

use sha2::{Digest, Sha256};

use crate::domain::TaskType;
use crate::engines::{SupportVerdict, UnitText};
use crate::error::{Error, Result};

/// Generated options per question (A–D); E is hardcoded `"I don't know"`.
pub const MCQ_OPTION_COUNT: usize = 4;
/// Displayed options (A–E).
pub const DISPLAYED_OPTION_COUNT: usize = 5;
/// Displayed index of the hardcoded `"I don't know"` option.
pub const IDK_INDEX: usize = 4;
/// Hardcoded fifth option: always incorrect, never LLM-generated.
pub const IDK_LABEL: &str = "I don't know";
/// Spec §7.1: 8–12 questions per set.
pub const MIN_QUESTIONS: usize = 8;
/// Spec §7.1: 8–12 questions per set.
pub const MAX_QUESTIONS: usize = 12;

/// Assessment phase: pretest (moderate) vs retest (maximum difficulty) vs
/// review (targeted re-probes of open misconceptions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McqPhase {
    /// Day N: definitions, core concepts, mechanics, prerequisites.
    Pretest,
    /// Day N+1: multi-step reasoning, synthesis, edge cases, transfer.
    Retest,
    /// Manual `cadence review`: maximum-difficulty questions re-probing open
    /// misconceptions on completed chapters (§12). No scheduler task executes
    /// it; storage, cache, and lifecycle reuse the retest machinery.
    Review,
}

impl McqPhase {
    /// Canonical label used in prompts, cache identity, and storage.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pretest => "pretest",
            Self::Retest => "retest",
            Self::Review => "review",
        }
    }

    /// Parse a phase label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "pretest" => Ok(Self::Pretest),
            "retest" => Ok(Self::Retest),
            "review" => Ok(Self::Review),
            other => Err(Error::InvalidInput(format!("unknown MCQ phase: {other}"))),
        }
    }

    /// Scheduler task types that execute as MCQ sessions (§7.1): pretest and
    /// retest. Reading, assignment, and notes run through their own stages;
    /// review is manual (§12) and never scheduler-driven.
    #[must_use]
    pub const fn for_task(task: TaskType) -> Option<Self> {
        match task {
            TaskType::Pretest => Some(Self::Pretest),
            TaskType::Retest => Some(Self::Retest),
            TaskType::Read | TaskType::AssignmentWrite | TaskType::Notes => None,
        }
    }
}

/// Source references attached to every question (§18 provenance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRefs {
    /// One-based physical pages supporting the question.
    pub pages: Vec<i64>,
    /// Section headings supporting the question.
    pub sections: Vec<String>,
}

/// One validated MCQ (pre-shuffle indices into `options`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedMcq {
    /// Question stem.
    pub question: String,
    /// Exactly 4 generated options (pre-shuffle).
    pub options: Vec<String>,
    /// Correct option pre-shuffle (0–3).
    pub correct_index: usize,
    /// Designated trap pre-shuffle (0–3, never the correct one).
    pub trap_index: usize,
    /// 2–5 sentence explanation.
    pub explanation: String,
    /// Topic label.
    pub topic: String,
    /// Provenance (pages inside the unit range).
    pub source_refs: SourceRefs,
}

/// One MCQ ready to display: 4 shuffled options plus hardcoded E.
/// `displayed_correct` / `displayed_trap` are post-shuffle (0–3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayedMcq {
    /// Question stem.
    pub question: String,
    /// 4 shuffled options (E is appended at render time).
    pub displayed_options: Vec<String>,
    /// Correct option post-shuffle (0–3).
    pub displayed_correct: usize,
    /// Trap option post-shuffle (0–3).
    pub displayed_trap: usize,
    /// Explanation (shown after answering).
    pub explanation: String,
    /// Topic label.
    pub topic: String,
}

/// Hex SHA-256 of chapter text: the `source_hash` half of the LLM cache
/// identity for MCQ calls.
#[must_use]
pub fn source_hash_for(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// Canonical params JSON for MCQ calls (count + phase cap the identity).
#[must_use]
pub fn mcq_params_json(count: usize, phase: McqPhase) -> String {
    mcq_params_json_with_generation(count, phase, None)
}

/// Canonical params JSON with an optional fresh-experiment `generation` label:
/// same seed + different generation = isolated cache identity (§16).
#[must_use]
pub fn mcq_params_json_with_generation(
    count: usize,
    phase: McqPhase,
    generation: Option<&str>,
) -> String {
    let generation_part = generation.map_or_else(
        || "null".to_string(),
        |g| {
            let escaped = g.replace('\\', "\\\\").replace('"', "\\\"");
            format!("\"{escaped}\"")
        },
    );
    let tag = crate::llm::schema_tag(&mcq_response_schema());
    format!(
        "{{\"max_tokens\":8000,\"mcq_count\":{count},\"phase\":\"{}\",\"generation\":{generation_part},\"rf\":\"{tag}\"}}",
        phase.as_str()
    )
}

/// JSON Schema for the MCQ set response (constrained decoding): a `questions`
/// array of 8–12 items mirroring [`validate_mcq_set_ranged`] structurally —
/// 4 options, 0–3 indices, non-empty provenance arrays. Semantic checks
/// (distinct options, trap≠correct, in-range pages, self-containment) stay in
/// the validator: schemas enforce shape, never judgment.
#[must_use]
pub fn mcq_response_schema() -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": MIN_QUESTIONS,
                "maxItems": MAX_QUESTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "minItems": MCQ_OPTION_COUNT,
                            "maxItems": MCQ_OPTION_COUNT,
                            "items": {"type": "string"},
                        },
                        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                        "trap_index": {"type": "integer", "minimum": 0, "maximum": 3},
                        "explanation": {"type": "string"},
                        "topic": {"type": "string"},
                        "source_refs": {
                            "type": "object",
                            "properties": {
                                "pages": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "integer"},
                                },
                                "sections": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["pages", "sections"],
                            "additionalProperties": false,
                        },
                    },
                    "required": ["question", "options", "correct_index", "trap_index", "explanation", "topic", "source_refs"],
                    "additionalProperties": false,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": false,
    })
    .to_string()
}

/// Shared `book > model_prior_knowledge` preamble (§18, verbatim intent).
const SOURCE_FIDELITY: &str = "Base every question strictly on the chapter text provided. Do not introduce facts, syntax, or claims not present in or directly inferable from this text. If uncertain, do not use it. If the source material is ambiguous, reject the question rather than manufacturing certainty.";

/// Shared JSON schema instruction.
const SCHEMA_INSTRUCTION: &str = "Respond with a single JSON object and nothing else: {\"questions\": [{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"correct_index\": 0, \"trap_index\": 1, \"explanation\": \"...\", \"topic\": \"...\", \"source_refs\": {\"pages\": [12], \"sections\": [\"...\"]}}]}. Rules: exactly 4 options (1 correct, exactly 1 designated trap as a plausible misconception, 2 plausible distractors); never include \"I don't know\" among the 4 (it is added by the application); correct_index and trap_index must differ; every question needs a 2-5 sentence explanation and source_refs with at least one page inside the unit range and at least one section name.";

/// Self-containment rule: the test is closed-book, so every question stem plus
/// its 4 options must be answerable by someone who knows the chapter's
/// concepts but does NOT have the book open. `source_refs` stay mandatory but
/// are internal provenance metadata — never mention them (or any
/// listing/page/line pointer) in the stem or options.
const SELF_CONTAINMENT: &str = "SELF-CONTAINED QUESTIONS (closed-book test): the student answers without the book open, so each question stem plus its 4 options must stand alone. NEVER refer to listings, figures, tables, sections, exercises, examples, page numbers, or line numbers (e.g. 'in Listing 1.2', 'line 22', 'the example above'); never write 'above'/'below'/'earlier' pointing at book content. If a question needs code, output, or quoted text, inline the complete minimal excerpt directly in the stem. `source_refs` are internal provenance only and must never appear in stems or options. Explanations may cite where to re-read (they are shown after answering).";

fn prompt_common(unit: &UnitText, count: usize, difficulty: &str) -> String {
    format!(
        "You are writing multiple-choice questions for a closed-book study system.\n{SOURCE_FIDELITY}\n{SELF_CONTAINMENT}\nDifficulty: {difficulty}\nWrite {count} questions on the chapter below (heading: {heading}, pages {start}-{end}).\n{SCHEMA_INSTRUCTION}\n\n--- CHAPTER TEXT (pages {start}-{end}) ---\n{text}\n--- END CHAPTER TEXT ---",
        heading = unit.heading,
        start = unit.page_start,
        end = unit.page_end,
        text = unit.text,
    )
}

/// Pretest prompt (moderate): definitions, core concepts, syntax/mechanics,
/// prerequisites (§7.1).
#[must_use]
pub fn build_pretest_prompt(unit: &UnitText, count: usize) -> String {
    prompt_common(
        unit,
        count,
        "moderate pretest (definitions, core concepts, syntax/mechanics, prerequisites)",
    )
}

/// Retest prompt (maximum difficulty): multi-step reasoning, synthesis, edge
/// cases, counterexamples, transfer, creative problem-solving; actively seek
/// applications to situations not directly stated in the text (§7.1).
#[must_use]
pub fn build_retest_prompt(unit: &UnitText, count: usize) -> String {
    prompt_common(
        unit,
        count,
        "MAXIMUM difficulty retest (multi-step reasoning, synthesis, edge cases, counterexamples, transfer, creative problem-solving; apply concepts to situations not directly stated in the text, never mere recall)",
    )
}

/// One open misconception a review question must re-probe (§12): the concept
/// label doubles as the required `topic` value so answers route back to the
/// row they probed; the description and evidence tell the generator which
/// wrong belief to rebuild as the trap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewConcept {
    /// Misconception row id (prompt context only).
    pub id: i64,
    /// Short concept label; emitted verbatim as the question `topic`.
    pub concept: String,
    /// Fuller description of the wrong belief.
    pub description: String,
    /// The user's wrong answer / explanation (trap material).
    pub evidence: String,
}

/// Review prompt (targeted retest, §12): maximum difficulty like a retest,
/// but every question re-probes exactly one listed misconception instead of
/// sampling the chapter. The generator rebuilds the documented wrong belief
/// as the trap and sets `topic` to the concept label verbatim — answers
/// match rows back by that label, so paraphrasing it breaks the lifecycle.
#[must_use]
pub fn build_review_prompt(unit: &UnitText, count: usize, concepts: &[ReviewConcept]) -> String {
    use std::fmt::Write as _;
    let mut listed = String::new();
    for target in concepts {
        let _ = writeln!(
            listed,
            "- [{}] concept: {}\n  belief: {}\n  evidence: {}",
            target.id, target.concept, target.description, target.evidence
        );
    }
    format!(
        "You are writing multiple-choice questions for a closed-book study system.\n{SOURCE_FIDELITY}\n{SELF_CONTAINMENT}\nDifficulty: MAXIMUM difficulty targeted retest (multi-step reasoning, synthesis, edge cases, transfer — the same bar as a retest, never mere recall).\nWrite {count} questions on the chapter below (heading: {heading}, pages {start}-{end}). Every question must re-probe exactly one misconception from the list below: aim the stem at the documented belief, rebuild that belief as the designated trap, and set the question `topic` to the concept label VERBATIM (character-for-character — the application matches answers back to rows on it). Cover every listed misconception at least once.\n{SCHEMA_INSTRUCTION}\n\n--- OPEN MISCONCEPTIONS ---\n{listed}--- END MISCONCEPTIONS ---\n\n--- CHAPTER TEXT (pages {start}-{end}) ---\n{text}\n--- END CHAPTER TEXT ---",
        heading = unit.heading,
        start = unit.page_start,
        end = unit.page_end,
        text = unit.text,
    )
}

/// Normalize an option for comparison (trim + lowercase).
fn normalize_option(text: &str) -> String {
    text.trim().to_lowercase()
}

/// Whether an option smuggles the hardcoded fifth choice.
fn is_idk_smuggle(text: &str) -> bool {
    let n = normalize_option(text);
    n == "i don't know" || n == "i dont know" || n == "e) i don't know"
}

/// Keywords that, followed by a number, point at book content instead of
/// stating it (`Listing 1.2`, `line 22`) — a closed-book violation in stems
/// and options. Includes book-specific box labels (`takeaway`, `challenge`:
/// only the numbered-pointer use matches, plain prose never does).
const DEICTIC_KEYWORDS: [&str; 10] = [
    "listing",
    "figure",
    "table",
    "section",
    "exercise",
    "example",
    "page",
    "line",
    "takeaway",
    "challenge",
];

/// Discourse phrases pointing at book content rather than inlining it.
/// Deliberately narrow: bare `above`/`below` stay legal (`values above zero`)
/// — only pointer phrases are rejected.
const DEICTIC_PHRASES: [&str; 14] = [
    "the above",
    "example above",
    "shown above",
    "described above",
    "discussed above",
    "mentioned above",
    "stated above",
    "given above",
    "see above",
    "the below",
    "shown below",
    "discussed earlier",
    "mentioned earlier",
    "as stated earlier",
];

/// Word constituent for boundary checks (ASCII-centric; prompts are English).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `keyword` occurs in `lower` as a whole word directly followed by a
/// number (after whitespace): e.g. `listing 1.2`, `line 22`. Substrings of
/// longer words (`inline`, `pipeline`) never match.
fn keyword_with_number(lower: &str, keyword: &str) -> bool {
    if keyword.is_empty() {
        return false;
    }
    let mut search_from = 0_usize;
    while let Some(slice) = lower.get(search_from..) {
        let Some(relative) = slice.find(keyword) else {
            return false;
        };
        let start = search_from.saturating_add(relative);
        let Some(end) = start.checked_add(keyword.len()) else {
            return false;
        };
        search_from = end;
        let before_ok = lower
            .get(..start)
            .and_then(|s| s.chars().next_back())
            .is_none_or(|c| !is_word_char(c));
        if !before_ok {
            continue;
        }
        let Some(after) = lower.get(end..) else {
            continue;
        };
        let mut chars = after.chars();
        let Some(first) = chars.next() else {
            continue;
        };
        if is_word_char(first) {
            continue;
        }
        if chars
            .find(|c| !c.is_whitespace())
            .is_some_and(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

/// Whether `phrase` occurs in `lower` with non-word characters on both sides.
fn contains_phrase(lower: &str, phrase: &str) -> bool {
    if phrase.is_empty() {
        return false;
    }
    let mut search_from = 0_usize;
    while let Some(slice) = lower.get(search_from..) {
        let Some(relative) = slice.find(phrase) else {
            return false;
        };
        let start = search_from.saturating_add(relative);
        let Some(end) = start.checked_add(phrase.len()) else {
            return false;
        };
        search_from = end;
        let before_ok = lower
            .get(..start)
            .and_then(|s| s.chars().next_back())
            .is_none_or(|c| !is_word_char(c));
        let after_ok = lower
            .get(end..)
            .and_then(|s| s.chars().next())
            .is_none_or(|c| !is_word_char(c));
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Inspect display text (stem or option) for closed-book violations: pointers
/// at book content (`Listing 1.2`, `line 22`, `the example above`) instead of
/// inlined material. Returns a human-readable reason for the first violation,
/// or `None` when the text stands alone. Explanations are intentionally
/// unchecked (post-answer re-read pointers are useful there).
#[must_use]
pub fn deictic_violation(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    for keyword in DEICTIC_KEYWORDS {
        if keyword_with_number(&lower, keyword) {
            return Some(format!(
                "forbidden reference to '{keyword} <number>' (closed-book: inline the material instead)"
            ));
        }
    }
    for phrase in DEICTIC_PHRASES {
        if contains_phrase(&lower, phrase) {
            return Some(format!(
                "forbidden reference '{phrase}' (closed-book: inline the material instead)"
            ));
        }
    }
    None
}

/// Validate the 4 generated options (non-empty, no `IDK` smuggle, distinct).
fn validate_options(options_value: &[serde_json::Value], ctx: &str) -> Result<Vec<String>> {
    if options_value.len() != MCQ_OPTION_COUNT {
        return Err(Error::LlmFatal(format!(
            "{ctx}: 'options' must have exactly {MCQ_OPTION_COUNT} entries"
        )));
    }
    let mut options = Vec::with_capacity(MCQ_OPTION_COUNT);
    for (position, raw) in options_value.iter().enumerate() {
        let text = raw
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::LlmFatal(format!("{ctx}: option {position} must be non-empty"))
            })?;
        if is_idk_smuggle(text) {
            return Err(Error::LlmFatal(format!(
                "{ctx}: option {position} must not supply \"I don't know\" (added by the app)"
            )));
        }
        if let Some(reason) = deictic_violation(text) {
            return Err(Error::LlmFatal(format!(
                "{ctx}: option {position}: {reason}"
            )));
        }
        options.push(text.to_string());
    }
    // Options must be pairwise distinct (case-insensitive).
    for (first, text_a) in options.iter().enumerate() {
        for (second, text_b) in options.iter().enumerate() {
            if second <= first {
                continue;
            }
            if normalize_option(text_a) == normalize_option(text_b) {
                return Err(Error::LlmFatal(format!(
                    "{ctx}: options {first} and {second} are duplicates"
                )));
            }
        }
    }
    Ok(options)
}

/// Validate `correct_index` / `trap_index` (both 0–3, differing).
fn validate_indices(value: &serde_json::Value, ctx: &str) -> Result<(usize, usize)> {
    let correct_index = value
        .get("correct_index")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: 'correct_index' must be 0-3")))?;
    let trap_index = value
        .get("trap_index")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: 'trap_index' must be 0-3")))?;
    if correct_index >= MCQ_OPTION_COUNT || trap_index >= MCQ_OPTION_COUNT {
        return Err(Error::LlmFatal(format!(
            "{ctx}: correct/trap indices must be 0-3"
        )));
    }
    if correct_index == trap_index {
        return Err(Error::LlmFatal(format!(
            "{ctx}: trap must differ from the correct option"
        )));
    }
    Ok((correct_index, trap_index))
}

/// Validate `source_refs` (pages inside the unit range, sections named).
fn validate_refs(value: &serde_json::Value, ctx: &str, unit: &UnitText) -> Result<SourceRefs> {
    let refs = value.get("source_refs").ok_or_else(|| {
        Error::LlmFatal(format!("{ctx}: missing 'source_refs' (pages + sections)"))
    })?;
    let pages_value = refs
        .get("pages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: 'source_refs.pages' must be an array")))?;
    if pages_value.is_empty() {
        return Err(Error::LlmFatal(format!(
            "{ctx}: 'source_refs.pages' must cite at least one page"
        )));
    }
    let mut pages = Vec::with_capacity(pages_value.len());
    for raw in pages_value {
        let page = raw
            .as_i64()
            .ok_or_else(|| Error::LlmFatal(format!("{ctx}: source pages must be integers")))?;
        if page < unit.page_start || page > unit.page_end {
            return Err(Error::LlmFatal(format!(
                "{ctx}: source page {page} outside unit range {}-{}",
                unit.page_start, unit.page_end
            )));
        }
        pages.push(page);
    }
    let sections_value = refs
        .get("sections")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Error::LlmFatal(format!("{ctx}: 'source_refs.sections' must be an array"))
        })?;
    if sections_value.is_empty() {
        return Err(Error::LlmFatal(format!(
            "{ctx}: 'source_refs.sections' must name at least one section"
        )));
    }
    let mut sections = Vec::with_capacity(sections_value.len());
    for raw in sections_value {
        let name = raw
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::LlmFatal(format!("{ctx}: source sections must be non-empty")))?;
        sections.push(name.to_string());
    }
    Ok(SourceRefs { pages, sections })
}

/// Validate one question object; `position` is only for error context.
fn validate_one(
    value: &serde_json::Value,
    position: usize,
    unit: &UnitText,
) -> Result<ValidatedMcq> {
    let ctx = format!("question {position}");
    let question = value
        .get("question")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: missing non-empty 'question'")))?;
    let options_value = value
        .get("options")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: 'options' must be an array of 4")))?;
    if let Some(reason) = deictic_violation(question) {
        return Err(Error::LlmFatal(format!(
            "{ctx}: {reason}; rewrite the stem self-contained"
        )));
    }
    let options = validate_options(options_value, &ctx)?;
    let (correct_index, trap_index) = validate_indices(value, &ctx)?;
    let explanation = value
        .get("explanation")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| s.len() >= 20)
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: missing 'explanation' (2-5 sentences)")))?;
    let topic = value
        .get("topic")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal(format!("{ctx}: missing non-empty 'topic'")))?;
    let source_refs = validate_refs(value, &ctx, unit)?;
    Ok(ValidatedMcq {
        question: question.to_string(),
        options,
        correct_index,
        trap_index,
        explanation: explanation.to_string(),
        topic: topic.to_string(),
        source_refs,
    })
}

/// Validate a full LLM response: top-level `questions` array of
/// `min..=max` items, each passing [`validate_one`].
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] with a positioned message on any shape,
/// trap, explanation, or source-membership violation (callers append this
/// detail on fast retry; nothing malformed is cached).
pub fn validate_mcq_set_ranged(
    text: &str,
    unit: &UnitText,
    min: usize,
    max: usize,
) -> Result<Vec<ValidatedMcq>> {
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::LlmFatal(format!("invalid MCQ JSON: {e}")))?;
    let questions = parsed
        .get("questions")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::LlmFatal("MCQ JSON needs a 'questions' array".to_string()))?;
    if questions.len() < min || questions.len() > max {
        return Err(Error::LlmFatal(format!(
            "expected {min}-{max} questions, got {}",
            questions.len()
        )));
    }
    let mut out = Vec::with_capacity(questions.len());
    for (i, item) in questions.iter().enumerate() {
        let position = i.saturating_add(1);
        out.push(validate_one(item, position, unit)?);
    }
    Ok(out)
}

/// Validate with the spec §7.1 count (8–12 questions).
///
/// # Errors
///
/// See [`validate_mcq_set_ranged`].
pub fn validate_mcq_set(text: &str, unit: &UnitText) -> Result<Vec<ValidatedMcq>> {
    validate_mcq_set_ranged(text, unit, MIN_QUESTIONS, MAX_QUESTIONS)
}

/// JSON Schema for a review set response (constrained decoding): the same
/// question shape as [`mcq_response_schema`], but the array floor is 1, not
/// 8 — a review set holds one question per open misconception
/// (`1..=MAX_QUESTIONS`), and the §7.1 8–12 floor does not apply.
#[must_use]
pub fn mcq_review_response_schema() -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "minItems": MCQ_OPTION_COUNT,
                            "maxItems": MCQ_OPTION_COUNT,
                            "items": {"type": "string"},
                        },
                        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                        "trap_index": {"type": "integer", "minimum": 0, "maximum": 3},
                        "explanation": {"type": "string"},
                        "topic": {"type": "string"},
                        "source_refs": {
                            "type": "object",
                            "properties": {
                                "pages": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "integer"},
                                },
                                "sections": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["pages", "sections"],
                            "additionalProperties": false,
                        },
                    },
                    "required": ["question", "options", "correct_index", "trap_index", "explanation", "topic", "source_refs"],
                    "additionalProperties": false,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": false,
    })
    .to_string()
}

/// Check that a validated review set actually targets its misconceptions
/// (§12): every question's `topic` must be a listed concept verbatim
/// (answers route back to rows on that label — a paraphrase orphans the
/// evidence), and every listed concept must be probed at least once.
/// Callers chain this after [`validate_mcq_set_ranged`] inside the
/// `complete_cached` validator so drift fast-retries with direction.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] with a positioned message naming the
/// offending question or the uncovered concept.
pub fn check_review_topics(items: &[ValidatedMcq], concepts: &[ReviewConcept]) -> Result<()> {
    for (i, item) in items.iter().enumerate() {
        let position = i.saturating_add(1);
        if !concepts.iter().any(|c| c.concept == item.topic) {
            return Err(Error::LlmFatal(format!(
                "question {position}: topic '{}' is not one of the listed misconception concepts — set `topic` to the concept label verbatim",
                item.topic
            )));
        }
    }
    for target in concepts {
        if !items.iter().any(|item| item.topic == target.concept) {
            return Err(Error::LlmFatal(format!(
                "misconception [{}] '{}' is never re-probed — cover every listed concept at least once",
                target.id, target.concept
            )));
        }
    }
    Ok(())
}

/// `SplitMix64` step: cheap deterministic PRNG for seeded shuffles.
pub const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic Fisher-Yates permutation of `[0,1,2,3]` from `seed`.
/// Pure: the same seed always yields the same order; the LLM never controls
/// presentation order.
#[must_use]
pub fn shuffle_order(seed: u64) -> [usize; MCQ_OPTION_COUNT] {
    let mut order = [0_usize, 1, 2, 3];
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15 | 1);
    let mut cursor = order.len();
    while cursor > 1 {
        cursor = cursor.saturating_sub(1);
        let bound = cursor.saturating_add(1);
        let divisor = u64::try_from(bound).unwrap_or(1).max(1);
        let rand = splitmix64(&mut state)
            .checked_rem(divisor)
            .unwrap_or_default();
        let slot = usize::try_from(rand)
            .unwrap_or_default()
            .checked_rem(bound)
            .unwrap_or_default();
        order.swap(cursor, slot);
    }
    let first = order.first().copied().unwrap_or_default();
    let second = order.get(1).copied().unwrap_or_default();
    let third = order.get(2).copied().unwrap_or_default();
    let fourth = order.get(3).copied().unwrap_or_default();
    [first, second, third, fourth]
}

/// Clock-derived seed for production shuffles (tests use fixed seeds).
#[must_use]
pub fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0x1234_5678_9ABC_DEF0, |d| {
            u64::from(d.subsec_nanos())
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(d.as_secs())
        })
}

/// Apply a seeded shuffle to a validated item: remaps correct/trap into
/// displayed positions. Returns `None` when the item is internally
/// inconsistent (practically unreachable after validation).
#[must_use]
pub fn apply_shuffle(item: &ValidatedMcq, seed: u64) -> Option<DisplayedMcq> {
    let order = shuffle_order(seed);
    let mut displayed = Vec::with_capacity(MCQ_OPTION_COUNT);
    for slot in order {
        displayed.push(item.options.get(slot)?.clone());
    }
    let mut shown_correct = None;
    let mut shown_trap = None;
    for (shown, slot) in order.iter().enumerate() {
        if *slot == item.correct_index {
            shown_correct = Some(shown);
        }
        if *slot == item.trap_index {
            shown_trap = Some(shown);
        }
    }
    Some(DisplayedMcq {
        question: item.question.clone(),
        displayed_options: displayed,
        displayed_correct: shown_correct?,
        displayed_trap: shown_trap?,
        explanation: item.explanation.clone(),
        topic: item.topic.clone(),
    })
}

/// Whether a wrong answer logs a misconception (§7.1): only incorrect
/// *retest* answers, excluding `"I don't know"` (displayed index 4).
#[must_use]
pub const fn should_log_misconception(
    phase: McqPhase,
    is_correct: bool,
    selected_displayed: usize,
) -> bool {
    matches!(phase, McqPhase::Retest) && !is_correct && selected_displayed != IDK_INDEX
}

/// Page-level source gate (§18): every cited page inside the unit range is
/// `DIRECTLY_SUPPORTED`; an empty citation is `NOT_SUPPORTED`; anything
/// outside is `CONTRADICTED_BY_SOURCE` (rejected upstream).
#[must_use]
pub fn classify_pages(pages: &[i64], unit: &UnitText) -> SupportVerdict {
    if pages.is_empty() {
        return SupportVerdict::NotSupported;
    }
    let inside = pages
        .iter()
        .all(|p| *p >= unit.page_start && *p <= unit.page_end);
    if inside {
        SupportVerdict::DirectlySupported
    } else {
        SupportVerdict::ContradictedBySource
    }
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

    fn good_item_json() -> serde_json::Value {
        serde_json::json!({
            "question": "What does &x yield?",
            "options": ["The address of x", "The value of x", "A null pointer", "A dangling reference"],
            "correct_index": 0,
            "trap_index": 1,
            "explanation": "The & operator takes the address of its operand. This is the foundation of pointer basics.",
            "topic": "addresses",
            "source_refs": {"pages": [12], "sections": ["Addresses"]}
        })
    }

    fn set_with(item: &serde_json::Value, count: usize) -> String {
        let items: Vec<serde_json::Value> = (0..count).map(|_| item.clone()).collect();
        serde_json::json!({"questions": items}).to_string()
    }

    #[test]
    fn phase_round_trip() {
        assert_eq!(McqPhase::parse("pretest").unwrap(), McqPhase::Pretest);
        assert_eq!(McqPhase::parse("retest").unwrap(), McqPhase::Retest);
        assert_eq!(McqPhase::parse("review").unwrap(), McqPhase::Review);
        assert!(McqPhase::parse("quiz").is_err());
        assert_eq!(McqPhase::Pretest.as_str(), "pretest");
        assert_eq!(McqPhase::Review.as_str(), "review");
    }

    #[test]
    fn task_types_map_to_mcq_phases() {
        use crate::domain::TaskType;
        assert_eq!(
            McqPhase::for_task(TaskType::Pretest),
            Some(McqPhase::Pretest)
        );
        assert_eq!(McqPhase::for_task(TaskType::Retest), Some(McqPhase::Retest));
        assert_eq!(McqPhase::for_task(TaskType::Read), None);
        assert_eq!(McqPhase::for_task(TaskType::AssignmentWrite), None);
        assert_eq!(McqPhase::for_task(TaskType::Notes), None);
    }

    fn review_concepts() -> Vec<ReviewConcept> {
        vec![
            ReviewConcept {
                id: 7,
                concept: "addresses".to_string(),
                description: "took &x for the value".to_string(),
                evidence: "Selected 'The value of x' instead of 'The address of x'".to_string(),
            },
            ReviewConcept {
                id: 9,
                concept: "deref".to_string(),
                description: "applied * to a non-pointer".to_string(),
                evidence: "Selected '*x compiles' instead of 'type error'".to_string(),
            },
        ]
    }

    fn reviewed_item(topic: &str) -> ValidatedMcq {
        ValidatedMcq {
            question: "What does &x yield?".to_string(),
            options: vec![
                "The address of x".to_string(),
                "The value of x".to_string(),
                "A null pointer".to_string(),
                "A dangling reference".to_string(),
            ],
            correct_index: 0,
            trap_index: 1,
            explanation: "The & operator takes the address of its operand. This is the foundation of pointer basics.".to_string(),
            topic: topic.to_string(),
            source_refs: SourceRefs {
                pages: vec![12],
                sections: vec!["Addresses".to_string()],
            },
        }
    }

    #[test]
    fn review_prompt_lists_concepts_verbatim() {
        let prompt = build_review_prompt(&unit(), 2, &review_concepts());
        assert!(prompt.contains("MAXIMUM difficulty targeted retest"));
        assert!(prompt.contains("VERBATIM"));
        assert!(prompt.contains("[7] concept: addresses"));
        assert!(prompt.contains("took &x for the value"));
        assert!(prompt.contains("The value of x"));
        assert!(prompt.contains("Cover every listed misconception"));
    }

    #[test]
    fn review_schema_floors_at_one_question() {
        let schema: serde_json::Value =
            serde_json::from_str(&mcq_review_response_schema()).unwrap();
        assert_eq!(schema["properties"]["questions"]["minItems"], 1);
        assert_eq!(schema["properties"]["questions"]["maxItems"], MAX_QUESTIONS);
        // The standard schema keeps the §7.1 floor: the two must differ.
        assert_ne!(mcq_review_response_schema(), mcq_response_schema());
    }

    #[test]
    fn review_topics_gate_routes_and_covers() {
        let concepts = review_concepts();
        let full = vec![reviewed_item("addresses"), reviewed_item("deref")];
        assert!(check_review_topics(&full, &concepts).is_ok());
        // A paraphrased topic orphans the evidence — rejected with position.
        let drifted = vec![reviewed_item("addresses"), reviewed_item("addressing")];
        let err = check_review_topics(&drifted, &concepts).unwrap_err();
        assert!(matches!(err, Error::LlmFatal(_)));
        assert!(err.to_string().contains("question 2"));
        // A listed concept with no question means the selection lied.
        let partial = vec![reviewed_item("addresses")];
        let uncovered = check_review_topics(&partial, &concepts).unwrap_err();
        assert!(uncovered.to_string().contains("deref"));
    }

    #[test]
    fn prompts_differ_by_phase_and_cite_source() {
        let u = unit();
        let pre = build_pretest_prompt(&u, 8);
        let re = build_retest_prompt(&u, 8);
        assert_ne!(pre, re);
        assert!(pre.contains("moderate pretest"));
        assert!(re.contains("MAXIMUM difficulty retest"));
        assert!(pre.contains("strictly on the chapter text"));
        assert!(re.contains("strictly on the chapter text"));
        assert!(pre.contains("Pointers"));
    }

    #[test]
    fn prompts_require_self_contained_stems() {
        let u = unit();
        for prompt in [build_pretest_prompt(&u, 8), build_retest_prompt(&u, 8)] {
            assert!(prompt.contains("SELF-CONTAINED"));
            assert!(prompt.contains("closed-book"));
            assert!(prompt.contains("inline the complete minimal excerpt"));
            assert!(prompt.contains("never appear in stems or options"));
        }
    }

    #[test]
    fn detector_catches_book_pointers() {
        let stem = "In the bad.c program from Listing 1.2, why does clang treat the diagnostic on line 22 as fatal?";
        let reason = deictic_violation(stem).unwrap();
        assert!(reason.contains("listing"), "{reason}");
        assert!(deictic_violation("What happens on line 22?").is_some());
        assert!(deictic_violation("See page 12 for the answer.").is_some());
        assert!(deictic_violation("As shown in Figure 3, what holds?").is_some());
        assert!(deictic_violation("Unlike the example above, what does this do?").is_some());
        assert!(deictic_violation("Which Section 4 rule applies here?").is_some());
        // Book-specific box labels used as pointers.
        assert!(
            deictic_violation(
                "According to TAKEAWAY 2.5, what relates declarations to definitions?"
            )
            .is_some()
        );
        assert!(deictic_violation("Solve Challenge 3 with pointers.").is_some());
    }

    #[test]
    fn detector_allows_standalone_text() {
        assert_eq!(deictic_violation("What does &x yield?"), None);
        // Bare above/below in domain content is legal.
        assert_eq!(
            deictic_violation("Which values count as true: above zero or below?"),
            None
        );
        // Substrings of longer words never match.
        assert_eq!(deictic_violation("When is a function inlined?"), None);
        assert_eq!(deictic_violation("Name the pipeline stages."), None);
        assert_eq!(deictic_violation("The outline covers pointers."), None);
        // Box labels as plain concepts (no number) stay legal.
        assert_eq!(
            deictic_violation("What does the as-if rule guarantee?"),
            None
        );
        assert_eq!(
            deictic_violation("What is the challenge with dangling pointers?"),
            None
        );
    }

    #[test]
    fn validation_rejects_listing_reference_in_stem() {
        let mut item = good_item_json();
        item["question"] = serde_json::json!(
            "In the bad.c program from Listing 1.2, why does clang treat the diagnostic on line 22 as fatal?"
        );
        let text = set_with(&item, 8);
        let err = validate_mcq_set(&text, &unit()).unwrap_err();
        assert!(err.to_string().contains("self-contained"), "{err}");
    }

    #[test]
    fn validation_rejects_pointer_in_option() {
        let mut item = good_item_json();
        item["options"][2] = serde_json::json!("The case from Listing 1.2");
        let text = set_with(&item, 8);
        assert!(validate_mcq_set(&text, &unit()).is_err());
    }

    #[test]
    fn validation_leaves_explanations_citable() {
        let mut item = good_item_json();
        item["explanation"] = serde_json::json!(
            "The & operator takes the address of its operand; see Section 2 on page 12 for details."
        );
        let text = set_with(&item, 8);
        // Explanations are shown post-answer, so re-read pointers stay legal.
        assert!(validate_mcq_set(&text, &unit()).is_ok());
    }

    #[test]
    fn params_and_hash_shape() {
        let params = mcq_params_json(8, McqPhase::Pretest);
        assert!(params.contains("\"phase\":\"pretest\""));
        assert!(params.contains("\"rf\":\""));
        assert_eq!(source_hash_for("abc").len(), 64);
    }

    #[test]
    fn response_schema_mirrors_validator_shape() {
        let schema: serde_json::Value = serde_json::from_str(&mcq_response_schema()).unwrap();
        let questions = &schema["properties"]["questions"];
        assert_eq!(questions["minItems"], serde_json::json!(MIN_QUESTIONS));
        assert_eq!(questions["maxItems"], serde_json::json!(MAX_QUESTIONS));
        let item = &questions["items"];
        assert_eq!(
            item["properties"]["options"]["minItems"],
            serde_json::json!(4)
        );
        assert_eq!(
            item["properties"]["options"]["maxItems"],
            serde_json::json!(4)
        );
        assert_eq!(
            item["properties"]["correct_index"]["maximum"],
            serde_json::json!(3)
        );
        assert_eq!(
            item["properties"]["trap_index"]["maximum"],
            serde_json::json!(3)
        );
        assert_eq!(
            item["properties"]["source_refs"]["properties"]["pages"]["minItems"],
            serde_json::json!(1)
        );
        assert_eq!(item["required"].as_array().unwrap().len(), 7);
        // The params tag tracks this exact schema text.
        let tag = crate::llm::schema_tag(&mcq_response_schema());
        assert!(mcq_params_json(8, McqPhase::Pretest).contains(&tag));
    }

    #[test]
    fn detector_boundaries() {
        // All three IDK spellings match; near-misses do not.
        assert!(is_idk_smuggle("I don't know"));
        assert!(is_idk_smuggle("i dont know"));
        assert!(is_idk_smuggle("e) i don't know"));
        assert!(!is_idk_smuggle("I don't know everything"));
        assert!(!is_idk_smuggle("dunno"));
        // Word chars: alphanumerics and underscore only.
        assert!(is_word_char('a'));
        assert!(is_word_char('7'));
        assert!(is_word_char('_'));
        assert!(!is_word_char(' '));
        assert!(!is_word_char('.'));
        // Whole-word occurrence needs non-word guards on BOTH sides.
        assert!(contains_phrase("see listing here", "listing"));
        assert!(!contains_phrase("see alisting here", "listing"));
        assert!(!contains_phrase("see listing7 here", "listing"));
    }

    #[test]
    fn index_bounds_reject_each_side() {
        let mut bad_correct = good_item_json();
        bad_correct["correct_index"] = serde_json::json!(4);
        assert!(validate_mcq_set(&set_with(&bad_correct, 8), &unit()).is_err());
        let mut bad_trap = good_item_json();
        bad_trap["trap_index"] = serde_json::json!(4);
        assert!(validate_mcq_set(&set_with(&bad_trap, 8), &unit()).is_err());
    }

    #[test]
    fn refs_and_counts_accept_edges() {
        // Unit-edge pages are inside the range.
        let mut low = good_item_json();
        low["source_refs"]["pages"] = serde_json::json!([10]);
        assert!(validate_mcq_set(&set_with(&low, 8), &unit()).is_ok());
        let mut high = good_item_json();
        high["source_refs"]["pages"] = serde_json::json!([20]);
        assert!(validate_mcq_set(&set_with(&high, 8), &unit()).is_ok());
        // Exactly MAX_QUESTIONS is valid, not over.
        assert!(validate_mcq_set(&set_with(&good_item_json(), 12), &unit()).is_ok());
    }

    #[test]
    fn shuffle_order_golden_vectors() {
        // Exact outputs: any PRNG-operator or loop-bound change alters these.
        let expected: [[usize; 4]; 8] = [
            [2, 3, 1, 0],
            [2, 1, 0, 3],
            [1, 3, 0, 2],
            [2, 3, 0, 1],
            [1, 2, 3, 0],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [2, 1, 3, 0],
        ];
        for (seed, want) in expected.iter().enumerate() {
            let seed = u64::try_from(seed).unwrap();
            assert_eq!(&shuffle_order(seed), want, "seed {seed}");
        }
    }

    #[test]
    fn accepts_valid_set() {
        let text = set_with(&good_item_json(), 8);
        let items = validate_mcq_set(&text, &unit()).unwrap();
        assert_eq!(items.len(), 8);
        assert_eq!(items[0].correct_index, 0);
        assert_eq!(items[0].trap_index, 1);
    }

    #[test]
    fn rejects_wrong_count() {
        let text = set_with(&good_item_json(), 2);
        assert!(validate_mcq_set(&text, &unit()).is_err());
        // Ranged entry point allows small sets for cheap smoke tests.
        assert!(validate_mcq_set_ranged(&text, &unit(), 1, 12).is_ok());
    }

    #[test]
    fn rejects_idk_smuggle() {
        let mut item = good_item_json();
        item["options"][2] = serde_json::json!("I don't know");
        let text = set_with(&item, 8);
        assert!(validate_mcq_set(&text, &unit()).is_err());
    }

    #[test]
    fn rejects_trap_equal_correct() {
        let mut item = good_item_json();
        item["trap_index"] = serde_json::json!(0);
        let text = set_with(&item, 8);
        let err = validate_mcq_set(&text, &unit()).unwrap_err();
        assert!(err.to_string().contains("trap must differ"));
    }

    #[test]
    fn rejects_duplicate_options() {
        let mut item = good_item_json();
        item["options"][3] = serde_json::json!("the ADDRESS of x");
        let text = set_with(&item, 8);
        assert!(validate_mcq_set(&text, &unit()).is_err());
    }

    #[test]
    fn rejects_out_of_range_page() {
        let mut item = good_item_json();
        item["source_refs"]["pages"] = serde_json::json!([99]);
        let text = set_with(&item, 8);
        let err = validate_mcq_set(&text, &unit()).unwrap_err();
        assert!(err.to_string().contains("outside unit range"));
    }

    #[test]
    fn rejects_missing_explanation() {
        let mut item = good_item_json();
        item["explanation"] = serde_json::json!("too short");
        let text = set_with(&item, 8);
        assert!(validate_mcq_set(&text, &unit()).is_err());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(validate_mcq_set("not json", &unit()).is_err());
        assert!(validate_mcq_set("{\"questions\": []}", &unit()).is_err());
    }

    #[test]
    fn shuffle_is_permutation_and_deterministic() {
        let first = shuffle_order(42);
        assert_eq!(first, shuffle_order(42));
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
        // Different seeds usually differ (probabilistic but stable for 7 vs 42
        // under SplitMix64; assert at least one differs across a few seeds).
        let others = [shuffle_order(1), shuffle_order(2), shuffle_order(3)];
        assert!(others.iter().any(|o| *o != first));
    }

    #[test]
    fn apply_shuffle_remaps_correct_and_trap() {
        let text = set_with(&good_item_json(), 8);
        let items = validate_mcq_set(&text, &unit()).unwrap();
        let item = items.first().unwrap();
        for seed in [0, 1, 42, 999] {
            let shown = apply_shuffle(item, seed).unwrap();
            assert_eq!(shown.displayed_options.len(), MCQ_OPTION_COUNT);
            assert_ne!(shown.displayed_correct, shown.displayed_trap);
            assert_eq!(
                shown.displayed_options.get(shown.displayed_correct),
                item.options.get(item.correct_index)
            );
            assert_eq!(
                shown.displayed_options.get(shown.displayed_trap),
                item.options.get(item.trap_index)
            );
        }
    }

    #[test]
    fn misconception_rule_only_retest_non_idk() {
        assert!(should_log_misconception(McqPhase::Retest, false, 0));
        assert!(!should_log_misconception(
            McqPhase::Retest,
            false,
            IDK_INDEX
        ));
        assert!(!should_log_misconception(McqPhase::Retest, true, 0));
        assert!(!should_log_misconception(McqPhase::Pretest, false, 0));
    }

    #[test]
    fn page_gate() {
        let u = unit();
        assert_eq!(
            classify_pages(&[10, 20], &u),
            SupportVerdict::DirectlySupported
        );
        assert_eq!(classify_pages(&[], &u), SupportVerdict::NotSupported);
        assert_eq!(
            classify_pages(&[99], &u),
            SupportVerdict::ContradictedBySource
        );
    }
}
