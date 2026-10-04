//! Written-assignment engine (§7.2 / §8.1): prompts, validation, serialization.
//!
//! The assignment is closed-book: memory, reasoning, and internalized mental
//! models only. Generation produces 3–5 multi-part written questions plus one
//! coding question, each with a **creation-time rubric** (criteria, max
//! scores, model solution) so grading later judges against a frozen standard
//! rather than inventing one. At least one question re-probes open
//! misconceptions when any exist. Like MCQs, every question carries internal
//! source provenance and closed-book self-contained wording.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

use crate::engines::{SupportVerdict, UnitText};
use crate::error::{Error, Result};
use crate::mcq::SourceRefs;
use crate::store::NewAssignmentQuestion;

/// Written questions per set (§7.2 flexes with chapter depth).
pub const MIN_WRITTEN: usize = 3;
/// Written questions per set (§7.2 flexes with chapter depth).
pub const MAX_WRITTEN: usize = 5;
/// Exactly one coding question per set (§7.2).
pub const CODING_COUNT: usize = 1;
/// Parts per written question (multi-part `a) … b) …`).
pub const MIN_PARTS: usize = 2;
/// Parts per written question (multi-part `a) … b) …`).
pub const MAX_PARTS: usize = 5;
/// Coding problems must be solvable in small programs (§7.2).
pub const MAX_ESTIMATED_LOC: i64 = 500;
/// Completion-token cap for assignment generation (solutions are long).
pub const ASSIGNMENT_MAX_TOKENS: u32 = 12_000;
/// Minimum model-solution length: a stub solution defeats rubric grading.
pub const MIN_SOLUTION_CHARS: usize = 20;

/// Question kind: multi-part written work or the single coding problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// Closed-book written question with 2–5 sub-parts.
    Written,
    /// Small creative program using chapter ideas (< 500 LOC).
    Coding,
}

impl QuestionKind {
    /// Canonical label stored in SQLite.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Written => "written",
            Self::Coding => "coding",
        }
    }

    /// Parse a stored or generated label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "written" => Ok(Self::Written),
            "coding" => Ok(Self::Coding),
            other => Err(Error::InvalidInput(format!(
                "unknown assignment kind: {other}"
            ))),
        }
    }
}

/// One grading criterion frozen at creation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Criterion {
    /// Short criterion name (`correctness`, `edge cases`, …).
    pub name: String,
    /// Points available for this criterion.
    pub max_score: i64,
    /// What a good answer demonstrates.
    pub what_good_looks_like: String,
}

/// Creation-time rubric: the grading standard, frozen before answers exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rubric {
    /// Per-criterion breakdown.
    pub criteria: Vec<Criterion>,
    /// Total points available.
    pub max_score: i64,
    /// Expected internal solution (reference for the grader, never shown
    /// to the student before grading).
    pub model_solution: String,
}

/// One validated assignment question, ready to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedQuestion {
    /// Written or coding.
    pub kind: QuestionKind,
    /// Question parts (`a) …`, `b) …`, …; coding has the problem statement).
    pub parts: Vec<String>,
    /// Frozen grading standard.
    pub rubric: Rubric,
    /// Misconception ids this question re-probes (subset of the open set).
    pub target_misconception_ids: Vec<i64>,
    /// Internal provenance (pages inside the unit range).
    pub source_refs: SourceRefs,
    /// Coding only: estimated solution size (< 500 LOC).
    pub estimated_loc: Option<i64>,
}

/// An open misconception the assignment must re-probe (§12): id plus concept,
/// mapped from store rows by the caller (the engine never touches the store).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenMisconception {
    /// Misconception row id (referenced by `target_misconception_ids`).
    pub id: i64,
    /// Short concept label.
    pub concept: String,
}

/// Hex SHA-256 over the open misconceptions: the identity half that keeps
/// cache entries apart when the same chapter is generated under different
/// misconception sets.
#[must_use]
pub fn misconceptions_hash(items: &[OpenMisconception]) -> String {
    let mut hasher = Sha256::new();
    for item in items {
        hasher.update(item.id.to_string().as_bytes());
        hasher.update(b":");
        hasher.update(item.concept.as_bytes());
        hasher.update(b";");
    }
    hex::encode(hasher.finalize())
}

/// Canonical params JSON for assignment calls (token cap + misconception-set
/// identity + response-schema tag; the prompt text itself varies with
/// misconception wording).
#[must_use]
pub fn assignment_params_json(misconceptions_hash: &str) -> String {
    let tag = crate::llm::schema_tag(&assignment_response_schema());
    format!(
        "{{\"max_tokens\":{ASSIGNMENT_MAX_TOKENS},\"operation\":\"assignment\",\"misconceptions_hash\":\"{misconceptions_hash}\",\"rf\":\"{tag}\"}}"
    )
}

/// JSON Schema for the assignment set response (constrained decoding):
/// `questions` items mirroring [`validate_assignment_set`] structurally —
/// kind enum, parts, full rubric, targets, provenance, and `estimated_loc` on
/// every question (`0` for written, the LOC estimate for coding — strict mode
/// requires all properties present, so the validator ignores the written
/// value). Counts (3–5 written, exactly 1 coding), re-probe coverage, and
/// closed-book wording stay in the validator.
#[must_use]
pub fn assignment_response_schema() -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["written", "coding"]},
                        "parts": {"type": "array", "items": {"type": "string"}},
                        "rubric": {
                            "type": "object",
                            "properties": {
                                "criteria": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "max_score": {"type": "integer", "minimum": 1},
                                            "what_good_looks_like": {"type": "string"},
                                        },
                                        "required": ["name", "max_score", "what_good_looks_like"],
                                        "additionalProperties": false,
                                    },
                                },
                                "max_score": {"type": "integer", "minimum": 1},
                                "model_solution": {"type": "string"},
                            },
                            "required": ["criteria", "max_score", "model_solution"],
                            "additionalProperties": false,
                        },
                        "target_misconception_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "source_refs": {
                            "type": "object",
                            "properties": {
                                "pages": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "sections": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["pages", "sections"],
                            "additionalProperties": false,
                        },
                        "estimated_loc": {"type": "integer", "minimum": 0},
                    },
                    "required": ["kind", "parts", "rubric", "target_misconception_ids", "source_refs", "estimated_loc"],
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

/// Closed-book framing (§7.2): no chapter text, notes, or external sources
/// while answering — questions must probe internalized models, and any code
/// or quoted text needed must be inlined in full.
const CLOSED_BOOK: &str = "CLOSED-BOOK assignment: the student answers from memory, reasoning, and internalized mental models only — no chapter text, notes, or external sources. NEVER refer to listings, figures, tables, sections, exercises, examples, page numbers, or line numbers; if a question needs code, output, or quoted text, inline the complete minimal excerpt directly in the question. Target maximum difficulty (retest-equivalent, harder in practice through active recall plus synthesis): multi-step reasoning, edge cases, counterexamples, transfer, and creative problem-solving — never mere recall or syntax reproduction.";

/// Rubric design: criteria test reasoning, never label recall — the student
/// reads once, so paraphrase counts and takeaway numbers do not.
const RUBRIC_DESIGN: &str = "Rubric design: criteria must test technical understanding and reasoning (mechanisms, traces, edge cases), never recall of labels. Do NOT write criteria that require naming takeaway numbers, section titles, or verbatim book terms; each criterion's \"what_good_looks_like\" must describe the observable technical content (e.g. 'states 0 is false and non-zero is true') rather than citation (e.g. 'names Takeaway 3.3'). A paraphrased but technically correct answer satisfies the criterion.";

/// Build the assignment prompt: 3–5 multi-part written questions plus one
/// coding question, each with a creation-time rubric, re-probing the open
/// misconceptions when any exist.
#[must_use]
pub fn build_assignment_prompt(unit: &UnitText, misconceptions: &[OpenMisconception]) -> String {
    let mut prompt = format!(
        "You are writing a closed-book written assignment for a study system.\n{SOURCE_FIDELITY}\n{CLOSED_BOOK}\n{RUBRIC_DESIGN}\nWrite the assignment on the chapter below (heading: {heading}, pages {start}-{end}).\nRespond with a single JSON object and nothing else: {{\"questions\": [{{\"kind\": \"written\", \"parts\": [\"a) ...\", \"b) ...\"], \"rubric\": {{\"criteria\": [{{\"name\": \"correctness\", \"max_score\": 5, \"what_good_looks_like\": \"...\"}}], \"max_score\": 10, \"model_solution\": \"...\"}}, \"target_misconception_ids\": [3], \"source_refs\": {{\"pages\": [12], \"sections\": [\"...\"]}}, \"estimated_loc\": 0}}, {{\"kind\": \"coding\", \"parts\": [\"Write a program that ...\"], \"rubric\": {{...}}, \"target_misconception_ids\": [], \"source_refs\": {{...}}, \"estimated_loc\": 120}}]}}.\nRules: 3-5 \"written\" questions, each with 2-5 non-empty parts (a/b/c...); exactly 1 \"coding\" question (a real small program using chapter ideas, estimated under 500 LOC, with \"estimated_loc\"); every question carries \"estimated_loc\" (the coding LOC estimate, 0 for written — the output schema requires it on all questions); every question needs a rubric (at least 1 criterion with max_score >= 1, a substantial model_solution) and source_refs (at least one page inside the unit range, at least one section name); target_misconception_ids may only reference the open misconceptions listed below. Emit raw JSON only (no markdown fences, no commentary); escape newlines inside strings as \\n.\n\n--- CHAPTER TEXT (pages {start}-{end}) ---\n{text}\n--- END CHAPTER TEXT ---",
        heading = unit.heading,
        start = unit.page_start,
        end = unit.page_end,
        text = unit.text,
    );
    if misconceptions.is_empty() {
        prompt.push_str("\n\nNo open misconceptions: no re-probing required.");
    } else {
        prompt.push_str("\n\nOPEN MISCONCEPTIONS (at least 1-2 questions must re-probe these via target_misconception_ids):\n");
        for item in misconceptions {
            // `writeln!` on a `String` never fails; the result is discarded.
            let _ = writeln!(prompt, "- id {}: {}", item.id, item.concept);
        }
    }
    prompt
}

/// Validate one rubric criterion: named, scored ≥ 1, with a "good" sketch.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect.
fn validate_criterion(raw: &serde_json::Value, position: usize, index: usize) -> Result<Criterion> {
    let name = raw
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            Error::LlmFatal(format!(
                "question {position}: criterion {index} needs a name"
            ))
        })?;
    let max_score = raw
        .get("max_score")
        .and_then(serde_json::Value::as_i64)
        .filter(|s| *s >= 1)
        .ok_or_else(|| {
            Error::LlmFatal(format!(
                "question {position}: criterion {index} needs max_score >= 1"
            ))
        })?;
    let what_good = raw
        .get("what_good_looks_like")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            Error::LlmFatal(format!(
                "question {position}: criterion {index} needs what_good_looks_like"
            ))
        })?;
    Ok(Criterion {
        name: name.to_string(),
        max_score,
        what_good_looks_like: what_good.to_string(),
    })
}

/// Validate one rubric: criteria present and scored, solution substantial.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect (fed back into the
/// fast-retry repair loop by `complete_cached`).
fn validate_rubric(value: &serde_json::Value, position: usize) -> Result<Rubric> {
    let get = |field: &str| {
        value.get(field).ok_or_else(|| {
            Error::LlmFatal(format!("question {position}: rubric missing '{field}'"))
        })
    };
    let criteria_raw = get("criteria")?.as_array().ok_or_else(|| {
        Error::LlmFatal(format!(
            "question {position}: rubric criteria must be an array"
        ))
    })?;
    if criteria_raw.is_empty() {
        return Err(Error::LlmFatal(format!(
            "question {position}: rubric needs at least 1 criterion"
        )));
    }
    let mut criteria = Vec::with_capacity(criteria_raw.len());
    for (index, raw) in criteria_raw.iter().enumerate() {
        criteria.push(validate_criterion(raw, position, index)?);
    }
    let max_score = get("max_score")?
        .as_i64()
        .filter(|s| *s >= 1)
        .ok_or_else(|| {
            Error::LlmFatal(format!("question {position}: rubric needs max_score >= 1"))
        })?;
    let solution = get("model_solution")?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            Error::LlmFatal(format!(
                "question {position}: rubric needs a model_solution"
            ))
        })?;
    if solution.chars().count() < MIN_SOLUTION_CHARS {
        return Err(Error::LlmFatal(format!(
            "question {position}: model_solution is a stub ({} chars, need {MIN_SOLUTION_CHARS}+)",
            solution.chars().count()
        )));
    }
    Ok(Rubric {
        criteria,
        max_score,
        model_solution: solution.to_string(),
    })
}

/// Validate source provenance: pages present, in-range, sections named.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on missing or out-of-range references.
fn validate_source_refs(
    value: &serde_json::Value,
    position: usize,
    unit: &UnitText,
) -> Result<SourceRefs> {
    let refs = value
        .get("source_refs")
        .ok_or_else(|| Error::LlmFatal(format!("question {position}: missing source_refs")))?;
    let pages: Vec<i64> = refs
        .get("pages")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    let sections: Vec<String> = refs
        .get("sections")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default();
    match crate::mcq::classify_pages(&pages, unit) {
        SupportVerdict::DirectlySupported | SupportVerdict::InferredFromSource => {}
        SupportVerdict::NotSupported => {
            return Err(Error::LlmFatal(format!(
                "question {position}: source_refs needs at least one page"
            )));
        }
        SupportVerdict::ContradictedBySource => {
            return Err(Error::LlmFatal(format!(
                "question {position}: source_refs pages fall outside pages {}-{}",
                unit.page_start, unit.page_end
            )));
        }
    }
    if sections.iter().all(|s| s.trim().is_empty()) {
        return Err(Error::LlmFatal(format!(
            "question {position}: source_refs needs at least one section name"
        )));
    }
    Ok(SourceRefs { pages, sections })
}

/// Validate a question's parts: count per kind plus closed-book wording
/// (no pointers at book content in any part).
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on the first defect found.
fn validate_question_parts(
    raw: &serde_json::Value,
    kind: QuestionKind,
    position: usize,
) -> Result<Vec<String>> {
    let parts: Vec<String> = raw
        .get("parts")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::trim).filter(|s| !s.is_empty()))
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default();
    match kind {
        QuestionKind::Written => {
            if parts.len() < MIN_PARTS || parts.len() > MAX_PARTS {
                return Err(Error::LlmFatal(format!(
                    "question {position}: written questions need {MIN_PARTS}-{MAX_PARTS} parts, got {}",
                    parts.len()
                )));
            }
        }
        QuestionKind::Coding => {
            if parts.is_empty() {
                return Err(Error::LlmFatal(format!(
                    "question {position}: coding question needs a problem statement"
                )));
            }
        }
    }
    // Closed-book wording: no pointers at book content in any part.
    let joined = parts.join("\n");
    if let Some(hit) = crate::mcq::deictic_violation(&joined) {
        return Err(Error::LlmFatal(format!(
            "question {position}: closed-book violation ({hit}) — inline the excerpt instead"
        )));
    }
    Ok(parts)
}

/// Validate misconception re-probe targets: every id must be open. Returns
/// the targets plus whether this question re-probes anything.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on unknown ids.
fn validate_reprobe_targets(
    raw: &serde_json::Value,
    position: usize,
    open: &[OpenMisconception],
) -> Result<(Vec<i64>, bool)> {
    let targets: Vec<i64> = raw
        .get("target_misconception_ids")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    for target in &targets {
        if !open.iter().any(|m| m.id == *target) {
            return Err(Error::LlmFatal(format!(
                "question {position}: target_misconception_ids references unknown id {target}"
            )));
        }
    }
    let reprobes = !targets.is_empty();
    Ok((targets, reprobes))
}

/// Validate one raw question object: kind, parts, closed-book wording,
/// creation-time rubric, provenance, misconception targets, coding cap.
/// Returns the question plus whether it re-probes an open misconception.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on the first defect found.
fn validate_one_question(
    raw: &serde_json::Value,
    position: usize,
    unit: &UnitText,
    open: &[OpenMisconception],
) -> Result<(ValidatedQuestion, bool)> {
    let kind_label = raw
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::LlmFatal(format!("question {position}: missing kind")))?;
    let kind = QuestionKind::parse(kind_label)
        .map_err(|_| Error::LlmFatal(format!("question {position}: bad kind '{kind_label}'")))?;
    let parts = validate_question_parts(raw, kind, position)?;
    let rubric = raw
        .get("rubric")
        .ok_or_else(|| Error::LlmFatal(format!("question {position}: missing rubric")))?;
    let rubric = validate_rubric(rubric, position)?;
    let source_refs = validate_source_refs(raw, position, unit)?;
    let (targets, reprobes) = validate_reprobe_targets(raw, position, open)?;
    let estimated_loc = match kind {
        QuestionKind::Written => None,
        QuestionKind::Coding => {
            let loc = raw
                .get("estimated_loc")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| {
                    Error::LlmFatal(format!("question {position}: coding needs estimated_loc"))
                })?;
            if !(1..=MAX_ESTIMATED_LOC).contains(&loc) {
                return Err(Error::LlmFatal(format!(
                    "question {position}: estimated_loc {loc} exceeds the {MAX_ESTIMATED_LOC}-LOC cap"
                )));
            }
            Some(loc)
        }
    };
    Ok((
        ValidatedQuestion {
            kind,
            parts,
            rubric,
            target_misconception_ids: targets,
            source_refs,
            estimated_loc,
        },
        reprobes,
    ))
}

/// Validate the full generated set: shape, rubrics, provenance, coding cap,
/// misconception re-probing, and closed-book wording.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on the first defect found (fail-fast keeps
/// repair prompts focused); [`Error::Io`] never occurs here.
pub fn validate_assignment_set(
    text: &str,
    unit: &UnitText,
    open: &[OpenMisconception],
) -> Result<Vec<ValidatedQuestion>> {
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::LlmFatal(format!("assignment is not valid JSON: {e}")))?;
    let raw_questions = parsed
        .get("questions")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::LlmFatal("assignment needs a 'questions' array".to_string()))?;
    let mut out = Vec::with_capacity(raw_questions.len());
    let mut written_count = 0_usize;
    let mut coding_count = 0_usize;
    let mut reprobed_count = 0_usize;
    for (position, raw) in raw_questions.iter().enumerate() {
        let (question, reprobes) = validate_one_question(raw, position, unit, open)?;
        match question.kind {
            QuestionKind::Written => {
                written_count = written_count.saturating_add(1);
            }
            QuestionKind::Coding => {
                coding_count = coding_count.saturating_add(1);
            }
        }
        if reprobes {
            reprobed_count = reprobed_count.saturating_add(1);
        }
        out.push(question);
    }
    if !(MIN_WRITTEN..=MAX_WRITTEN).contains(&written_count) {
        return Err(Error::LlmFatal(format!(
            "assignment needs {MIN_WRITTEN}-{MAX_WRITTEN} written questions, got {written_count}"
        )));
    }
    if coding_count != CODING_COUNT {
        return Err(Error::LlmFatal(format!(
            "assignment needs exactly {CODING_COUNT} coding question, got {coding_count}"
        )));
    }
    if !open.is_empty() && reprobed_count == 0 {
        return Err(Error::LlmFatal(
            "assignment must re-probe at least 1 open misconception".to_string(),
        ));
    }
    Ok(out)
}

/// Serialize validated output into storable rows (rubric frozen at creation).
///
/// # Errors
///
/// Returns [`Error::Io`] when JSON serialization fails (practically
/// unreachable: validated strings always serialize).
pub fn to_new_questions(
    chapter_id: i64,
    attempt_no: i64,
    validated: &[ValidatedQuestion],
) -> Result<Vec<NewAssignmentQuestion>> {
    let mut out = Vec::with_capacity(validated.len());
    for (position, item) in validated.iter().enumerate() {
        let parts_json =
            serde_json::to_string(&item.parts).map_err(|e| Error::Io(e.to_string()))?;
        let rubric_json = serde_json::to_string(&serde_json::json!({
            "criteria": item.rubric.criteria.iter().map(|c| serde_json::json!({
                "name": c.name,
                "max_score": c.max_score,
                "what_good_looks_like": c.what_good_looks_like,
            })).collect::<Vec<_>>(),
            "max_score": item.rubric.max_score,
            "model_solution": item.rubric.model_solution,
        }))
        .map_err(|e| Error::Io(e.to_string()))?;
        let targets_json = serde_json::to_string(&item.target_misconception_ids)
            .map_err(|e| Error::Io(e.to_string()))?;
        let Some(position_i64) = i64::try_from(position).ok() else {
            return Err(Error::Io("assignment position overflow".to_string()));
        };
        out.push(NewAssignmentQuestion {
            chapter_id,
            position: position_i64,
            kind: item.kind.as_str().to_string(),
            parts_json,
            rubric_json,
            target_misconception_ids: targets_json,
            attempt_no,
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

    fn good_set() -> serde_json::Value {
        let written = |targets: Vec<i64>| {
            serde_json::json!({
                "kind": "written",
                "parts": ["a) Explain what &x yields and why.", "b) Give a counterexample where address reasoning fails."],
                "rubric": {
                    "criteria": [{"name": "correctness", "max_score": 5, "what_good_looks_like": "Names the address and justifies it."}],
                    "max_score": 10,
                    "model_solution": "The & operator yields the address of x because it takes the address of its operand in memory."
                },
                "target_misconception_ids": targets,
                "source_refs": {"pages": [12], "sections": ["Addresses"]}
            })
        };
        serde_json::json!({
            "questions": [
                written(vec![3]),
                written(vec![]),
                written(vec![]),
                {
                    "kind": "coding",
                    "parts": ["Write a program that swaps two integers without a temporary, using only addresses."],
                    "rubric": {
                        "criteria": [{"name": "correctness", "max_score": 8, "what_good_looks_like": "Swaps correctly for all inputs."}],
                        "max_score": 8,
                        "model_solution": "Use the xor-swap or pointer dance carefully; the reference solution handles aliasing explicitly."
                    },
                    "target_misconception_ids": [],
                    "source_refs": {"pages": [13], "sections": ["Deref"]},
                    "estimated_loc": 40
                }
            ]
        })
    }

    fn open() -> Vec<OpenMisconception> {
        vec![OpenMisconception {
            id: 3,
            concept: "addresses".to_string(),
        }]
    }

    #[test]
    fn valid_set_passes_with_reprobe() {
        let validated = validate_assignment_set(&good_set().to_string(), &unit(), &open()).unwrap();
        assert_eq!(validated.len(), 4);
        assert_eq!(validated[0].kind, QuestionKind::Written);
        assert_eq!(validated[3].kind, QuestionKind::Coding);
        assert_eq!(validated[3].estimated_loc, Some(40));
        assert_eq!(validated[0].target_misconception_ids, vec![3]);
        let rows = to_new_questions(7, 2, &validated).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].attempt_no, 2);
        assert_eq!(rows[3].kind, "coding");
    }

    #[test]
    fn kind_round_trip() {
        assert_eq!(
            QuestionKind::parse("written").unwrap(),
            QuestionKind::Written
        );
        assert_eq!(QuestionKind::parse("coding").unwrap(), QuestionKind::Coding);
        assert!(QuestionKind::parse("quiz").is_err());
        assert_eq!(QuestionKind::Coding.as_str(), "coding");
    }

    #[test]
    fn wrong_counts_fail() {
        // Too few written.
        let mut few = good_set();
        few["questions"].as_array_mut().unwrap().remove(0);
        assert!(validate_assignment_set(&few.to_string(), &unit(), &open()).is_err());
        // No coding question.
        let mut no_code = good_set();
        no_code["questions"].as_array_mut().unwrap().pop();
        assert!(validate_assignment_set(&no_code.to_string(), &unit(), &open()).is_err());
        // Second coding question.
        let mut two_code = good_set();
        let coding = two_code["questions"][3].clone();
        two_code["questions"].as_array_mut().unwrap().push(coding);
        assert!(validate_assignment_set(&two_code.to_string(), &unit(), &open()).is_err());
    }

    #[test]
    fn boundary_counts_accepted() {
        // Exactly MAX_PARTS parts is valid; one more is not.
        let mut full = good_set();
        full["questions"][1]["parts"] =
            serde_json::json!(["a) One.", "b) Two.", "c) Three.", "d) Four.", "e) Five."]);
        assert!(validate_assignment_set(&full.to_string(), &unit(), &open()).is_ok());
        let mut over = good_set();
        over["questions"][1]["parts"] = serde_json::json!([
            "a) One.",
            "b) Two.",
            "c) Three.",
            "d) Four.",
            "e) Five.",
            "f) Six."
        ]);
        assert!(validate_assignment_set(&over.to_string(), &unit(), &open()).is_err());
        // A solution of exactly MIN_SOLUTION_CHARS is substantial, not a stub.
        let mut exact = good_set();
        exact["questions"][0]["rubric"]["model_solution"] =
            serde_json::json!("12345678901234567890");
        assert_eq!(
            exact["questions"][0]["rubric"]["model_solution"]
                .as_str()
                .unwrap_or_default()
                .chars()
                .count(),
            MIN_SOLUTION_CHARS
        );
        assert!(validate_assignment_set(&exact.to_string(), &unit(), &open()).is_ok());
    }

    #[test]
    fn single_part_written_fails() {
        let mut set = good_set();
        set["questions"][1]["parts"] = serde_json::json!(["a) Only one part here."]);
        assert!(validate_assignment_set(&set.to_string(), &unit(), &open()).is_err());
    }

    #[test]
    fn rubric_and_solution_gates() {
        // Missing rubric.
        let mut set = good_set();
        set["questions"][0]
            .as_object_mut()
            .unwrap()
            .remove("rubric");
        assert!(validate_assignment_set(&set.to_string(), &unit(), &open()).is_err());
        // Stub solution.
        let mut stub = good_set();
        stub["questions"][0]["rubric"]["model_solution"] = serde_json::json!("short");
        assert!(validate_assignment_set(&stub.to_string(), &unit(), &open()).is_err());
        // LOC cap.
        let mut big = good_set();
        big["questions"][3]["estimated_loc"] = serde_json::json!(501);
        assert!(validate_assignment_set(&big.to_string(), &unit(), &open()).is_err());
    }

    #[test]
    fn provenance_and_reprobe_gates() {
        // Out-of-range page.
        let mut set = good_set();
        set["questions"][0]["source_refs"]["pages"] = serde_json::json!([99]);
        assert!(validate_assignment_set(&set.to_string(), &unit(), &open()).is_err());
        // Dangling misconception target.
        let mut dangling = good_set();
        dangling["questions"][0]["target_misconception_ids"] = serde_json::json!([999]);
        assert!(validate_assignment_set(&dangling.to_string(), &unit(), &open()).is_err());
        // Open misconceptions but nothing re-probes them.
        let mut cold = good_set();
        cold["questions"][0]["target_misconception_ids"] = serde_json::json!([]);
        assert!(validate_assignment_set(&cold.to_string(), &unit(), &open()).is_err());
        // No open misconceptions: re-probing not required (and no targets).
        let mut bare = good_set();
        bare["questions"][0]["target_misconception_ids"] = serde_json::json!([]);
        assert!(validate_assignment_set(&bare.to_string(), &unit(), &[]).is_ok());
    }

    #[test]
    fn deictic_parts_fail() {
        let mut set = good_set();
        set["questions"][1]["parts"] =
            serde_json::json!(["a) Explain Listing 1.2 in detail.", "b) Why does it work?"]);
        assert!(validate_assignment_set(&set.to_string(), &unit(), &open()).is_err());
    }

    #[test]
    fn malformed_json_fails() {
        assert!(validate_assignment_set("{broken", &unit(), &open()).is_err());
        assert!(validate_assignment_set("{\"questions\": []}", &unit(), &open()).is_err());
    }

    #[test]
    fn misconception_hash_is_stable_and_sensitive() {
        let first = misconceptions_hash(&open());
        assert_eq!(first, misconceptions_hash(&open()));
        assert_ne!(
            first,
            misconceptions_hash(&[OpenMisconception {
                id: 4,
                concept: "addresses".to_string()
            }])
        );
        assert!(assignment_params_json(&first).contains(&first));
        assert!(assignment_params_json(&first).contains("\"max_tokens\":12000"));
        assert!(assignment_params_json(&first).contains("\"rf\":\""));
    }

    #[test]
    fn response_schema_covers_all_question_fields() {
        let schema: serde_json::Value =
            serde_json::from_str(&assignment_response_schema()).unwrap();
        let item = &schema["properties"]["questions"]["items"];
        assert_eq!(
            item["properties"]["kind"]["enum"],
            serde_json::json!(["written", "coding"])
        );
        assert_eq!(item["required"].as_array().unwrap().len(), 6);
        // `estimated_loc` is required on every question (0 for written) so
        // strict mode accepts the shape; the validator ignores written values.
        assert!(
            item["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("estimated_loc"))
        );
        assert_eq!(
            item["properties"]["rubric"]["properties"]["criteria"]["minItems"],
            serde_json::json!(1)
        );
        let tag = crate::llm::schema_tag(&assignment_response_schema());
        assert!(assignment_params_json("h").contains(&tag));
    }

    #[test]
    fn prompt_names_misconceptions() {
        let prompt = build_assignment_prompt(&unit(), &open());
        assert!(prompt.contains("closed-book"));
        assert!(prompt.contains("id 3"));
        assert!(prompt.contains("addresses"));
        assert!(prompt.contains("model_solution"));
        let bare = build_assignment_prompt(&unit(), &[]);
        assert!(bare.contains("No open misconceptions"));
    }

    #[test]
    fn prompt_forbids_label_recall_criteria() {
        let prompt = build_assignment_prompt(&unit(), &open());
        assert!(prompt.contains("never recall of labels"));
        assert!(prompt.contains("Do NOT write criteria that require naming takeaway numbers"));
        assert!(prompt.contains("paraphrased but technically correct"));
    }
}
