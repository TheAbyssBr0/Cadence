//! Rubric-based grading engine (§7.3 / §10): prompts, validation, identity.
//!
//! Grading judges a closed-book answer against the rubric frozen at
//! assignment-creation time — it never invents criteria. The prompt carries
//! the §7.3 conciseness instruction verbatim in intent (brevity, bullets, and
//! informal phrasing are never penalized when the technical facts are right)
//! and restricts verdicts to the seven §10 output categories. Like every
//! other generation stage, graded output is validated before it is cached,
//! and the cache identity hashes the exact (question, rubric, answer) triple
//! so a revised answer never collides with an earlier grade.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// Frozen-rubric types live in [`crate::assignment`] (creation time); grading
/// re-exports them so callers name one type for both stages.
pub use crate::assignment::{Criterion, Rubric};
use crate::error::{Error, Result};

/// Completion-token cap for grading calls (rubric + answer are small).
pub const GRADING_MAX_TOKENS: u32 = 4_000;

/// Grading verdicts (§10). `QUESTION_DEFECTIVE` never penalizes the user —
/// the prompt awards full credit and flags the item for replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradeClass {
    /// Technically correct and fully evidenced.
    Correct,
    /// Correct facts, minimal elaboration (full credit, §7.3).
    CorrectButBrief,
    /// Some criteria met, others missed or wrong.
    PartiallyCorrect,
    /// Technically wrong or off-target.
    Incorrect,
    /// Blank answer (score 0).
    Unanswered,
    /// Cannot be judged fairly (illegible, off-topic without signal).
    Ambiguous,
    /// The question or rubric is at fault — full credit, flagged.
    QuestionDefective,
}

impl GradeClass {
    /// Canonical label emitted by the grader and accepted by validation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "CORRECT",
            Self::CorrectButBrief => "CORRECT_BUT_BRIEF",
            Self::PartiallyCorrect => "PARTIALLY_CORRECT",
            Self::Incorrect => "INCORRECT",
            Self::Unanswered => "UNANSWERED",
            Self::Ambiguous => "AMBIGUOUS",
            Self::QuestionDefective => "QUESTION_DEFECTIVE",
        }
    }

    /// Parse a grader-emitted label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "CORRECT" => Ok(Self::Correct),
            "CORRECT_BUT_BRIEF" => Ok(Self::CorrectButBrief),
            "PARTIALLY_CORRECT" => Ok(Self::PartiallyCorrect),
            "INCORRECT" => Ok(Self::Incorrect),
            "UNANSWERED" => Ok(Self::Unanswered),
            "AMBIGUOUS" => Ok(Self::Ambiguous),
            "QUESTION_DEFECTIVE" => Ok(Self::QuestionDefective),
            other => Err(Error::InvalidInput(format!(
                "unknown grade classification: {other}"
            ))),
        }
    }
}

/// One per-criterion judgment, echoed against the frozen rubric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriterionResult {
    /// Must match a frozen criterion name exactly.
    pub name: String,
    /// Points awarded (`0 <= score <= max_score`).
    pub score: i64,
    /// Must echo the frozen criterion's `max_score`.
    pub max_score: i64,
    /// One-line justification.
    pub comment: String,
}

/// A validated grade for one answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grade {
    /// §10 verdict.
    pub classification: GradeClass,
    /// Total awarded (`0 <= score <= rubric max_score`).
    pub score: i64,
    /// Per-criterion breakdown (covers the rubric exactly once).
    pub criteria_results: Vec<CriterionResult>,
    /// Technical feedback for the student (2–5 sentences).
    pub feedback: String,
}

/// Conciseness instruction (§7.3, verbatim intent): the CLI buffer rewards
/// brevity, and the grader must not confuse terseness with ignorance.
const CONCISENESS: &str = "The student is completing this assignment in a CLI buffer. Evaluate answers based strictly on technical correctness, underlying logic, and mental models. Do NOT penalize brevity, informal phrasing, bullet-point formatting, or lack of narrative structure if the technical facts are correct. A concise answer can receive full credit.";

/// Category contract (§10): the only legal verdicts and their meanings.
const CATEGORIES: &str = "Classify the answer as exactly one of: CORRECT (technically correct and fully evidenced), CORRECT_BUT_BRIEF (correct facts with minimal elaboration — still full credit), PARTIALLY_CORRECT (some rubric criteria met, others missed or wrong), INCORRECT (technically wrong or off-target), UNANSWERED (blank answer — score 0), AMBIGUOUS (cannot be judged fairly), QUESTION_DEFECTIVE (the question or rubric itself is at fault — award full credit and explain the defect; this never penalizes the user).";

/// Adversarial self-audit: the frozen rubric was written by the same fallible
/// model that writes answers, so the grader must distrust both sides — the
/// student's answer and the model solution alike — and check each from first
/// principles instead of assuming the rubric is ground truth.
const ADVERSARIAL_AUDIT: &str = "Adversarial self-audit: assume NEITHER the student's answer NOR the frozen rubric/model solution is correct until checked. First, steelman the student's answer — construct the strongest case that it is technically correct (an unusual but valid phrasing, a correct alternative the rubric missed, a counterexample that actually holds). Only deliver a failing verdict when that case collapses. Second, audit the model solution itself from first principles: if it is wrong, self-contradictory, or the criteria contradict it, return QUESTION_DEFECTIVE with full credit and explain the defect instead of grading against a broken standard. Never fail a student for disagreeing with a wrong rubric.";

/// Closed-book leniency (single read): paraphrase is mastery — the student
/// answers from memory after one reading, so technical content in their own
/// words earns full credit without label recall.
const LENIENCY: &str = "Leniency (closed-book, single read): award full credit for technically correct content expressed in the student's own words. Never deduct for missing takeaway numbers, section names, or verbatim book phrasing when the underlying idea is right. Ignore minor wording imprecision (e.g. describing an omitted for-clause as having 'no conditional check' when its logic is correct) when the reasoning holds. Accept any technically valid alternative justification — a correct example need not match the rubric's anticipated illustration to earn the point. Deduct only for substantive technical errors, genuinely missing sub-parts, or wrong mechanisms — not for brevity, paraphrase, or omitted labels. CORRECT_BUT_BRIEF covers correct-but-terse answers at full or near-full score; use PARTIALLY_CORRECT only when a substantive technical gap remains.";

/// Hex SHA-256 over the (question, rubric, answer) triple: the identity half
/// that keeps cache entries apart when an answer is revised or a rubric is
/// replaced.
#[must_use]
pub fn grade_source_hash(question: &str, rubric_json: &str, answer: &str) -> String {
    let mut hasher = Sha256::new();
    for part in [question, rubric_json, answer] {
        hasher.update(part.as_bytes());
        hasher.update([0_u8]);
    }
    hex::encode(hasher.finalize())
}

/// Canonical params JSON for grading calls (token cap + operation identity +
/// response-schema tag). The tag binds the rubric-baked schema below, so a
/// revised rubric never collides with an earlier grade's cache entry.
#[must_use]
pub fn grade_params_json_for(rubric: &Rubric) -> String {
    let tag = crate::llm::schema_tag(&grade_response_schema(rubric));
    format!("{{\"max_tokens\":{GRADING_MAX_TOKENS},\"operation\":\"grade\",\"rf\":\"{tag}\"}}")
}

/// JSON Schema for one grading response (constrained decoding): the frozen
/// rubric baked into `prefixItems` — each position pins its criterion name
/// (single-value enum) and per-criterion maximum, with the array length fixed
/// to the criterion count. Coverage-exactly-once and echoed maxima are thus
/// structural; the validator keeps the semantic checks (does the awarded
/// score match the answer, is the verdict consistent).
#[must_use]
pub fn grade_response_schema(rubric: &Rubric) -> String {
    let items: Vec<serde_json::Value> = rubric
        .criteria
        .iter()
        .map(|criterion| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [criterion.name]},
                    "score": {"type": "integer", "minimum": 0, "maximum": criterion.max_score},
                    "max_score": {"type": "integer", "enum": [criterion.max_score]},
                    "comment": {"type": "string"},
                },
                "required": ["name", "score", "max_score", "comment"],
                "additionalProperties": false,
            })
        })
        .collect();
    let count = items.len();
    serde_json::json!({
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["CORRECT", "CORRECT_BUT_BRIEF", "PARTIALLY_CORRECT", "INCORRECT", "UNANSWERED", "AMBIGUOUS", "QUESTION_DEFECTIVE"],
            },
            "score": {"type": "integer", "minimum": 0, "maximum": rubric.max_score},
            "criteria_results": {
                "type": "array",
                "prefixItems": items,
                "minItems": count,
                "maxItems": count,
            },
            "feedback": {"type": "string"},
        },
        "required": ["classification", "score", "criteria_results", "feedback"],
        "additionalProperties": false,
    })
    .to_string()
}

/// Parse a frozen rubric from its stored JSON shape (`criteria`, `max_score`,
/// `model_solution` — the same shape [`crate::assignment::to_new_questions`]
/// writes into `rubric_json`).
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the JSON is not a well-formed frozen
/// rubric.
pub fn parse_rubric_json(text: &str) -> Result<Rubric> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::InvalidInput(format!("rubric is not valid JSON: {e}")))?;
    let get = |field: &str| {
        value
            .get(field)
            .ok_or_else(|| Error::InvalidInput(format!("rubric missing '{field}'")))
    };
    let raw_criteria = get("criteria")?
        .as_array()
        .ok_or_else(|| Error::InvalidInput("rubric criteria must be an array".to_string()))?;
    if raw_criteria.is_empty() {
        return Err(Error::InvalidInput(
            "rubric needs at least 1 criterion".to_string(),
        ));
    }
    let mut criteria = Vec::with_capacity(raw_criteria.len());
    for (index, raw) in raw_criteria.iter().enumerate() {
        let name = raw
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::InvalidInput(format!("rubric criterion {index} needs a name")))?;
        let max_score = raw
            .get("max_score")
            .and_then(serde_json::Value::as_i64)
            .filter(|s| *s >= 1)
            .ok_or_else(|| {
                Error::InvalidInput(format!("rubric criterion {index} needs max_score >= 1"))
            })?;
        let what_good = raw
            .get("what_good_looks_like")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "rubric criterion {index} needs what_good_looks_like"
                ))
            })?;
        criteria.push(Criterion {
            name: name.to_string(),
            max_score,
            what_good_looks_like: what_good.to_string(),
        });
    }
    let max_score = get("max_score")?
        .as_i64()
        .filter(|s| *s >= 1)
        .ok_or_else(|| Error::InvalidInput("rubric needs max_score >= 1".to_string()))?;
    let model_solution = get("model_solution")?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::InvalidInput("rubric needs a model_solution".to_string()))?;
    Ok(Rubric {
        criteria,
        max_score,
        model_solution: model_solution.to_string(),
    })
}

/// Build the grading prompt: frozen rubric + model solution + question +
/// student answer, with the §7.3 conciseness rule and the §10 category
/// contract. `question` may be empty (bare-rubric grading falls back to the
/// model solution as the standard).
#[must_use]
pub fn build_grading_prompt(question: &str, rubric: &Rubric, answer: &str) -> String {
    let mut criteria_text = String::new();
    for criterion in &rubric.criteria {
        // `write!` on a `String` never fails; the result is discarded.
        let _ = writeln!(
            criteria_text,
            "- {} (max {}): {}",
            criterion.name, criterion.max_score, criterion.what_good_looks_like
        );
    }
    let question_block = if question.trim().is_empty() {
        "No question text supplied: judge the answer against the rubric and model solution below."
            .to_string()
    } else {
        format!("--- QUESTION ---\n{question}\n--- END QUESTION ---")
    };
    format!(
        "You are grading a closed-book written-assignment answer against a FROZEN rubric created before the answer existed. Never invent criteria: judge only the criteria below, comparing the student's answer to the model solution.\n{CONCISENESS}\n{CATEGORIES}\n{ADVERSARIAL_AUDIT}\n{LENIENCY}\nRespond with a single JSON object and nothing else: {{\"classification\": \"CORRECT\", \"score\": 8, \"criteria_results\": [{{\"name\": \"correctness\", \"score\": 5, \"max_score\": 5, \"comment\": \"...\"}}], \"feedback\": \"...\"}}.\nRules: \"score\" is the total within 0-{max}; \"criteria_results\" covers every rubric criterion exactly once (echo each criterion's \"name\" and \"max_score\" verbatim, score within 0-max, non-empty comment); \"feedback\" is 2-5 sentences of technical feedback grounded in the model solution; a blank student answer is UNANSWERED with score 0; emit raw JSON only (no markdown fences, no commentary); escape newlines inside strings as \\n.\n\n--- RUBRIC (max {max}) ---\n{criteria_text}Model solution: {solution}\n--- END RUBRIC ---\n\n{question_block}\n\n--- STUDENT ANSWER ---\n{answer}\n--- END STUDENT ANSWER ---",
        max = rubric.max_score,
        solution = rubric.model_solution,
    )
}

/// Validate one per-criterion result: named, echoed maximum matches a frozen
/// criterion exactly once (consumed by name+max), score in range, commented.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect.
fn validate_criterion_result(
    raw: &serde_json::Value,
    index: usize,
    remaining: &mut Vec<(&str, i64)>,
) -> Result<CriterionResult> {
    let name = raw
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::LlmFatal(format!("grade criterion {index} needs a name")))?;
    let result_max = raw
        .get("max_score")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            Error::LlmFatal(format!("grade criterion {index} needs integer 'max_score'"))
        })?;
    let result_score = raw
        .get("score")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Error::LlmFatal(format!("grade criterion {index} needs integer 'score'")))?;
    let comment = raw
        .get("comment")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal(format!("grade criterion {index} needs a comment")))?;
    let Some(matched) = remaining
        .iter()
        .position(|(n, m)| *n == name && *m == result_max)
    else {
        return Err(Error::LlmFatal(format!(
            "grade criterion {index} ('{name}' max {result_max}) matches no frozen criterion exactly once"
        )));
    };
    remaining.remove(matched);
    if result_score < 0 || result_score > result_max {
        return Err(Error::LlmFatal(format!(
            "grade criterion {index} score {result_score} outside 0-{result_max}"
        )));
    }
    Ok(CriterionResult {
        name: name.to_string(),
        score: result_score,
        max_score: result_max,
        comment: comment.to_string(),
    })
}

/// Validate one grader response against the frozen rubric: known §10 verdict,
/// total within bounds, per-criterion coverage exactly once with echoed
/// maxima, non-empty comments and feedback.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect (fed back into the
/// fast-retry repair loop by `complete_cached`).
pub fn validate_grade(text: &str, rubric: &Rubric) -> Result<Grade> {
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::LlmFatal(format!("grade is not valid JSON: {e}")))?;
    let classification_label = parsed
        .get("classification")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::LlmFatal("grade missing 'classification'".to_string()))?;
    let classification = GradeClass::parse(classification_label).map_err(|_| {
        Error::LlmFatal(format!(
            "grade has bad classification '{classification_label}'"
        ))
    })?;
    let score = parsed
        .get("score")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Error::LlmFatal("grade missing integer 'score'".to_string()))?;
    if score < 0 || score > rubric.max_score {
        return Err(Error::LlmFatal(format!(
            "grade score {score} outside 0-{}",
            rubric.max_score
        )));
    }
    let raw_results = parsed
        .get("criteria_results")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::LlmFatal("grade missing 'criteria_results' array".to_string()))?;
    // Each frozen criterion must be covered exactly once (matched by
    // name+max, so duplicate names resolve by consumption).
    let mut remaining: Vec<(&str, i64)> = rubric
        .criteria
        .iter()
        .map(|c| (c.name.as_str(), c.max_score))
        .collect();
    let mut criteria_results = Vec::with_capacity(raw_results.len());
    for (index, raw) in raw_results.iter().enumerate() {
        criteria_results.push(validate_criterion_result(raw, index, &mut remaining)?);
    }
    if !remaining.is_empty() {
        return Err(Error::LlmFatal(format!(
            "grade leaves {} rubric criterion/criteria uncovered",
            remaining.len()
        )));
    }
    let feedback = parsed
        .get("feedback")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal("grade missing 'feedback'".to_string()))?;
    Ok(Grade {
        classification,
        score,
        criteria_results,
        feedback: feedback.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rubric() -> Rubric {
        Rubric {
            criteria: vec![
                Criterion {
                    name: "correctness".to_string(),
                    max_score: 5,
                    what_good_looks_like: "Names the address and justifies it.".to_string(),
                },
                Criterion {
                    name: "edge cases".to_string(),
                    max_score: 3,
                    what_good_looks_like: "Handles aliasing.".to_string(),
                },
            ],
            max_score: 8,
            model_solution:
                "The & operator yields the address of its operand; aliasing needs care.".to_string(),
        }
    }

    fn good_grade() -> serde_json::Value {
        serde_json::json!({
            "classification": "CORRECT_BUT_BRIEF",
            "score": 7,
            "criteria_results": [
                {"name": "correctness", "score": 5, "max_score": 5, "comment": "Address named correctly."},
                {"name": "edge cases", "score": 2, "max_score": 3, "comment": "Aliasing mentioned briefly."}
            ],
            "feedback": "Correct core model. Say more about aliasing next time for full credit."
        })
    }

    #[test]
    fn valid_grade_passes() {
        let grade = validate_grade(&good_grade().to_string(), &rubric()).unwrap();
        assert_eq!(grade.classification, GradeClass::CorrectButBrief);
        assert_eq!(grade.score, 7);
        assert_eq!(grade.criteria_results.len(), 2);
        assert_eq!(grade.criteria_results[0].name, "correctness");
    }

    #[test]
    fn class_round_trip() {
        assert_eq!(GradeClass::parse("CORRECT").unwrap(), GradeClass::Correct);
        assert_eq!(
            GradeClass::parse("QUESTION_DEFECTIVE").unwrap(),
            GradeClass::QuestionDefective
        );
        assert!(GradeClass::parse("PERFECT").is_err());
        assert_eq!(GradeClass::Unanswered.as_str(), "UNANSWERED");
    }

    #[test]
    fn bad_classification_fails() {
        let mut grade = good_grade();
        grade["classification"] = serde_json::json!("PERFECT");
        assert!(validate_grade(&grade.to_string(), &rubric()).is_err());
    }

    #[test]
    fn score_bounds_fail() {
        let mut high = good_grade();
        high["score"] = serde_json::json!(9);
        assert!(validate_grade(&high.to_string(), &rubric()).is_err());
        let mut negative = good_grade();
        negative["score"] = serde_json::json!(-1);
        assert!(validate_grade(&negative.to_string(), &rubric()).is_err());
    }

    #[test]
    fn criterion_coverage_gates() {
        // Missing one criterion.
        let mut short = good_grade();
        short["criteria_results"].as_array_mut().unwrap().pop();
        assert!(validate_grade(&short.to_string(), &rubric()).is_err());
        // Extra unknown criterion.
        let mut extra = good_grade();
        extra["criteria_results"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(
                {"name": "style", "score": 1, "max_score": 1, "comment": "Neat."}
            ));
        assert!(validate_grade(&extra.to_string(), &rubric()).is_err());
        // Echoed max must match the frozen rubric (no invented totals).
        let mut wrong_max = good_grade();
        wrong_max["criteria_results"][0]["max_score"] = serde_json::json!(10);
        assert!(validate_grade(&wrong_max.to_string(), &rubric()).is_err());
        // Per-criterion score bounded.
        let mut overflow = good_grade();
        overflow["criteria_results"][0]["score"] = serde_json::json!(6);
        assert!(validate_grade(&overflow.to_string(), &rubric()).is_err());
        // Duplicate coverage of one criterion leaves the other uncovered.
        let mut duplicated = good_grade();
        duplicated["criteria_results"][1] = duplicated["criteria_results"][0].clone();
        assert!(validate_grade(&duplicated.to_string(), &rubric()).is_err());
    }

    #[test]
    fn feedback_and_comment_gates() {
        let mut no_feedback = good_grade();
        no_feedback["feedback"] = serde_json::json!("  ");
        assert!(validate_grade(&no_feedback.to_string(), &rubric()).is_err());
        let mut no_comment = good_grade();
        no_comment["criteria_results"][0]["comment"] = serde_json::json!("");
        assert!(validate_grade(&no_comment.to_string(), &rubric()).is_err());
    }

    #[test]
    fn malformed_json_fails() {
        assert!(validate_grade("{broken", &rubric()).is_err());
        assert!(validate_grade("{\"score\": 1}", &rubric()).is_err());
    }

    #[test]
    fn rubric_parsing_round_trip() {
        let stored = serde_json::json!({
            "criteria": [
                {"name": "correctness", "max_score": 5, "what_good_looks_like": "Names it."},
                {"name": "edge cases", "max_score": 3, "what_good_looks_like": "Handles aliasing."}
            ],
            "max_score": 8,
            "model_solution": "The & operator yields the address of its operand."
        });
        let parsed = parse_rubric_json(&stored.to_string()).unwrap();
        assert_eq!(parsed.max_score, 8);
        assert_eq!(parsed.criteria.len(), 2);
        assert_eq!(parsed.criteria[0].name, "correctness");
        // A grade validated against the parsed rubric passes.
        assert!(validate_grade(&good_grade().to_string(), &parsed).is_ok());
        // Malformed rubrics fail loudly (never silently graded).
        assert!(parse_rubric_json("{\"criteria\": []}").is_err());
        assert!(parse_rubric_json("{broken").is_err());
        assert!(
            parse_rubric_json("{\"criteria\": [{}], \"max_score\": 1, \"model_solution\": \"x\"}")
                .is_err()
        );
    }

    #[test]
    fn prompt_carries_conciseness_and_categories() {
        let prompt = build_grading_prompt("What does &x yield?", &rubric(), "An address.");
        assert!(prompt.contains("Do NOT penalize brevity"));
        assert!(prompt.contains("CORRECT_BUT_BRIEF"));
        assert!(prompt.contains("QUESTION_DEFECTIVE"));
        assert!(prompt.contains("FROZEN rubric"));
        assert!(prompt.contains("What does &x yield?"));
        assert!(prompt.contains("An address."));
        let bare = build_grading_prompt("", &rubric(), "An address.");
        assert!(bare.contains("No question text supplied"));
    }

    #[test]
    fn prompt_orders_adversarial_audit() {
        let prompt = build_grading_prompt("What does &x yield?", &rubric(), "An address.");
        assert!(prompt.contains("steelman"));
        assert!(prompt.contains("NEITHER the student's answer NOR the frozen rubric"));
        assert!(prompt.contains("audit the model solution itself"));
        assert!(prompt.contains("Never fail a student for disagreeing with a wrong rubric"));
    }

    #[test]
    fn prompt_orders_closed_book_leniency() {
        let prompt = build_grading_prompt("What does &x yield?", &rubric(), "An address.");
        assert!(prompt.contains("Never deduct for missing takeaway numbers"));
        assert!(prompt.contains("own words"));
        assert!(prompt.contains("PARTIALLY_CORRECT only when a substantive technical gap remains"));
    }

    #[test]
    fn grade_identity_is_stable_and_sensitive() {
        let first = grade_source_hash("q", "{\"max_score\": 8}", "a");
        assert_eq!(first, grade_source_hash("q", "{\"max_score\": 8}", "a"));
        assert_ne!(
            first,
            grade_source_hash("q", "{\"max_score\": 8}", "revised")
        );
        assert_ne!(first, grade_source_hash("q2", "{\"max_score\": 8}", "a"));
        assert!(grade_params_json_for(&rubric()).contains("\"operation\":\"grade\""));
        assert!(grade_params_json_for(&rubric()).contains("\"max_tokens\":4000"));
        assert!(grade_params_json_for(&rubric()).contains("\"rf\":\""));
    }

    #[test]
    fn grade_schema_bakes_the_rubric() {
        let schema: serde_json::Value =
            serde_json::from_str(&grade_response_schema(&rubric())).unwrap();
        let criteria = schema["properties"]["criteria_results"].clone();
        assert_eq!(criteria["minItems"], serde_json::json!(2));
        assert_eq!(criteria["maxItems"], serde_json::json!(2));
        assert_eq!(
            criteria["prefixItems"][0]["properties"]["name"]["enum"][0],
            serde_json::json!("correctness")
        );
        assert_eq!(
            criteria["prefixItems"][0]["properties"]["max_score"]["enum"][0],
            serde_json::json!(5)
        );
        assert_eq!(
            criteria["prefixItems"][1]["properties"]["score"]["maximum"],
            serde_json::json!(3)
        );
        assert_eq!(
            schema["properties"]["score"]["maximum"],
            serde_json::json!(8)
        );
        assert_eq!(
            schema["properties"]["classification"]["enum"]
                .as_array()
                .unwrap()
                .len(),
            7
        );
        // The params tag tracks the schema: different rubrics, different tags.
        let tag = crate::llm::schema_tag(&grade_response_schema(&rubric()));
        assert!(grade_params_json_for(&rubric()).contains(&tag));
        let mut other = rubric();
        other.max_score = 9;
        assert_ne!(
            grade_params_json_for(&rubric()),
            grade_params_json_for(&other)
        );
    }
}
