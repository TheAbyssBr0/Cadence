//! Independent dispute audit (§9): prompts, validation, identity.
//!
//! A dispute re-examines one graded answer from first principles. The auditor
//! is a fresh call that must not assume the original grade was right: it
//! steelmans the student's answer, audits the frozen rubric and the original
//! verdict alike, and returns `REVISED`, `UPHELD`, or `QUESTION_DEFECTIVE`.
//! `QUESTION_DEFECTIVE` awards full credit and never penalizes the user.
//!
//! Determinism note: the transport ([`crate::llm::HttpLlmProvider::send`])
//! exposes no temperature knob, so there is no `temperature = 0.0` equivalent
//! to set. Determinism here comes from the raw-JSON-only contract plus the
//! validated cache (reruns are stable), and the prompt demands a verdict
//! grounded in the rubric rather than open-ended prose.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

use crate::assignment::Rubric;
use crate::error::{Error, Result};
use crate::store::Misconception;

/// Completion-token cap for dispute audits (rubric + answer + grade are small).
pub const DISPUTE_MAX_TOKENS: u32 = 4_000;

/// Characters of chapter source excerpt included in the audit prompt when the
/// unit file loads (bounds the prompt; the model solution stays primary).
pub const SOURCE_EXCERPT_CHARS: usize = 2_000;

/// Audit decisions (§9). `REVISED` corrects the grade, `UPHELD` keeps it, and
/// `QUESTION_DEFECTIVE` flags a broken question or rubric with full credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisputeAction {
    /// The dispute holds: the grade changes to `final_score`.
    Revised,
    /// The dispute fails: the original grade stands.
    Upheld,
    /// The question or rubric is at fault — full credit, flagged.
    QuestionDefective,
}

impl DisputeAction {
    /// Canonical label emitted by the auditor and accepted by validation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Revised => "REVISED",
            Self::Upheld => "UPHELD",
            Self::QuestionDefective => "QUESTION_DEFECTIVE",
        }
    }

    /// Parse an auditor-emitted label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "REVISED" => Ok(Self::Revised),
            "UPHELD" => Ok(Self::Upheld),
            "QUESTION_DEFECTIVE" => Ok(Self::QuestionDefective),
            other => Err(Error::InvalidInput(format!(
                "unknown dispute action: {other}"
            ))),
        }
    }
}

/// A validated dispute audit for one graded answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisputeResult {
    /// Whether the student's dispute holds.
    pub dispute_valid: bool,
    /// Corrected total (`0 <= score <= rubric max`).
    pub final_score: i64,
    /// Technical explanation of the decision (non-empty).
    pub explanation: String,
    /// §9 decision.
    pub action: DisputeAction,
}

/// Hex SHA-256 over the (question, rubric, answer, grade, dispute) tuple: the
/// identity half that keeps audit cache entries apart when any input changes.
#[must_use]
pub fn dispute_source_hash(
    question: &str,
    rubric_json: &str,
    answer: &str,
    grade_summary: &str,
    dispute_text: &str,
) -> String {
    let mut hasher = Sha256::new();
    for part in [question, rubric_json, answer, grade_summary, dispute_text] {
        hasher.update(part.as_bytes());
        hasher.update([0_u8]);
    }
    hex::encode(hasher.finalize())
}

/// Canonical params JSON for dispute calls (token cap + operation identity +
/// response-schema tag). The tag binds the max-baked schema below, so audits
/// against different rubric totals never share a cache entry.
#[must_use]
pub fn dispute_params_json_for(max_score: i64) -> String {
    let tag = crate::llm::schema_tag(&dispute_response_schema(max_score));
    format!("{{\"max_tokens\":{DISPUTE_MAX_TOKENS},\"operation\":\"dispute\",\"rf\":\"{tag}\"}}")
}

/// JSON Schema for one dispute audit response (constrained decoding): the
/// verdict enum, a max-bounded final score, and the explanation. Action/score
/// consistency (upheld keeps, revised changes, defective takes full credit)
/// stays in the validator: schemas enforce shape, never judgment.
#[must_use]
pub fn dispute_response_schema(max_score: i64) -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "dispute_valid": {"type": "boolean"},
            "final_score": {"type": "integer", "minimum": 0, "maximum": max_score},
            "explanation": {"type": "string"},
            "action": {"type": "string", "enum": ["REVISED", "UPHELD", "QUESTION_DEFECTIVE"]},
        },
        "required": ["dispute_valid", "final_score", "explanation", "action"],
        "additionalProperties": false,
    })
    .to_string()
}

/// Build the audit prompt: frozen rubric + model solution, original grade,
/// student answer, dispute text, and an optional chapter source excerpt. The
/// auditor verifies from first principles and never inherits the original
/// verdict.
#[must_use]
pub fn build_dispute_prompt(
    question: &str,
    rubric: &Rubric,
    answer: &str,
    original_summary: &str,
    dispute_text: &str,
    source_excerpt: Option<&str>,
) -> String {
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
        "No question text supplied: audit against the rubric and model solution below.".to_string()
    } else {
        format!("--- QUESTION ---\n{question}\n--- END QUESTION ---")
    };
    let source_block = match source_excerpt {
        Some(excerpt) if !excerpt.trim().is_empty() => {
            format!("--- CHAPTER SOURCE (excerpt) ---\n{excerpt}\n--- END SOURCE ---")
        }
        _ => {
            "No chapter source excerpt was supplied: audit against the frozen rubric and model solution below.".to_string()
        }
    };
    format!(
        "You are an independent dispute auditor re-examining one graded assignment answer from first principles. Do NOT assume the original grade is correct: verify the frozen rubric, the original verdict, and the student's answer each against the chapter source and basic technical facts.\nAdversarial audit: first, steelman the student's dispute — construct the strongest case that the original grade was wrong (a correct alternative the rubric missed, a misapplied criterion, a model solution that is itself wrong). Only uphold the grade when that case collapses. Second, audit the model solution itself: if it is wrong, self-contradictory, or the criteria contradict it, return QUESTION_DEFECTIVE with full credit instead of defending a broken standard.\nRespond with a single JSON object and nothing else: {{\"dispute_valid\": true, \"final_score\": 8, \"explanation\": \"...\", \"action\": \"REVISED\"}}.\nRules: \"action\" is exactly one of REVISED (the dispute holds — the grade changes), UPHELD (the dispute fails — the original grade stands), QUESTION_DEFECTIVE (the question or rubric is at fault — award full credit, the maximum below); \"final_score\" is an integer within 0-{max}; UPHELD keeps the original score exactly, REVISED changes it, QUESTION_DEFECTIVE awards the maximum; \"dispute_valid\" is true exactly when the action is REVISED or QUESTION_DEFECTIVE; \"explanation\" is 2-5 sentences of technical reasoning grounded in the rubric or source; emit raw JSON only (no markdown fences, no commentary); escape newlines inside strings as \\n.\n\n--- RUBRIC (max {max}) ---\n{criteria_text}Model solution: {solution}\n--- END RUBRIC ---\n\n--- ORIGINAL GRADE ---\n{original_summary}\n--- END ORIGINAL GRADE ---\n\n{question_block}\n\n--- STUDENT ANSWER ---\n{answer}\n--- END STUDENT ANSWER ---\n\n--- DISPUTE ---\n{dispute_text}\n--- END DISPUTE ---\n\n{source_block}",
        max = rubric.max_score,
        solution = rubric.model_solution,
    )
}

/// Validate one auditor response: well-formed JSON, known §9 action,
/// `final_score` within bounds, non-empty explanation, and verdict/score
/// consistency (an upheld dispute keeps the original score, a revision changes
/// it, a defective question earns full credit).
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] describing the defect (fed back into the
/// fast-retry repair loop by `complete_cached`).
pub fn validate_dispute(text: &str, max_score: i64, original_score: i64) -> Result<DisputeResult> {
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::LlmFatal(format!("dispute audit is not valid JSON: {e}")))?;
    let dispute_valid = parsed
        .get("dispute_valid")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            Error::LlmFatal("dispute audit missing boolean 'dispute_valid'".to_string())
        })?;
    let final_score = parsed
        .get("final_score")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            Error::LlmFatal("dispute audit missing integer 'final_score'".to_string())
        })?;
    if final_score < 0 || final_score > max_score {
        return Err(Error::LlmFatal(format!(
            "dispute final_score {final_score} outside 0-{max_score}"
        )));
    }
    let action_label = parsed
        .get("action")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::LlmFatal("dispute audit missing 'action'".to_string()))?;
    let action = DisputeAction::parse(action_label)
        .map_err(|_| Error::LlmFatal(format!("dispute audit has bad action '{action_label}'")))?;
    let explanation = parsed
        .get("explanation")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::LlmFatal("dispute audit missing 'explanation'".to_string()))?;
    // Verdict/score consistency: the flags must agree with the numbers, so a
    // confused auditor cannot silently keep a score it claims to revise.
    if dispute_valid {
        if action == DisputeAction::Upheld {
            return Err(Error::LlmFatal(
                "dispute audit claims dispute_valid=true with action UPHELD".to_string(),
            ));
        }
        if action == DisputeAction::Revised && final_score == original_score {
            return Err(Error::LlmFatal(
                "dispute audit claims REVISED but keeps the original score".to_string(),
            ));
        }
        if action == DisputeAction::QuestionDefective && final_score != max_score {
            return Err(Error::LlmFatal(
                "dispute audit claims QUESTION_DEFECTIVE without full credit".to_string(),
            ));
        }
    } else {
        if action != DisputeAction::Upheld {
            return Err(Error::LlmFatal(
                "dispute audit claims dispute_valid=false without action UPHELD".to_string(),
            ));
        }
        if final_score != original_score {
            return Err(Error::LlmFatal(
                "dispute audit claims UPHELD but changes the score".to_string(),
            ));
        }
    }
    Ok(DisputeResult {
        dispute_valid,
        final_score,
        explanation: explanation.to_string(),
        action,
    })
}

/// Whether a §9 verdict purges misconceptions (§9 step 3): a successful
/// dispute means the grade was wrong, so the `ASSIGNMENT` row it logged and
/// the re-probed targets it nudged down rest on bad evidence. `UPHELD` keeps
/// the grade and purges nothing.
#[must_use]
pub const fn should_purge_misconceptions(action: DisputeAction) -> bool {
    match action {
        DisputeAction::Revised | DisputeAction::QuestionDefective => true,
        DisputeAction::Upheld => false,
    }
}

/// Select the misconception rows a successful dispute purges (§9 step 3):
/// re-probed targets from `targets_json` plus the `ASSIGNMENT` row this
/// question logged (matched by its `Assignment Q<N>: ` concept prefix, where
/// `N = position + 1`). Only open rows (`ACTIVE`/`IMPROVING`) are selected —
/// a wrong grade only ever nudged targets down, so a `RESOLVED` row was
/// resolved on earlier evidence and stays, and `DISPUTED` rows are already
/// purged. Corrupt `targets_json` selects nothing on the target side (the
/// caller reports it); the prefix side still applies.
#[must_use]
pub fn select_dispute_purge_ids(
    targets_json: &str,
    position: i64,
    rows: &[Misconception],
) -> Vec<i64> {
    let targets: Vec<i64> = serde_json::from_str(targets_json).unwrap_or_default();
    let prefix = format!("Assignment Q{}: ", position.saturating_add(1));
    let mut out = Vec::new();
    for row in rows {
        if row.status != "ACTIVE" && row.status != "IMPROVING" {
            continue;
        }
        let targeted = targets.contains(&row.id);
        let logged_here =
            row.source_task == "ASSIGNMENT" && row.concept_description.starts_with(&prefix);
        if targeted || logged_here {
            out.push(row.id);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment::Criterion;

    fn rubric() -> Rubric {
        Rubric {
            criteria: vec![Criterion {
                name: "correctness".to_string(),
                max_score: 5,
                what_good_looks_like: "Explains why one lea cannot emit 5*x.".to_string(),
            }],
            max_score: 5,
            model_solution: "Scale factors are 1, 2, 4, 8 only, so 5*x needs two steps."
                .to_string(),
        }
    }

    fn audit(valid: bool, score: i64, action: &str) -> serde_json::Value {
        serde_json::json!({
            "dispute_valid": valid,
            "final_score": score,
            "explanation": "The SIB byte allows one scaled index, so 4*x consumes it.",
            "action": action
        })
    }

    #[test]
    fn valid_vectors_pass() {
        let upheld = validate_dispute(&audit(false, 2, "UPHELD").to_string(), 5, 2).unwrap();
        assert!(!upheld.dispute_valid);
        assert_eq!(upheld.action, DisputeAction::Upheld);
        let revised = validate_dispute(&audit(true, 5, "REVISED").to_string(), 5, 2).unwrap();
        assert!(revised.dispute_valid);
        assert_eq!(revised.final_score, 5);
        let defective =
            validate_dispute(&audit(true, 5, "QUESTION_DEFECTIVE").to_string(), 5, 2).unwrap();
        assert_eq!(defective.action, DisputeAction::QuestionDefective);
    }

    #[test]
    fn action_round_trip() {
        assert_eq!(
            DisputeAction::parse("REVISED").unwrap(),
            DisputeAction::Revised
        );
        assert_eq!(
            DisputeAction::parse("QUESTION_DEFECTIVE").unwrap(),
            DisputeAction::QuestionDefective
        );
        assert!(DisputeAction::parse("OVERTURNED").is_err());
        assert_eq!(DisputeAction::Upheld.as_str(), "UPHELD");
    }

    #[test]
    fn score_bounds_fail() {
        assert!(validate_dispute(&audit(true, 6, "REVISED").to_string(), 5, 2).is_err());
        assert!(validate_dispute(&audit(true, -1, "REVISED").to_string(), 5, 2).is_err());
    }

    #[test]
    fn boundary_scores_accepted() {
        // Zero is a valid revised score, not "negative".
        let zeroed = validate_dispute(&audit(true, 0, "REVISED").to_string(), 5, 2).unwrap();
        assert_eq!(zeroed.final_score, 0);
    }

    #[test]
    fn verdict_score_consistency_gates() {
        // Valid flag with UPHELD action contradicts itself.
        assert!(validate_dispute(&audit(true, 2, "UPHELD").to_string(), 5, 2).is_err());
        // Invalid flag without UPHELD contradicts itself.
        assert!(validate_dispute(&audit(false, 2, "REVISED").to_string(), 5, 2).is_err());
        // UPHELD must keep the original score exactly.
        assert!(validate_dispute(&audit(false, 3, "UPHELD").to_string(), 5, 2).is_err());
        // REVISED must change the score.
        assert!(validate_dispute(&audit(true, 2, "REVISED").to_string(), 5, 2).is_err());
        // QUESTION_DEFECTIVE awards full credit, never less.
        assert!(validate_dispute(&audit(true, 4, "QUESTION_DEFECTIVE").to_string(), 5, 2).is_err());
    }

    #[test]
    fn shape_gates() {
        assert!(validate_dispute("{broken", 5, 2).is_err());
        let mut missing = audit(true, 5, "REVISED");
        missing.as_object_mut().unwrap().remove("explanation");
        assert!(validate_dispute(&missing.to_string(), 5, 2).is_err());
        let mut blank = audit(true, 5, "REVISED");
        blank["explanation"] = serde_json::json!("  ");
        assert!(validate_dispute(&blank.to_string(), 5, 2).is_err());
        let mut bad_action = audit(true, 5, "REVISED");
        bad_action["action"] = serde_json::json!("OVERTURNED");
        assert!(validate_dispute(&bad_action.to_string(), 5, 2).is_err());
        let mut bad_flag = audit(true, 5, "REVISED");
        bad_flag["dispute_valid"] = serde_json::json!("yes");
        assert!(validate_dispute(&bad_flag.to_string(), 5, 2).is_err());
    }

    #[test]
    fn prompt_carries_audit_contract() {
        let prompt = build_dispute_prompt(
            "Can one lea emit 5*x+y+12?",
            &rubric(),
            "No: scales are 1, 2, 4, 8.",
            "INCORRECT 2/5: claimed one lea suffices.",
            "Scales exclude 5, so one lea is impossible.",
            Some("SIB-index scales encode 1, 2, 4, 8."),
        );
        assert!(prompt.contains("Do NOT assume the original grade is correct"));
        assert!(prompt.contains("steelman"));
        assert!(prompt.contains("QUESTION_DEFECTIVE"));
        assert!(prompt.contains("\"dispute_valid\""));
        assert!(prompt.contains("Can one lea emit 5*x+y+12?"));
        assert!(prompt.contains("Scales exclude 5"));
        assert!(prompt.contains("SIB-index scales encode"));
        let bare = build_dispute_prompt("", &rubric(), "No.", "INCORRECT 0/5.", "Wrong.", None);
        assert!(bare.contains("No question text supplied"));
        assert!(bare.contains("No chapter source excerpt was supplied"));
        // Whitespace-only excerpts count as absent, not as source.
        let blank_excerpt = build_dispute_prompt(
            "Can one lea emit 5*x+y+12?",
            &rubric(),
            "No.",
            "INCORRECT 0/5.",
            "Wrong.",
            Some("   "),
        );
        assert!(blank_excerpt.contains("No chapter source excerpt was supplied"));
        assert!(!blank_excerpt.contains("CHAPTER SOURCE"));
    }

    #[test]
    fn audit_identity_is_stable_and_sensitive() {
        let first = dispute_source_hash("q", "r", "a", "g", "d");
        assert_eq!(first, dispute_source_hash("q", "r", "a", "g", "d"));
        assert_ne!(
            first,
            dispute_source_hash("q", "r", "a", "g", "revised dispute")
        );
        assert_ne!(first, dispute_source_hash("q", "r", "revised", "g", "d"));
        assert!(dispute_params_json_for(5).contains("\"operation\":\"dispute\""));
        assert!(dispute_params_json_for(5).contains("\"max_tokens\":4000"));
        assert!(dispute_params_json_for(5).contains("\"rf\":\""));
    }

    #[test]
    fn purge_predicate_fires_on_successful_disputes_only() {
        assert!(should_purge_misconceptions(DisputeAction::Revised));
        assert!(should_purge_misconceptions(
            DisputeAction::QuestionDefective
        ));
        assert!(!should_purge_misconceptions(DisputeAction::Upheld));
    }

    fn purge_row(id: i64, concept: &str, source_task: &str, status: &str) -> Misconception {
        Misconception {
            id,
            chapter_id: 1,
            concept_description: concept.to_string(),
            description: "description".to_string(),
            evidence: "evidence".to_string(),
            source_task: source_task.to_string(),
            status: status.to_string(),
            confidence: 0.5,
            created_at: "2026-09-25".to_string(),
            updated_at: "2026-09-25".to_string(),
            resolved_at: None,
        }
    }

    #[test]
    fn purge_selects_targets_and_logged_row() {
        let rows = vec![
            purge_row(3, "aliasing", "RETEST", "ACTIVE"),
            purge_row(4, "Assignment Q1: Explain &x", "ASSIGNMENT", "ACTIVE"),
            purge_row(5, "unrelated trap", "RETEST", "IMPROVING"),
        ];
        let ids = select_dispute_purge_ids("[3]", 0, &rows);
        assert_eq!(ids, vec![3, 4]);
    }

    #[test]
    fn purge_skips_resolved_disputed_and_other_questions() {
        let rows = vec![
            purge_row(3, "aliasing", "RETEST", "RESOLVED"),
            purge_row(4, "Assignment Q1: Explain &x", "ASSIGNMENT", "DISPUTED"),
            purge_row(
                5,
                "Assignment Q2: Explain lifetimes",
                "ASSIGNMENT",
                "ACTIVE",
            ),
            purge_row(6, "Assignment Q1: Explain &x", "RETEST", "ACTIVE"),
        ];
        // Target 3 is resolved on earlier evidence — untouched by the bad
        // grade, so it stays. Row 4 is already purged. Row 5 belongs to Q2.
        // Row 6 wears the prefix but is not an assignment log.
        let ids = select_dispute_purge_ids("[3]", 0, &rows);
        assert!(ids.is_empty(), "got {ids:?}");
    }

    #[test]
    fn purge_survives_corrupt_targets() {
        let rows = vec![
            purge_row(3, "aliasing", "RETEST", "ACTIVE"),
            purge_row(4, "Assignment Q1: Explain &x", "ASSIGNMENT", "ACTIVE"),
        ];
        let ids = select_dispute_purge_ids("{broken", 0, &rows);
        assert_eq!(ids, vec![4]);
        let empty = select_dispute_purge_ids("[3]", 0, &[]);
        assert!(empty.is_empty(), "got {empty:?}");
    }
    #[test]
    fn audit_schema_bounds_final_score() {
        let schema: serde_json::Value = serde_json::from_str(&dispute_response_schema(8)).unwrap();
        assert_eq!(
            schema["properties"]["final_score"]["maximum"],
            serde_json::json!(8)
        );
        assert_eq!(
            schema["properties"]["action"]["enum"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let tag = crate::llm::schema_tag(&dispute_response_schema(8));
        assert!(dispute_params_json_for(8).contains(&tag));
        assert_ne!(dispute_params_json_for(8), dispute_params_json_for(5));
    }
}
