//! Misconception lifecycle (§12): confidence deltas, status bands, logging rules.
//!
//! Confidence moves with evidence. Wrong answers push it down — a little on
//! retests, substantially on assignments (which demand active recall plus
//! synthesis); correct answers move it up symmetrically. Past the resolve
//! threshold a row flips to `RESOLVED` and the caller stamps `resolved_at`.
//! Statuses: `ACTIVE` (low) → `IMPROVING` (mid) → `RESOLVED` (past
//! threshold). `DISPUTED` belongs to the dispute purge (§9) and is never
//! produced here; resolved rows never reopen here either — a fresh wrong
//! answer logs a fresh row, and callers only route open rows through
//! [`apply_outcome`].
//!
//! The engine is pure: callers map store rows through [`apply_outcome`] and
//! persist with `Store::update_misconception`.

use crate::grading::GradeClass;

/// Confidence of a freshly logged misconception (mirrors the store default).
pub const INITIAL_CONFIDENCE: f64 = 0.5;
/// Retest evidence moves confidence a little (§12: small MCQ nudges).
pub const RETEST_DELTA: f64 = 0.1;
/// Assignment evidence moves confidence substantially (§12: recall+synthesis).
pub const ASSIGNMENT_DELTA: f64 = 0.25;
/// Confidence at or above this resolves the row.
pub const RESOLVED_THRESHOLD: f64 = 0.85;
/// Confidence at or above this (but below resolved) marks improvement.
pub const IMPROVING_THRESHOLD: f64 = 0.6;

/// One lifecycle step: the new confidence, its status band, and whether this
/// step resolved the row (callers stamp `resolved_at` only then, preserving
/// the original timestamp on later nudges).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition {
    /// Clamped `0.0–1.0` confidence after the delta.
    pub confidence: f64,
    /// `ACTIVE` | `IMPROVING` | `RESOLVED` band for the new confidence.
    pub status: &'static str,
    /// True only when this step flips an open row to `RESOLVED`.
    pub just_resolved: bool,
}

/// Status band for a confidence value (§12 thresholds, inclusive floors).
#[must_use]
pub const fn status_for(confidence: f64) -> &'static str {
    if confidence >= RESOLVED_THRESHOLD {
        "RESOLVED"
    } else if confidence >= IMPROVING_THRESHOLD {
        "IMPROVING"
    } else {
        "ACTIVE"
    }
}

/// Apply one evidence step: `correct` moves confidence up, wrong moves it
/// down; `major` selects the assignment-sized delta over the retest nudge.
/// The result is clamped to `0.0–1.0` (NaN input clamps to `0.0` — a corrupt
/// confidence degrades to the floor, never propagates).
#[must_use]
pub fn apply_outcome(confidence: f64, status: &str, correct: bool, major: bool) -> Transition {
    let delta = if major { ASSIGNMENT_DELTA } else { RETEST_DELTA };
    let stepped = if correct {
        confidence + delta
    } else {
        confidence - delta
    };
    // `clamp` passes NaN through, so floor corrupt input explicitly: a NaN
    // confidence degrades to the floor, never propagates.
    let next = if stepped.is_nan() {
        0.0
    } else {
        stepped.clamp(0.0, 1.0)
    };
    let next_status = status_for(next);
    Transition {
        confidence: next,
        status: next_status,
        just_resolved: next_status == "RESOLVED" && status != "RESOLVED",
    }
}

/// Whether a §10 verdict logs a fresh `ASSIGNMENT` misconception (§12):
/// wrong answers carry evidence of a misconception; correct answers boost
/// re-probed rows instead; blanks and defective questions carry no signal.
#[must_use]
pub const fn should_log_assignment_misconception(classification: GradeClass) -> bool {
    match classification {
        GradeClass::Incorrect
        | GradeClass::PartiallyCorrect
        | GradeClass::Ambiguous => true,
        GradeClass::Correct
        | GradeClass::CorrectButBrief
        | GradeClass::Unanswered
        | GradeClass::QuestionDefective => false,
    }
}

/// Whether a §10 verdict counts as mastering the question (§12): full-credit
/// verdicts boost the confidence of re-probed rows substantially.
#[must_use]
pub const fn is_assignment_correct(classification: GradeClass) -> bool {
    match classification {
        GradeClass::Correct | GradeClass::CorrectButBrief => true,
        GradeClass::PartiallyCorrect
        | GradeClass::Incorrect
        | GradeClass::Unanswered
        | GradeClass::Ambiguous
        | GradeClass::QuestionDefective => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn status_bands_hit_inclusive_floors() {
        assert_eq!(status_for(0.0), "ACTIVE");
        assert_eq!(status_for(0.599_999), "ACTIVE");
        assert_eq!(status_for(0.6), "IMPROVING");
        assert_eq!(status_for(0.849_999), "IMPROVING");
        assert_eq!(status_for(0.85), "RESOLVED");
        assert_eq!(status_for(1.0), "RESOLVED");
    }

    #[test]
    fn retest_steps_are_small() {
        let down = apply_outcome(0.5, "ACTIVE", false, false);
        approx(down.confidence, 0.4);
        assert_eq!(down.status, "ACTIVE");
        assert!(!down.just_resolved);
        let up = apply_outcome(0.5, "ACTIVE", true, false);
        approx(up.confidence, 0.6);
        assert_eq!(up.status, "IMPROVING");
        assert!(!up.just_resolved);
    }

    #[test]
    fn assignment_steps_are_substantial() {
        let down = apply_outcome(0.5, "ACTIVE", false, true);
        approx(down.confidence, 0.25);
        assert_eq!(down.status, "ACTIVE");
        let up = apply_outcome(0.5, "ACTIVE", true, true);
        approx(up.confidence, 0.75);
        assert_eq!(up.status, "IMPROVING");
        assert!(!up.just_resolved);
    }

    #[test]
    fn resolution_fires_once_at_threshold() {
        let resolving = apply_outcome(0.7, "IMPROVING", true, true);
        approx(resolving.confidence, 0.95);
        assert_eq!(resolving.status, "RESOLVED");
        assert!(resolving.just_resolved);
        // A later nudge on an already-resolved row never re-fires.
        let again = apply_outcome(0.95, "RESOLVED", true, true);
        assert_eq!(again.status, "RESOLVED");
        assert!(!again.just_resolved);
    }

    #[test]
    fn confidence_clamps_instead_of_escaping() {
        let top = apply_outcome(1.0, "RESOLVED", true, true);
        approx(top.confidence, 1.0);
        assert_eq!(top.status, "RESOLVED");
        let bottom = apply_outcome(0.0, "ACTIVE", false, true);
        approx(bottom.confidence, 0.0);
        assert_eq!(bottom.status, "ACTIVE");
        assert!(!bottom.just_resolved);
        let corrupt = apply_outcome(f64::NAN, "ACTIVE", false, false);
        approx(corrupt.confidence, 0.0);
        assert_eq!(corrupt.status, "ACTIVE");
    }

    #[test]
    fn assignment_logging_covers_wrong_verdicts_only() {
        assert!(should_log_assignment_misconception(GradeClass::Incorrect));
        assert!(should_log_assignment_misconception(
            GradeClass::PartiallyCorrect
        ));
        assert!(should_log_assignment_misconception(GradeClass::Ambiguous));
        assert!(!should_log_assignment_misconception(GradeClass::Correct));
        assert!(!should_log_assignment_misconception(
            GradeClass::CorrectButBrief
        ));
        assert!(!should_log_assignment_misconception(GradeClass::Unanswered));
        assert!(!should_log_assignment_misconception(
            GradeClass::QuestionDefective
        ));
    }

    #[test]
    fn assignment_correct_is_full_credit_only() {
        assert!(is_assignment_correct(GradeClass::Correct));
        assert!(is_assignment_correct(GradeClass::CorrectButBrief));
        assert!(!is_assignment_correct(GradeClass::PartiallyCorrect));
        assert!(!is_assignment_correct(GradeClass::Incorrect));
        assert!(!is_assignment_correct(GradeClass::Unanswered));
        assert!(!is_assignment_correct(GradeClass::Ambiguous));
        assert!(!is_assignment_correct(GradeClass::QuestionDefective));
    }
}
