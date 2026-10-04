//! Manual cumulative review (§12): targeted retests over open misconceptions.
//!
//! `cadence review` pulls every open (`ACTIVE` / `IMPROVING`) misconception
//! across completed chapters — plus skipped chapters, whose rows stay open
//! under review (§4.1) — and generates one maximum-difficulty MCQ per row
//! (capped at 12 per chapter). Selection and cache identity live here; they
//! are pure over caller-mapped views (the engine never touches the store).
//! Generation runs through the shared MCQ pipeline with a review prompt
//! ([`crate::mcq::build_review_prompt`]), and answers route back to rows by
//! verbatim concept topic ([`crate::mcq::check_review_topics`] enforces it).

use sha2::{Digest, Sha256};

use crate::domain::ChapterStatus;
use crate::mcq::{ReviewConcept, MAX_QUESTIONS};

/// Completion-token cap for review generation calls (a full 12-Q JSON must
/// fit — mirrors the MCQ pipeline cap).
pub const REVIEW_MAX_TOKENS: u32 = 8_000;

/// One misconception row as review selection sees it (mapped from the store
/// by the caller).
#[derive(Debug, Clone)]
pub struct MisconceptionView {
    /// Misconception row id.
    pub id: i64,
    /// Short concept label (the required verbatim `topic`).
    pub concept: String,
    /// Fuller description of the wrong belief.
    pub description: String,
    /// The user's wrong answer / explanation.
    pub evidence: String,
    /// `ACTIVE` | `IMPROVING` | `RESOLVED` | `DISPUTED`.
    pub status: String,
    /// Confidence 0.0–1.0.
    pub confidence: f64,
}

/// One chapter with its misconception rows.
#[derive(Debug, Clone)]
pub struct ChapterOpen {
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Chapter title (display only).
    pub title: String,
    /// Lifecycle state (gates eligibility).
    pub status: ChapterStatus,
    /// All misconception rows for the chapter.
    pub rows: Vec<MisconceptionView>,
}

/// One selected re-probe target: an open row on a review-eligible chapter.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewTarget {
    /// Owning chapter id (routes generation + answers).
    pub chapter_id: i64,
    /// Chapter title (display only).
    pub title: String,
    /// Misconception row id.
    pub id: i64,
    /// Short concept label (emitted verbatim as the question `topic`).
    pub concept: String,
    /// Fuller description of the wrong belief.
    pub description: String,
    /// The user's wrong answer / explanation (trap material).
    pub evidence: String,
    /// Confidence at selection time.
    pub confidence: f64,
}

/// Whether a row is still open (§12): `ACTIVE` or `IMPROVING`. `RESOLVED`
/// rows stay cleared and `DISPUTED` rows belong to the §9 purge — neither is
/// ever re-probed.
#[must_use]
pub fn is_open(status: &str) -> bool {
    status == "ACTIVE" || status == "IMPROVING"
}

/// Whether a chapter's rows are review-eligible: completed chapters (§12),
/// plus skipped chapters whose open rows stay under review (§4.1).
/// Mid-pipeline chapters are excluded — their open rows get the assignment
/// re-probe, not the cumulative review.
#[must_use]
pub const fn is_eligible(status: ChapterStatus) -> bool {
    matches!(
        status,
        ChapterStatus::Completed | ChapterStatus::Skipped
    )
}

/// Select re-probe targets across chapters (§12): open rows on eligible
/// chapters, weakest confidence first per chapter (ties by row id),
/// capped at [`MAX_QUESTIONS`] per chapter. Input chapter order is
/// preserved (callers pass book/index order).
#[must_use]
pub fn select_targets(chapters: &[ChapterOpen]) -> Vec<ReviewTarget> {
    let mut out = Vec::new();
    for chapter in chapters {
        if !is_eligible(chapter.status) {
            continue;
        }
        let mut open: Vec<&MisconceptionView> =
            chapter.rows.iter().filter(|row| is_open(&row.status)).collect();
        open.sort_by(|a, b| {
            a.confidence
                .total_cmp(&b.confidence)
                .then_with(|| a.id.cmp(&b.id))
        });
        for row in open.into_iter().take(MAX_QUESTIONS) {
            out.push(ReviewTarget {
                chapter_id: chapter.chapter_id,
                title: chapter.title.clone(),
                id: row.id,
                concept: row.concept.clone(),
                description: row.description.clone(),
                evidence: row.evidence.clone(),
                confidence: row.confidence,
            });
        }
    }
    out
}

/// Questions to generate for `open` targets: one per misconception, within
/// `1..=MAX_QUESTIONS` (callers only invoke generation on non-empty
/// selections; the floor keeps a zero count from becoming an empty prompt).
#[must_use]
pub fn review_question_count(open: usize) -> usize {
    open.clamp(1, MAX_QUESTIONS)
}

/// Hex SHA-256 over the target set: the identity half that keeps cache
/// entries apart when the same chapter is reviewed under different open
/// sets (mirrors the assignment `misconceptions_hash`).
#[must_use]
pub fn targets_hash(targets: &[ReviewTarget]) -> String {
    let mut hasher = Sha256::new();
    for target in targets {
        hasher.update(target.id.to_string().as_bytes());
        hasher.update(b":");
        hasher.update(target.concept.as_bytes());
        hasher.update(b";");
    }
    hex::encode(hasher.finalize())
}

/// Canonical params JSON for review calls (token cap + target-set identity +
/// response-schema tag).
#[must_use]
pub fn review_params_json(count: usize, targets_hash: &str) -> String {
    let tag = crate::llm::schema_tag(&crate::mcq::mcq_review_response_schema());
    format!(
        "{{\"max_tokens\":{REVIEW_MAX_TOKENS},\"operation\":\"review\",\"phase\":\"review\",\"review_count\":{count},\"misconceptions_hash\":\"{targets_hash}\",\"rf\":\"{tag}\"}}"
    )
}

/// Map targets to the prompt/validator concept view.
#[must_use]
pub fn concepts_of(targets: &[ReviewTarget]) -> Vec<ReviewConcept> {
    targets
        .iter()
        .map(|target| ReviewConcept {
            id: target.id,
            concept: target.concept.clone(),
            description: target.description.clone(),
            evidence: target.evidence.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: i64, concept: &str, status: &str, confidence: f64) -> MisconceptionView {
        MisconceptionView {
            id,
            concept: concept.to_string(),
            description: format!("belief {concept}"),
            evidence: format!("evidence {concept}"),
            status: status.to_string(),
            confidence,
        }
    }

    fn chapter(id: i64, status: ChapterStatus, rows: Vec<MisconceptionView>) -> ChapterOpen {
        ChapterOpen {
            chapter_id: id,
            title: format!("Ch {id}"),
            status,
            rows,
        }
    }

    #[test]
    fn eligibility_covers_completed_and_skipped_only() {
        assert!(is_eligible(ChapterStatus::Completed));
        assert!(is_eligible(ChapterStatus::Skipped));
        assert!(!is_eligible(ChapterStatus::Locked));
        assert!(!is_eligible(ChapterStatus::PretestReady));
        assert!(!is_eligible(ChapterStatus::ReadComplete));
        assert!(!is_eligible(ChapterStatus::RetestComplete));
        assert!(!is_eligible(ChapterStatus::AssignmentComplete));
    }

    #[test]
    fn openness_excludes_resolved_disputed_and_unknown() {
        assert!(is_open("ACTIVE"));
        assert!(is_open("IMPROVING"));
        assert!(!is_open("RESOLVED"));
        assert!(!is_open("DISPUTED"));
        assert!(!is_open("STALE"));
    }

    #[test]
    fn selection_orders_weakest_first_and_skips_closed() {
        let chapters = vec![chapter(
            1,
            ChapterStatus::Completed,
            vec![
                row(1, "strong", "IMPROVING", 0.7),
                row(2, "weak", "ACTIVE", 0.3),
                row(3, "done", "RESOLVED", 0.9),
                row(4, "purged", "DISPUTED", 0.5),
                row(5, "weaker", "ACTIVE", 0.2),
            ],
        )];
        let targets = select_targets(&chapters);
        assert_eq!(
            targets.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![5, 2, 1]
        );
        assert_eq!(targets[0].chapter_id, 1);
        assert_eq!(targets[0].title, "Ch 1");
    }

    #[test]
    fn selection_tie_breaks_by_row_id_and_caps_per_chapter() {
        let rows: Vec<MisconceptionView> = (0..15)
            .map(|i| row(100 + i, &format!("c{i}"), "ACTIVE", 0.5))
            .collect();
        let chapters = vec![chapter(1, ChapterStatus::Skipped, rows)];
        let targets = select_targets(&chapters);
        assert_eq!(targets.len(), MAX_QUESTIONS);
        // Equal confidence → lowest ids survive the cap.
        assert_eq!(targets.last().map(|t| t.id), Some(100 + 11));
    }

    #[test]
    fn selection_skips_live_chapters_and_keeps_order() {
        let chapters = vec![
            chapter(1, ChapterStatus::Completed, vec![row(1, "a", "ACTIVE", 0.5)]),
            chapter(
                2,
                ChapterStatus::RetestComplete,
                vec![row(2, "b", "ACTIVE", 0.1)],
            ),
            chapter(3, ChapterStatus::Skipped, vec![row(3, "c", "ACTIVE", 0.4)]),
            chapter(4, ChapterStatus::Locked, vec![row(4, "d", "ACTIVE", 0.1)]),
        ];
        let targets = select_targets(&chapters);
        assert_eq!(
            targets.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn selection_empty_without_open_rows() {
        let chapters = vec![
            chapter(1, ChapterStatus::Completed, vec![]),
            chapter(
                2,
                ChapterStatus::Completed,
                vec![row(1, "done", "RESOLVED", 0.9)],
            ),
        ];
        assert_eq!(select_targets(&chapters).len(), 0);
        assert_eq!(select_targets(&[]).len(), 0);
    }

    #[test]
    fn question_count_clamps_to_range() {
        assert_eq!(review_question_count(0), 1);
        assert_eq!(review_question_count(1), 1);
        assert_eq!(review_question_count(5), 5);
        assert_eq!(review_question_count(MAX_QUESTIONS), MAX_QUESTIONS);
        assert_eq!(review_question_count(99), MAX_QUESTIONS);
    }

    #[test]
    fn target_hash_is_stable_and_sensitive() {
        let chapters = vec![chapter(1, ChapterStatus::Completed, vec![row(1, "a", "ACTIVE", 0.5)])];
        let targets = select_targets(&chapters);
        let again = select_targets(&chapters);
        assert_eq!(targets_hash(&targets), targets_hash(&again));
        let mut renamed = targets.clone();
        renamed[0].concept = "b".to_string();
        assert_ne!(targets_hash(&targets), targets_hash(&renamed));
    }

    #[test]
    fn params_carry_review_identity() {
        let params = review_params_json(3, "abc123");
        assert!(params.contains("\"operation\":\"review\""));
        assert!(params.contains("\"phase\":\"review\""));
        assert!(params.contains("\"review_count\":3"));
        assert!(params.contains("\"misconceptions_hash\":\"abc123\""));
        assert!(params.contains("\"max_tokens\":8000"));
        assert!(params.contains("\"rf\":\""));
    }

    #[test]
    fn concepts_map_verbatim() {
        let chapters = vec![chapter(
            1,
            ChapterStatus::Completed,
            vec![row(7, "addresses", "ACTIVE", 0.4)],
        )];
        let targets = select_targets(&chapters);
        let concepts = concepts_of(&targets);
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].id, 7);
        assert_eq!(concepts[0].concept, "addresses");
        assert!(concepts[0].description.contains("belief addresses"));
    }
}
