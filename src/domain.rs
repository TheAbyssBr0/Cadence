//! Domain types and the unit learning state machine (§4).
//!
//! The scheduler creates tasks only when prerequisites are met; these types
//! encode the legal transitions structurally so future tasks cannot execute
//! early. All transitions are pure and time-free.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A registered book.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Book {
    /// Row id (0 = not yet persisted).
    pub id: i64,
    /// Display title.
    pub title: String,
    /// Original PDF path.
    pub filepath: String,
    /// Hex SHA-256 of the PDF bytes.
    pub file_hash: String,
    /// Mandatory: pages before this (one-based) are `FRONT_MATTER`.
    pub start_page: i64,
    /// RFC 3339 / ISO date string.
    pub created_at: String,
}

/// Lifecycle of a study unit (chapter or subchapter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChapterStatus {
    /// Registered but pretest not yet available.
    Locked,
    /// Pretest may be taken.
    PretestReady,
    /// Pretest done; reading unlocked.
    PretestComplete,
    /// Reading unlocked, not yet finished.
    ReadAvailable,
    /// Reading done; retest unlocked (next calendar day).
    ReadComplete,
    /// Retest done; assignment unlocked.
    RetestComplete,
    /// Assignment done; notes generated.
    AssignmentComplete,
    /// Fully finished.
    Completed,
    /// Skipped by the user (§4.1); generates no tasks, excluded from metrics.
    /// Returns only via `unskip` to `PRETEST_READY` on a fresh attempt.
    Skipped,
}

impl ChapterStatus {
    /// Human-readable label stored in SQLite.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Locked => "LOCKED",
            Self::PretestReady => "PRETEST_READY",
            Self::PretestComplete => "PRETEST_COMPLETE",
            Self::ReadAvailable => "READ_AVAILABLE",
            Self::ReadComplete => "READ_COMPLETE",
            Self::RetestComplete => "RETEST_COMPLETE",
            Self::AssignmentComplete => "ASSIGNMENT_COMPLETE",
            Self::Completed => "COMPLETED",
            Self::Skipped => "SKIPPED",
        }
    }

    /// Parse a stored label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "LOCKED" => Ok(Self::Locked),
            "PRETEST_READY" => Ok(Self::PretestReady),
            "PRETEST_COMPLETE" => Ok(Self::PretestComplete),
            "READ_AVAILABLE" => Ok(Self::ReadAvailable),
            "READ_COMPLETE" => Ok(Self::ReadComplete),
            "RETEST_COMPLETE" => Ok(Self::RetestComplete),
            "ASSIGNMENT_COMPLETE" => Ok(Self::AssignmentComplete),
            "COMPLETED" => Ok(Self::Completed),
            "SKIPPED" => Ok(Self::Skipped),
            other => Err(Error::InvalidInput(format!(
                "unknown chapter status: {other}"
            ))),
        }
    }

    /// Legal successor after completing `task`, or an error describing why the
    /// task cannot run in this state. This is the single enforcement point for
    /// the §4 invariant: `RETEST` requires `READ` complete, `ASSIGNMENT`
    /// requires `RETEST` complete.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransition`] when `task` is not legal now.
    pub fn complete(self, task: TaskType) -> Result<Self> {
        match (self, task) {
            (Self::PretestReady, TaskType::Pretest) => Ok(Self::PretestComplete),
            (Self::PretestComplete, TaskType::Read) => Ok(Self::ReadAvailable),
            // Reading is a stage: entering it moves to READ_AVAILABLE is done at
            // scheduling time (see `open_reading`); completing it moves on.
            (Self::ReadAvailable, TaskType::Read) => Ok(Self::ReadComplete),
            (Self::ReadComplete, TaskType::Retest) => Ok(Self::RetestComplete),
            (Self::RetestComplete, TaskType::AssignmentWrite) => Ok(Self::AssignmentComplete),
            (Self::AssignmentComplete, TaskType::Notes) => Ok(Self::Completed),
            _ => Err(Error::InvalidTransition(format!(
                "cannot complete {task:?} while chapter is {}",
                self.as_str()
            ))),
        }
    }

    /// The next task type the scheduler may create for this status, if any.
    /// `Skipped` (and terminal states) yield nothing: skipped chapters
    /// generate no tasks and are invisible to the 3-day window (§4.1).
    #[must_use]
    pub const fn next_task(self) -> Option<TaskType> {
        match self {
            Self::PretestReady => Some(TaskType::Pretest),
            Self::PretestComplete | Self::ReadAvailable => Some(TaskType::Read),
            Self::ReadComplete => Some(TaskType::Retest),
            Self::RetestComplete => Some(TaskType::AssignmentWrite),
            Self::AssignmentComplete => Some(TaskType::Notes),
            Self::Locked | Self::Completed | Self::Skipped => None,
        }
    }

    /// Whether the chapter is skipped (excluded from metrics denominators).
    #[must_use]
    pub const fn is_skipped(self) -> bool {
        matches!(self, Self::Skipped)
    }
}

/// A study unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chapter {
    /// Row id (0 = not yet persisted).
    pub id: i64,
    /// Owning book id.
    pub book_id: i64,
    /// Zero-based position within the book.
    pub index_in_book: i64,
    /// Outline depth (1 = chapter, 2+ = subchapter).
    pub level: i64,
    /// Heading text.
    pub title: String,
    /// One-based physical PDF pages, inclusive.
    pub start_page: i64,
    /// One-based physical PDF pages, inclusive.
    pub end_page: i64,
    /// JSON file holding sanitized text + provenance.
    pub file_path: String,
    /// Lifecycle state.
    pub status: ChapterStatus,
    /// Attempt number (§4.1): starts at 1, increments on every `unskip`.
    /// Metrics only consider rows matching the chapter's current attempt.
    pub attempt_no: i64,
}

/// Schedulable work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Day N moderate MCQ.
    Pretest,
    /// Day N reading stage.
    Read,
    /// Day N+1 hardest MCQ.
    Retest,
    /// Day N+2 closed-book written work.
    AssignmentWrite,
    /// Post-grading synthesis notes.
    Notes,
}

impl TaskType {
    /// Human-readable label stored in SQLite.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pretest => "PRETEST",
            Self::Read => "READ",
            Self::Retest => "RETEST",
            Self::AssignmentWrite => "ASSIGNMENT_WRITE",
            Self::Notes => "NOTES",
        }
    }

    /// Parse a stored label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "PRETEST" => Ok(Self::Pretest),
            "READ" => Ok(Self::Read),
            "RETEST" => Ok(Self::Retest),
            "ASSIGNMENT_WRITE" => Ok(Self::AssignmentWrite),
            "NOTES" => Ok(Self::Notes),
            other => Err(Error::InvalidInput(format!("unknown task type: {other}"))),
        }
    }
}

/// Lifecycle of a task row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Scheduled, prerequisites met, awaiting execution.
    Pending,
    /// Finished; never recomputed or rerun.
    Done,
}

impl TaskStatus {
    /// Human-readable label stored in SQLite.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "PENDING",
            Self::Done => "DONE",
        }
    }

    /// Parse a stored label.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] on unknown labels.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "PENDING" => Ok(Self::Pending),
            "DONE" => Ok(Self::Done),
            other => Err(Error::InvalidInput(format!("unknown task status: {other}"))),
        }
    }
}

/// A scheduled task. `scheduled_for` is a calendar date (`YYYY-MM-DD`),
/// never a session count (§5).
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// Row id (0 = not yet persisted).
    pub id: i64,
    /// Owning book id.
    pub book_id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Kind of work.
    pub task_type: TaskType,
    /// Calendar date the task is due.
    pub scheduled_for: NaiveDate,
    /// Lifecycle state.
    pub status: TaskStatus,
    /// Completion timestamp, if done.
    pub completed_at: Option<String>,
    /// Global creation order for stable tie-breaking.
    pub sequence: i64,
    /// Attempt number copied from the chapter at creation (§4.1). Metrics
    /// only consider tasks matching the chapter's current `attempt_no`;
    /// rows from abandoned attempts stay as an audit trail.
    pub attempt_no: i64,
}

/// Validate a new book registration: `--start-page` is mandatory and
/// one-based (§3).
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when `start_page` is below 1.
pub fn validate_book_registration(start_page: i64) -> Result<()> {
    if start_page < 1 {
        return Err(Error::InvalidInput(
            "start-page is mandatory and must be >= 1 (one-based physical page)".to_string(),
        ));
    }
    Ok(())
}

/// Validate a chapter page range: one-based, inclusive, within the book.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on inverted or non-positive ranges.
pub fn validate_page_range(start_page: i64, end_page: i64) -> Result<()> {
    if start_page < 1 || end_page < 1 {
        return Err(Error::InvalidInput(
            "chapter pages are one-based and must be >= 1".to_string(),
        ));
    }
    if end_page < start_page {
        return Err(Error::InvalidInput(format!(
            "chapter end page ({end_page}) precedes start page ({start_page})"
        )));
    }
    Ok(())
}

/// Validate the 50-page unit cap (§3 / §8.2).
#[must_use]
pub fn unit_page_count(start_page: i64, end_page: i64) -> Option<i64> {
    if start_page < 1 || end_page < start_page {
        return None;
    }
    // end >= start >= 1, so `end - start + 1 >= 1`; use checked ops to
    // satisfy `arithmetic_side_effects` without panicking.
    end_page
        .checked_sub(start_page)
        .and_then(|span| span.checked_add(1))
}

/// Check whether a unit fits the page cap.
#[must_use]
pub fn unit_fits_cap(start_page: i64, end_page: i64, max_unit_pages: i64) -> bool {
    unit_page_count(start_page, end_page).is_some_and(|count| count <= max_unit_pages)
}

/// Presentation-only hint for the `skip` listing (§4.1): `true` marks units
/// that are *likely* meta (level-intro overviews, back matter) with a `~`
/// flag. The app never auto-skips on this signal.
///
/// Heuristic: short level-1 units (≤ 8 pages) or index/bibliography-style
/// titles (`index`, `bibliography`, `appendix`, `glossary`, `references`,
/// `acquaintance`, `preface`, `foreword`, `introduction` as a prefix match on
/// the lowercased title).
#[must_use]
pub fn likely_meta(title: &str, level: i64, page_count: i64) -> bool {
    let lower = title.to_lowercase();
    let meta_words = [
        "index",
        "bibliography",
        "appendix",
        "glossary",
        "references",
        "acquaintance",
        "preface",
        "foreword",
    ];
    if meta_words.iter().any(|w| lower.contains(w)) {
        return true;
    }
    if lower.starts_with("level 1 ") && page_count <= 10 {
        return true;
    }
    level <= 1 && page_count <= 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_lifecycle_transitions() {
        let mut status = ChapterStatus::PretestReady;
        for (task, next) in [
            (TaskType::Pretest, ChapterStatus::PretestComplete),
            (TaskType::Read, ChapterStatus::ReadAvailable),
            (TaskType::Read, ChapterStatus::ReadComplete),
            (TaskType::Retest, ChapterStatus::RetestComplete),
            (
                TaskType::AssignmentWrite,
                ChapterStatus::AssignmentComplete,
            ),
            (TaskType::Notes, ChapterStatus::Completed),
        ] {
            status = status.complete(task).unwrap();
            assert_eq!(status, next);
        }
    }

    #[test]
    fn retest_requires_read_complete() {
        let err = ChapterStatus::PretestComplete
            .complete(TaskType::Retest)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
    }

    #[test]
    fn assignment_requires_retest_complete() {
        let err = ChapterStatus::ReadComplete
            .complete(TaskType::AssignmentWrite)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
    }

    #[test]
    fn locked_completed_and_skipped_have_no_next_task() {
        assert_eq!(ChapterStatus::Locked.next_task(), None);
        assert_eq!(ChapterStatus::Completed.next_task(), None);
        assert_eq!(ChapterStatus::Skipped.next_task(), None);
        assert!(ChapterStatus::Skipped.is_skipped());
        assert!(!ChapterStatus::Completed.is_skipped());
    }

    #[test]
    fn rejects_bad_registrations() {
        assert!(validate_book_registration(0).is_err());
        assert!(validate_book_registration(1).is_ok());
        assert!(validate_page_range(10, 5).is_err());
        assert!(validate_page_range(0, 5).is_err());
        assert!(validate_page_range(5, 5).is_ok());
    }

    #[test]
    fn page_count_and_cap() {
        assert_eq!(unit_page_count(1, 50), Some(50));
        assert_eq!(unit_page_count(5, 5), Some(1));
        assert_eq!(unit_page_count(10, 5), None);
        assert!(unit_fits_cap(1, 50, 50));
        assert!(!unit_fits_cap(1, 51, 50));
    }

    #[test]
    fn status_round_trips() {
        for s in [
            ChapterStatus::Locked,
            ChapterStatus::PretestReady,
            ChapterStatus::PretestComplete,
            ChapterStatus::ReadAvailable,
            ChapterStatus::ReadComplete,
            ChapterStatus::RetestComplete,
            ChapterStatus::AssignmentComplete,
            ChapterStatus::Completed,
            ChapterStatus::Skipped,
        ] {
            assert_eq!(ChapterStatus::parse(s.as_str()).unwrap(), s);
        }
    }

    #[test]
    fn likely_meta_flags() {
        assert!(likely_meta("Index", 1, 20));
        assert!(likely_meta("Bibliography", 2, 30));
        assert!(likely_meta("Level 1 Acquaintance", 1, 6));
        assert!(likely_meta("Short intro", 1, 5));
        assert!(!likely_meta("Pointers and memory", 2, 30));
        assert!(!likely_meta("Ownership", 1, 40));
    }
}
