//! Engine trait boundaries (§2.1).
//!
//! Engines never touch the filesystem, SQLite, or the network directly. They
//! receive chapter text ([`UnitText`]), configuration, and injected
//! collaborators via constructors.
//!
//! Only seams with live callers remain here: [`PdfSource`] backs ingestion,
//! and [`SupportVerdict`] is the shared source-fidelity vocabulary. Staging
//! (MCQ, assignment, grading, dispute, notes) calls concrete functions plus
//! [`crate::llm::complete_cached`] directly; the aspirational per-stage
//! traits were removed rather than kept as dead scaffolding.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Chapter text handed to content engines (DCCI: raw text injected directly).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitText {
    /// Sanitized chapter text.
    pub text: String,
    /// One-based inclusive physical pages.
    pub page_start: i64,
    /// One-based inclusive physical pages.
    pub page_end: i64,
    /// Section heading.
    pub heading: String,
}

/// A semantic unit boundary from outline / layout analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitBoundary {
    /// Heading text.
    pub heading: String,
    /// One-based physical page.
    pub page: i64,
    /// Outline depth (1 = chapter).
    pub level: i64,
}

/// A planned study unit: a contiguous one-based inclusive page range with
/// provenance. Units tile their parent span with no gaps or overlaps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedUnit {
    /// Heading text (from the outline entry that opens the unit).
    pub heading: String,
    /// Outline depth of the opening entry.
    pub level: i64,
    /// One-based inclusive physical pages.
    pub start_page: i64,
    /// One-based inclusive physical pages.
    pub end_page: i64,
}

/// PDF outline + text extraction (filesystem fixture in tests).
pub trait PdfSource {
    /// Outline entries at or after `start_page` (one-based), in document order.
    ///
    /// # Errors
    ///
    /// Returns an error when the PDF cannot be read or has no usable outline.
    fn outline(&self, start_page: i64) -> Result<Vec<UnitBoundary>>;

    /// Extract sanitized text for a one-based inclusive page range.
    ///
    /// # Errors
    ///
    /// Returns an error when the range is unreadable.
    fn text_for_range(&self, start_page: i64, end_page: i64) -> Result<UnitText>;
}

/// Source-fidelity verdicts (§18).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportVerdict {
    /// Directly stated in the source.
    DirectlySupported,
    /// Directly inferable from the source.
    InferredFromSource,
    /// Absent from the source.
    NotSupported,
    /// Contradicted by the source.
    ContradictedBySource,
}
