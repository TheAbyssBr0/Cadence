//! PDF ingestion via `MuPDF` (§3).
//!
//! [`MuPdfSource`] implements [`crate::engines::PdfSource`]: outline parsing
//! (primary source) plus per-page text extraction. Page references are
//! one-based physical pages everywhere outside this module; `MuPDF` destination
//! page numbers are zero-based and converted with [`zero_based_to_one_based`].
//!
//! Raw text and layout provenance are preserved: [`sanitize_text`] only
//! normalizes line endings and strips trailing whitespace per line. It never
//! touches code punctuation, quotation marks, indentation, or hyphenation.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mupdf::{Document, TextExtractOptions};

use crate::engines::{PdfSource, UnitBoundary, UnitText};
use crate::error::{Error, Result};

/// Filesystem-backed `MuPDF` source. The struct only retains the path and page
/// count; a fresh [`Document`] is opened per call (`MuPDF` documents are
/// `!Send` and short-lived here).
#[derive(Debug, Clone)]
pub struct MuPdfSource {
    path: PathBuf,
    pages: i64,
}

impl MuPdfSource {
    /// Open a PDF and read its page count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pdf`] when the file cannot be opened, needs a
    /// password, or reports no pages.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::Pdf(format!("no such PDF: {}", path.display())));
        }
        let doc = open_document(path)?;
        let needs_password = doc
            .needs_password()
            .map_err(|e| Error::Pdf(e.to_string()))?;
        if needs_password {
            return Err(Error::Pdf(format!(
                "encrypted PDF requires a password: {}",
                path.display()
            )));
        }
        let count_i32 = doc
            .page_count()
            .map_err(|e| Error::Pdf(e.to_string()))?;
        let pages = i64::from(count_i32);
        if pages < 1 {
            return Err(Error::Pdf(format!(
                "PDF has no pages: {}",
                path.display()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            pages,
        })
    }

    /// One-based physical page count.
    #[must_use]
    pub const fn page_count(&self) -> i64 {
        self.pages
    }

    /// Open a fresh `MuPDF` document handle.
    fn document(&self) -> Result<Document> {
        open_document(&self.path)
    }
}

/// Open a `MuPDF` document from a filesystem path (`Document::open` takes a
/// `MuPDF` [`mupdf::FilePath`], convertible from `&str`).
fn open_document(path: &Path) -> Result<Document> {
    let owned = path.to_string_lossy().into_owned();
    Document::open(&owned).map_err(|e| Error::Pdf(e.to_string()))
}

/// Convert a zero-based `MuPDF` destination page number to a one-based physical
/// page reference.
///
/// # Errors
///
/// Returns [`Error::Pdf`] on overflow (practically unreachable; `u32 -> i64`
/// always fits, but the `+1` is still checked).
pub fn zero_based_to_one_based(zero_based: u32) -> Result<i64> {
    let base = i64::try_from(zero_based)
        .map_err(|e| Error::Pdf(format!("page number overflow: {e}")))?;
    base.checked_add(1)
        .ok_or_else(|| Error::Pdf("page number overflow".to_string()))
}

/// Normalize line endings (`\r\n`, `\r` → `\n`) and strip trailing whitespace
/// on each line. Leading indentation (code blocks), punctuation, quotation
/// marks, and intra-line spacing are preserved byte-for-byte.
#[must_use]
pub fn sanitize_text(raw: &str) -> String {
    let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
    normalized
        .split('\n')
        .map(|line| line.trim_end_matches([' ', '\t']))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolve one outline entry to a zero-based page number: direct destination
/// first, then `uri` via `resolve_link` (handles `#nameddest=N`). Link
/// resolution failures degrade to `None` (entry skipped, children still
/// visited) rather than failing the whole ingest.
fn resolve_entry(doc: &Document, entry: &mupdf::Outline) -> Option<u32> {
    if let Some(dest) = entry.dest {
        return Some(dest.loc.page_number);
    }
    let uri = entry.uri.as_deref()?;
    doc.resolve_link(uri).ok().flatten().map(|d| d.loc.page_number)
}

/// Depth-first flatten of the outline tree into `(title, one_based_page,
/// level)` triples in document order. Entries without a resolvable page are
/// skipped (their children are still visited).
fn flatten(
    doc: &Document,
    entries: &[mupdf::Outline],
    depth: i64,
    out: &mut Vec<(String, i64, i64)>,
) {
    for entry in entries {
        let child_depth = depth.saturating_add(1);
        if let Some(zero_based) = resolve_entry(doc, entry) {
            if let Ok(one_based) = zero_based_to_one_based(zero_based) {
                out.push((entry.title.clone(), one_based, depth));
            }
        }
        flatten(doc, &entry.down, child_depth, out);
    }
}

impl PdfSource for MuPdfSource {
    fn outline(&self, start_page: i64) -> Result<Vec<UnitBoundary>> {
        if start_page < 1 {
            return Err(Error::InvalidInput(
                "start-page is mandatory and must be >= 1 (one-based physical page)".to_string(),
            ));
        }
        let doc = self.document()?;
        let raw = doc.outlines().map_err(|e| Error::Pdf(e.to_string()))?;
        let mut flat: Vec<(String, i64, i64)> = Vec::new();
        flatten(&doc, &raw, 1, &mut flat);
        let mut out = Vec::with_capacity(flat.len());
        for (heading, page, level) in flat {
            if page < start_page {
                continue;
            }
            if page > self.pages {
                continue;
            }
            out.push(UnitBoundary {
                heading,
                page,
                level,
            });
        }
        if out.is_empty() {
            return Err(Error::Pdf(format!(
                "no outline entries at or after start page {start_page}; the PDF has no usable outline there (typography/layout fallback lands later)"
            )));
        }
        Ok(out)
    }

    fn text_for_range(&self, start_page: i64, end_page: i64) -> Result<UnitText> {
        crate::domain::validate_page_range(start_page, end_page)?;
        if end_page > self.pages {
            return Err(Error::InvalidInput(format!(
                "end page {end_page} exceeds document page count {}",
                self.pages
            )));
        }
        let doc = self.document()?;
        let mut combined = String::new();
        let mut page = start_page;
        while page <= end_page {
            let zero_based = page
                .checked_sub(1)
                .ok_or_else(|| Error::InvalidInput("page underflow".to_string()))?;
            let index = i32::try_from(zero_based).map_err(|e| {
                Error::InvalidInput(format!("page index overflow: {e}"))
            })?;
            let loaded = doc
                .load_page(index)
                .map_err(|e| Error::Pdf(e.to_string()))?;
            let raw = loaded
                .text(TextExtractOptions::default())
                .map_err(|e| Error::Pdf(e.to_string()))?;
            let clean = sanitize_text(&raw);
            if combined.is_empty() {
                let _ = write!(combined, "--- physical page {page} ---\n");
            } else {
                let _ = write!(combined, "\n\n--- physical page {page} ---\n");
            }
            combined.push_str(&clean);
            let Some(next) = page.checked_add(1) else {
                break;
            };
            page = next;
        }
        Ok(UnitText {
            text: combined,
            page_start: start_page,
            page_end: end_page,
            heading: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_pdf() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.push("Modern C.pdf");
        p
    }

    #[test]
    fn zero_based_conversion() {
        assert_eq!(zero_based_to_one_based(0).unwrap(), 1);
        assert_eq!(zero_based_to_one_based(407).unwrap(), 408);
    }

    #[test]
    fn sanitize_preserves_code_and_indentation() {
        let raw = "for (i = 0; i < 5; ++i) {   \r\n    printf(\"%d\", i);\t \r\n";
        let clean = sanitize_text(raw);
        assert!(clean.contains("for (i = 0; i < 5; ++i) {"));
        assert!(clean.contains("    printf(\"%d\", i);"));
        assert!(!clean.contains("   \n"));
        assert!(!clean.contains('\r'));
    }

    #[test]
    fn sanitize_does_not_normalize_quotes_or_hyphens() {
        let raw = "“quoted” co-operate don’t\n";
        assert_eq!(sanitize_text(raw), "“quoted” co-operate don’t\n");
    }

    #[test]
    fn sanitize_empty_stays_empty() {
        assert_eq!(sanitize_text(""), "");
    }

    #[test]
    fn outline_filters_front_matter() {
        let source = MuPdfSource::open(&fixture_pdf()).unwrap();
        assert_eq!(source.page_count(), 408);
        let all = source.outline(1).unwrap();
        assert!(all.first().is_some());
        let from_18 = source.outline(18).unwrap();
        assert!(from_18.iter().all(|b| b.page >= 18));
        assert!(from_18.len() < all.len());
        let chapter_one = from_18
            .iter()
            .find(|b| b.heading == "1 Getting started")
            .unwrap();
        assert_eq!(chapter_one.page, 20);
    }

    #[test]
    fn outline_pages_within_document() {
        let source = MuPdfSource::open(&fixture_pdf()).unwrap();
        for entry in source.outline(18).unwrap() {
            assert!(entry.page >= 18);
            assert!(entry.page <= 408);
            assert!(entry.level >= 1);
            assert_ne!(entry.heading.as_str(), "");
        }
    }

    #[test]
    fn text_range_carries_provenance_and_content() {
        let source = MuPdfSource::open(&fixture_pdf()).unwrap();
        let unit = source.text_for_range(25, 25).unwrap();
        assert_eq!(unit.page_start, 25);
        assert_eq!(unit.page_end, 25);
        assert!(unit.text.contains("physical page 25"));
        assert!(unit.text.contains("Getting started"));
        assert!(unit.text.len() > 500);
    }

    #[test]
    fn text_range_rejects_bad_ranges() {
        let source = MuPdfSource::open(&fixture_pdf()).unwrap();
        assert!(source.text_for_range(0, 5).is_err());
        assert!(source.text_for_range(10, 5).is_err());
        assert!(source.text_for_range(400, 409).is_err());
    }

    #[test]
    fn open_rejects_missing_file() {
        let err = MuPdfSource::open(Path::new("/nonexistent/book.pdf")).unwrap_err();
        assert!(matches!(err, Error::Pdf(_)));
    }
}
