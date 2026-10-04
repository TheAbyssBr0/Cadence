//! Ingest orchestration (§3 steps 1–5).
//!
//! Pure planning ([`plan_book`]) is separated from side effects so the split
//! logic stays unit-testable: `MuPDF` supplies the outline and text through
//! [`PdfSource`], [`plan_book`] tiles the page range, and only then do
//! [`store_units`] / [`register_book`] touch the filesystem and [`Store`].
//! `dev ingest` calls [`plan_book`] plus text extraction and prints JSON
//! without mutating production storage.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::ChapterStatus;
use crate::engines::{PdfSource, PlannedUnit, UnitBoundary, UnitText};
use crate::error::{Error, Result};
use crate::split::{parse_manual_boundaries, plan_units, plan_units_at_level};
use crate::store::{NewBook, NewChapter, Store};

/// On-disk JSON for one study unit (`unit_<n>.json`). The corpus answers
/// *"Where did this claim come from?"* via `page_start`/`page_end`,
/// `heading`, and `source_pdf_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitPayload {
    /// Section heading from the outline.
    pub heading: String,
    /// Outline depth of the opening entry.
    pub level: i64,
    /// One-based inclusive physical pages.
    pub page_start: i64,
    /// One-based inclusive physical pages.
    pub page_end: i64,
    /// Sanitized raw text with per-page provenance markers.
    pub text: String,
    /// Hex SHA-256 of the source PDF bytes.
    pub source_pdf_hash: String,
}

/// One planned unit plus its extracted text, ready to store or print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyUnit {
    /// Heading text.
    pub heading: String,
    /// Outline depth.
    pub level: i64,
    /// One-based inclusive pages.
    pub start_page: i64,
    /// One-based inclusive pages.
    pub end_page: i64,
    /// Extracted chapter text.
    pub text: UnitText,
}

/// Hex SHA-256 of bytes (book identity: `books/<hash>/`).
#[must_use]
pub fn file_hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Default book title: the PDF file stem, falling back to the full file name.
#[must_use]
pub fn default_title(pdf_path: &Path) -> String {
    pdf_path.file_stem().and_then(|s| s.to_str()).map_or_else(
        || pdf_path.to_string_lossy().into_owned(),
        ToString::to_string,
    )
}

/// Heading for a manual range: the shallowest outline entry inside the range
/// (first on ties); falling back to the nearest entry at or before the range
/// start so provenance stays explicit without inventing titles.
fn heading_for(boundaries: &[UnitBoundary], start: i64, end: i64) -> (String, i64) {
    let mut inside: Option<(&str, i64)> = None;
    for boundary in boundaries {
        if boundary.page < start || boundary.page > end {
            continue;
        }
        let replace = match &inside {
            None => true,
            Some((_, best_level)) => boundary.level < *best_level,
        };
        if replace {
            inside = Some((boundary.heading.as_str(), boundary.level));
        }
    }
    if let Some((heading, level)) = inside {
        return (heading.to_string(), level);
    }
    let mut before: Option<(&str, i64, i64)> = None; // (heading, level, page)
    for boundary in boundaries {
        if boundary.page > start {
            continue;
        }
        let replace = match &before {
            None => true,
            Some((_, best_level, best_page)) => {
                boundary.page > *best_page
                    || (boundary.page == *best_page && boundary.level < *best_level)
            }
        };
        if replace {
            before = Some((boundary.heading.as_str(), boundary.level, boundary.page));
        }
    }
    before.map_or_else(
        || ("Untitled".to_string(), 1),
        |(heading, level, _)| (heading.to_string(), level),
    )
}

/// Tile `[start_page, doc_end_page]` from `--manual-boundaries` text, taking
/// headings from the nearest outline entry at or before each range start.
fn apply_manual(
    text: &str,
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
) -> Result<Vec<PlannedUnit>> {
    let ranges = parse_manual_boundaries(text, start_page, doc_end_page, "", 1)?;
    let mut out = Vec::with_capacity(ranges.len());
    for range in &ranges {
        let (heading, level) = heading_for(boundaries, range.start_page, range.end_page);
        out.push(PlannedUnit {
            heading,
            level,
            start_page: range.start_page,
            end_page: range.end_page,
        });
    }
    Ok(out)
}

/// Plan units for a book: outline → chapter detection + 50-page split, or
/// whole-range manual boundaries when semantic splitting fails (or when
/// `manual` is supplied and the automatic plan needs it).
///
/// When `manual` is supplied it overrides only on [`Error::NeedsManual`];
/// otherwise the semantic plan always wins — numerical splits are never
/// preferred when semantic boundaries work. An explicit `chapter_level`
/// tiles strictly at that depth (page cap ignored) and rejects `manual`:
/// pick one steering mechanism.
///
/// # Errors
///
/// Returns [`Error::NeedsManual`] when a leaf exceeds the cap and no manual
/// boundaries were supplied (automatic mode only), [`Error::InvalidInput`]
/// for a manual/explicit conflict or bad levels, plus outline/validation
/// errors from the source.
pub fn plan_book(
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
    max_unit_pages: i64,
    manual: Option<&str>,
    chapter_level: Option<i64>,
) -> Result<Vec<PlannedUnit>> {
    if chapter_level.is_some() && manual.is_some() {
        return Err(Error::InvalidInput(
            "--manual-boundaries needs automatic chapter detection; drop --chapter-level to use it"
                .to_string(),
        ));
    }
    let planned = if chapter_level.is_some() {
        plan_units_at_level(
            boundaries,
            start_page,
            doc_end_page,
            max_unit_pages,
            chapter_level,
        )
    } else {
        plan_units(boundaries, start_page, doc_end_page, max_unit_pages)
    };
    match planned {
        Ok(units) => Ok(units),
        Err(Error::NeedsManual(msg)) => {
            let Some(text) = manual else {
                return Err(Error::NeedsManual(msg));
            };
            apply_manual(text, boundaries, start_page, doc_end_page)
        }
        Err(other) => Err(other),
    }
}

/// Extract text for every planned unit, attaching headings.
///
/// # Errors
///
/// Propagates [`PdfSource`] extraction failures.
pub fn extract_units(
    source: &impl PdfSource,
    units: &[PlannedUnit],
    source_hash: &str,
) -> Result<Vec<ReadyUnit>> {
    let mut out = Vec::with_capacity(units.len());
    for unit in units {
        let mut text = source.text_for_range(unit.start_page, unit.end_page)?;
        text.heading.clone_from(&unit.heading);
        let _ = source_hash;
        out.push(ReadyUnit {
            heading: unit.heading.clone(),
            level: unit.level,
            start_page: unit.start_page,
            end_page: unit.end_page,
            text,
        });
    }
    Ok(out)
}

/// Write unit payloads to `<data_dir>/books/<hash>/chapters/unit_<n>.json`
/// (`n` one-based) and return the written paths in order.
///
/// # Errors
///
/// Returns [`Error::Io`] on filesystem failures.
pub fn store_units(
    data_dir: &Path,
    source_hash: &str,
    units: &[ReadyUnit],
) -> Result<Vec<PathBuf>> {
    let mut dir = data_dir.to_path_buf();
    dir.push("books");
    dir.push(source_hash);
    dir.push("chapters");
    std::fs::create_dir_all(&dir)?;
    let mut paths = Vec::with_capacity(units.len());
    for (index, unit) in units.iter().enumerate() {
        let Some(number) = index.checked_add(1) else {
            return Err(Error::Io("too many units".to_string()));
        };
        let mut path = dir.clone();
        path.push(format!("unit_{number}.json"));
        let payload = UnitPayload {
            heading: unit.heading.clone(),
            level: unit.level,
            page_start: unit.start_page,
            page_end: unit.end_page,
            text: unit.text.text.clone(),
            source_pdf_hash: source_hash.to_string(),
        };
        let json = serde_json::to_string_pretty(&payload).map_err(|e| Error::Io(e.to_string()))?;
        std::fs::write(&path, json)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Read a stored `unit_<n>.json` payload back into chapter text for the
/// generation stages (production MCQ/assignment/notes). The file is the
/// corpus of record: prompts cite it and validators check claims against it.
///
/// # Errors
///
/// Returns [`Error::Io`] when the file is missing, unreadable, or not a
/// valid [`UnitPayload`].
pub fn load_unit_text(path: &Path) -> Result<UnitText> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(format!("cannot read unit file {}: {e}", path.display())))?;
    let payload: UnitPayload = serde_json::from_str(&raw)
        .map_err(|e| Error::Io(format!("corrupt unit file {}: {e}", path.display())))?;
    Ok(UnitText {
        text: payload.text,
        page_start: payload.page_start,
        page_end: payload.page_end,
        heading: payload.heading,
    })
}

/// Register the book and its chapters in the [`Store`]: the first chapter is
/// `PRETEST_READY`, the rest `LOCKED`. Task creation belongs to the scheduler
/// (Phase 4) — ingest only stores the corpus and its metadata.
///
/// # Errors
///
/// Returns [`Error::Store`] on persistence failures.
pub fn register_book(
    store: &mut impl Store,
    book: &NewBook,
    units: &[ReadyUnit],
    chapter_files: &[PathBuf],
    created_at: &str,
) -> Result<crate::domain::Book> {
    let book = store.create_book(book, created_at)?;
    for (index, unit) in units.iter().enumerate() {
        let Ok(index_in_book) = i64::try_from(index) else {
            continue;
        };
        let file_path = chapter_files.get(index).map_or_else(
            || format!("unit_{}.json", index.saturating_add(1)),
            |p| p.to_string_lossy().into_owned(),
        );
        let status = if index == 0 {
            ChapterStatus::PretestReady
        } else {
            ChapterStatus::Locked
        };
        store.create_chapter(&NewChapter {
            book_id: book.id,
            index_in_book,
            level: unit.level,
            title: unit.heading.clone(),
            start_page: unit.start_page,
            end_page: unit.end_page,
            file_path,
            status,
        })?;
        store.log_event(
            "CHAPTER_REGISTERED",
            None,
            None,
            Some(unit.heading.as_str()),
            created_at,
        )?;
    }
    Ok(book)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::UnitBoundary;
    use crate::store::MemoryStore;

    fn boundary(heading: &str, page: i64, level: i64) -> UnitBoundary {
        UnitBoundary {
            heading: heading.to_string(),
            page,
            level,
        }
    }

    #[test]
    fn hashes_stably() {
        assert_eq!(
            file_hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(file_hash(b"abc"), file_hash(b"abc"));
    }

    #[test]
    fn titles_fall_back_to_filename() {
        assert_eq!(default_title(Path::new("/books/Modern C.pdf")), "Modern C");
        assert_eq!(default_title(Path::new("plain")), "plain");
    }

    #[test]
    fn semantic_plan_wins_over_manual() {
        let bounds = vec![boundary("Ch 1", 20, 2), boundary("Ch 2", 28, 2)];
        let units = plan_book(&bounds, 20, 35, 50, Some("20-35"), None).unwrap();
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn manual_rescues_oversized_leaf() {
        let bounds = vec![boundary("Big Ch", 100, 2), boundary("Next", 201, 2)];
        assert!(matches!(
            plan_book(&bounds, 100, 250, 50, None, None).unwrap_err(),
            Error::NeedsManual(_)
        ));
        let units = plan_book(&bounds, 100, 250, 50, Some("100-175,176-250"), None).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].heading, "Big Ch");
        assert_eq!(units[1].heading, "Next");
    }

    #[test]
    fn explicit_level_rejects_manual_boundaries() {
        let bounds = vec![boundary("Ch 1", 20, 2), boundary("Ch 2", 28, 2)];
        assert!(plan_book(&bounds, 20, 35, 50, Some("20-35"), Some(2)).is_err());
        let units = plan_book(&bounds, 20, 35, 50, None, Some(2)).unwrap();
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn registers_chapters_with_first_ready() {
        let mut store = MemoryStore::new();
        let units = vec![
            ReadyUnit {
                heading: "Ch 1".to_string(),
                level: 2,
                start_page: 20,
                end_page: 27,
                text: UnitText {
                    text: "t1".to_string(),
                    page_start: 20,
                    page_end: 27,
                    heading: "Ch 1".to_string(),
                },
            },
            ReadyUnit {
                heading: "Ch 2".to_string(),
                level: 2,
                start_page: 28,
                end_page: 35,
                text: UnitText {
                    text: "t2".to_string(),
                    page_start: 28,
                    page_end: 35,
                    heading: "Ch 2".to_string(),
                },
            },
        ];
        let book = register_book(
            &mut store,
            &NewBook {
                title: "Modern C".to_string(),
                filepath: "/books/m.pdf".to_string(),
                file_hash: "hash1".to_string(),
                start_page: 20,
            },
            &units,
            &[],
            "2026-01-01",
        )
        .unwrap();
        let chapters = store.list_chapters(book.id).unwrap();
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].status, ChapterStatus::PretestReady);
        assert_eq!(chapters[1].status, ChapterStatus::Locked);
        assert_eq!((chapters[0].start_page, chapters[0].end_page), (20, 27));
    }

    #[test]
    fn stores_unit_files_under_hash_dir() {
        let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), ".scratch", "ingest-store-test"]
            .iter()
            .collect();
        let _ = std::fs::remove_dir_all(&base);
        let units = vec![ReadyUnit {
            heading: "Ch 1".to_string(),
            level: 2,
            start_page: 20,
            end_page: 27,
            text: UnitText {
                text: "hello".to_string(),
                page_start: 20,
                page_end: 27,
                heading: "Ch 1".to_string(),
            },
        }];
        let paths = store_units(&base, "deadbeef", &units).unwrap();
        assert_eq!(paths.len(), 1);
        let raw = std::fs::read_to_string(&paths[0]).unwrap();
        let payload: UnitPayload = serde_json::from_str(&raw).unwrap();
        assert_eq!(payload.heading, "Ch 1");
        assert_eq!(payload.source_pdf_hash, "deadbeef");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn loads_stored_unit_text_for_generation() {
        let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), ".scratch", "ingest-load-test"]
            .iter()
            .collect();
        let _ = std::fs::remove_dir_all(&base);
        let units = vec![ReadyUnit {
            heading: "Ch 1".to_string(),
            level: 2,
            start_page: 20,
            end_page: 27,
            text: UnitText {
                text: "hello".to_string(),
                page_start: 20,
                page_end: 27,
                heading: "Ch 1".to_string(),
            },
        }];
        let paths = store_units(&base, "deadbeef", &units).unwrap();
        assert_eq!(paths.len(), 1);
        let unit = load_unit_text(&paths[0]).unwrap();
        assert_eq!(unit.text, "hello");
        assert_eq!((unit.page_start, unit.page_end), (20, 27));
        assert_eq!(unit.heading, "Ch 1");
        // Missing and corrupt files fail loudly, never silently empty.
        assert!(load_unit_text(&base.join("nope.json")).is_err());
        std::fs::write(&paths[0], "{broken").unwrap();
        assert!(load_unit_text(&paths[0]).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }
}
