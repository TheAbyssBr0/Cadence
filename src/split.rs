//! Unit splitting: chapter detection + the 50-page rule (§3).
//!
//! Pure logic over [`UnitBoundary`] slices — no filesystem, no clock.
//! [`plan_units`] tiles `[start_page, doc_end_page]` with contiguous,
//! non-overlapping [`PlannedUnit`]s. Units open at the *chapter level*:
//! every outline entry at that depth or shallower starts a unit running to
//! the next such entry. By default the chapter level is auto-detected
//! ([`detect_chapter_level`]); an explicit level (see
//! [`plan_units_at_level`]) tiles strictly and ignores the page cap.
//! A span fitting the cap is kept whole, otherwise it splits at the
//! shallowest deeper outline level inside the span (subchapter boundaries).
//! A leaf with no deeper split that still exceeds the cap returns
//! [`Error::NeedsManual`]: the caller must prompt for `--manual-boundaries`
//! — numerical splits are never invented when semantic boundaries exist,
//! and a subsection is never split for numerical convenience.

use crate::domain::{unit_fits_cap, unit_page_count};
use crate::engines::{PlannedUnit, UnitBoundary};
use crate::error::{Error, Result};

/// Minimum median entry span (pages) for an outline level to count as
/// chapters: sections and subsections typically span a few pages, while
/// chapters run longer. Well under the 50-page unit cap on purpose — this
/// is a floor telling chapters apart from sections, not a size limit.
const MIN_CHAPTER_SPAN_PAGES: i64 = 8;

/// Flat, page-grouped outline entry: one row per distinct page (the
/// shallowest entry wins ties, so same-page sub-entries never create
/// zero-width units).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    heading: String,
    level: i64,
    page: i64,
}

/// Validate inputs and group boundaries by distinct page.
fn grouped_entries(
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
    max_unit_pages: i64,
) -> Result<Vec<Entry>> {
    if max_unit_pages < 1 {
        return Err(Error::InvalidInput(
            "--max-unit-pages must be >= 1".to_string(),
        ));
    }
    if start_page < 1 || doc_end_page < 1 {
        return Err(Error::InvalidInput(
            "pages are one-based and must be >= 1".to_string(),
        ));
    }
    if doc_end_page < start_page {
        return Err(Error::InvalidInput(format!(
            "document end page ({doc_end_page}) precedes start page ({start_page})"
        )));
    }
    if boundaries.is_empty() {
        return Err(Error::InvalidInput(
            "no outline boundaries to split".to_string(),
        ));
    }
    for boundary in boundaries {
        if boundary.page < start_page || boundary.page > doc_end_page {
            return Err(Error::InvalidInput(format!(
                "boundary '{}' on page {} is outside [{start_page}, {doc_end_page}]",
                boundary.heading, boundary.page
            )));
        }
        if boundary.level < 1 {
            return Err(Error::InvalidInput(format!(
                "boundary '{}' has invalid level {}",
                boundary.heading, boundary.level
            )));
        }
    }
    // Stable sort by page; grouping below keeps document order within a page.
    let mut sorted: Vec<UnitBoundary> = boundaries.to_vec();
    sorted.sort_by_key(|b| b.page);
    let mut grouped: Vec<Entry> = Vec::new();
    for boundary in &sorted {
        match grouped.last() {
            Some(last) if last.page == boundary.page => {
                // Same-page duplicate: keep the shallowest opener (and the
                // first on ties) so the unit heading names the section.
                if boundary.level < last.level
                    && let Some(slot) = grouped.last_mut()
                {
                    slot.heading.clone_from(&boundary.heading);
                    slot.level = boundary.level;
                }
            }
            _ => grouped.push(Entry {
                heading: boundary.heading.clone(),
                level: boundary.level,
                page: boundary.page,
            }),
        }
    }
    Ok(grouped)
}

/// Distinct outline depths present, ascending.
fn sorted_levels(entries: &[Entry]) -> Vec<i64> {
    let mut levels: Vec<i64> = entries.iter().map(|e| e.level).collect();
    levels.sort_unstable();
    levels.dedup();
    levels
}

/// Auto-detect the chapter level: the deepest outline level whose entries
/// span chapter-sized ranges (median span of at least
/// [`MIN_CHAPTER_SPAN_PAGES`] pages). Part dividers lose by being few with
/// huge spans only at shallower levels; sections lose by being numerous with
/// tiny spans. Entry spans run to the next entry at the same depth or
/// shallower (else `doc_end_page`); zero-width spans never count.
///
/// Returns `None` when no level qualifies — callers fall back to the
/// shallowest level (historical behavior).
#[must_use]
pub fn detect_chapter_level(boundaries: &[UnitBoundary], doc_end_page: i64) -> Option<i64> {
    let mut ordered: Vec<&UnitBoundary> = boundaries.iter().filter(|b| b.level >= 1).collect();
    ordered.sort_by_key(|b| b.page);
    let mut levels: Vec<i64> = ordered.iter().map(|b| b.level).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut chapter_level = None;
    for level in levels {
        let mut spans: Vec<i64> = Vec::new();
        for (position, boundary) in ordered.iter().enumerate() {
            if boundary.level != level {
                continue;
            }
            let end = ordered
                .iter()
                .skip(position.saturating_add(1))
                .find(|later| later.level <= level)
                .map_or(doc_end_page, |later| {
                    later.page.checked_sub(1).unwrap_or(doc_end_page)
                });
            if let Some(span) = unit_page_count(boundary.page, end) {
                spans.push(span);
            }
        }
        spans.sort_unstable();
        let median = spans.get(spans.len() / 2).copied();
        if median.is_some_and(|span| span >= MIN_CHAPTER_SPAN_PAGES) {
            chapter_level = Some(level);
        }
    }
    chapter_level
}

/// Human-readable outline inventory for ingest output, so the chapter level
/// is never a blind guess: per-depth entry counts with first/last titles,
/// plus the effective chapter level (explicit or detected).
#[must_use]
pub fn describe_levels(
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
    chapter_level: Option<i64>,
) -> String {
    use std::collections::BTreeMap;
    let mut lines = vec![format!(
        "Outline depths (entries at/after page {start_page}):"
    )];
    let mut per_level: BTreeMap<i64, Vec<&UnitBoundary>> = BTreeMap::new();
    for boundary in boundaries
        .iter()
        .filter(|b| b.level >= 1 && b.page >= start_page)
    {
        per_level.entry(boundary.level).or_default().push(boundary);
    }
    if per_level.is_empty() {
        lines.push("  (no outline entries)".to_string());
    }
    for (level, items) in &per_level {
        let titles: Vec<&str> = items.iter().map(|b| b.heading.as_str()).collect();
        let first = titles.first().copied().unwrap_or("?");
        let last = titles.last().copied().unwrap_or("?");
        let noun = if items.len() == 1 { "entry" } else { "entries" };
        lines.push(format!(
            "  level {level}: {} {noun}, e.g. '{first}' … '{last}'",
            items.len()
        ));
    }
    match chapter_level {
        Some(level) => lines.push(format!(
            "Using chapter level: {level} (explicit; page cap ignored)"
        )),
        None => match detect_chapter_level(boundaries, doc_end_page) {
            Some(level) => lines.push(format!("Detected chapter level: {level}")),
            None => lines.push(
                "No chapter level detected — falling back to the shallowest outline level"
                    .to_string(),
            ),
        },
    }
    lines.join("\n")
}

/// Tile `[start_page, doc_end_page]` with study units: automatic chapter
/// detection plus the 50-page rule.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on incoherent input and
/// [`Error::NeedsManual`] when a leaf with no deeper semantic split still
/// exceeds `max_unit_pages`.
pub fn plan_units(
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
    max_unit_pages: i64,
) -> Result<Vec<PlannedUnit>> {
    plan_units_at_level(boundaries, start_page, doc_end_page, max_unit_pages, None)
}

/// Resolve the chapter level: explicit (validated against the outline) or
/// detected, falling back to the shallowest outline depth.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on depths below 1 or when no outline
/// entry sits at the requested depth.
fn resolve_chapter_level(
    entries: &[Entry],
    boundaries: &[UnitBoundary],
    doc_end_page: i64,
    chapter_level: Option<i64>,
) -> Result<i64> {
    if let Some(level) = chapter_level {
        if level < 1 {
            return Err(Error::InvalidInput(format!(
                "chapter level must be >= 1 (got {level})"
            )));
        }
        if !entries.iter().any(|e| e.level == level) {
            let available = sorted_levels(entries)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::InvalidInput(format!(
                "no outline entries at depth {level}; available depths: {available}"
            )));
        }
        return Ok(level);
    }
    let shallowest = entries.iter().map(|e| e.level).min().unwrap_or(1);
    Ok(detect_chapter_level(boundaries, doc_end_page).unwrap_or(shallowest))
}

/// Bounds of one chapter-level span: opener page through the page before the
/// next opener (or the document end). `None` when the span is empty.
fn span_bounds(
    entries: &[Entry],
    starts: &[usize],
    position: usize,
    start_page: i64,
    doc_end_page: i64,
) -> Option<(i64, i64)> {
    let start_index = starts.get(position).copied()?;
    let span_start = if position == 0 {
        start_page
    } else {
        entries.get(start_index).map_or(start_page, |e| e.page)
    };
    let span_end = starts
        .get(position.saturating_add(1))
        .and_then(|next| entries.get(*next))
        .map_or(doc_end_page, |e| {
            e.page.checked_sub(1).unwrap_or(doc_end_page)
        });
    (span_start <= span_end).then_some((span_start, span_end))
}

/// Plan one chapter-level span: kept whole when explicit or small, else split
/// recursively at subchapter boundaries.
///
/// # Errors
///
/// Propagates [`Error::NeedsManual`] for oversized spans without subchapter
/// boundaries.
fn push_span_units(
    entries: &[Entry],
    span_start: i64,
    span_end: i64,
    strict: bool,
    max_unit_pages: i64,
    out: &mut Vec<PlannedUnit>,
) -> Result<()> {
    let owned: Vec<Entry> = entries
        .iter()
        .filter(|e| e.page >= span_start && e.page <= span_end)
        .cloned()
        .collect();
    let Some(head) = owned.first() else {
        return Ok(());
    };
    if strict {
        out.push(PlannedUnit {
            heading: head.heading.clone(),
            level: head.level,
            start_page: span_start,
            end_page: span_end,
        });
        return Ok(());
    }
    let opener = if head.page > span_start {
        let mut with_opener = vec![Entry {
            heading: head.heading.clone(),
            level: head.level,
            page: span_start,
        }];
        with_opener.extend(owned);
        with_opener
    } else {
        owned
    };
    split_span(&opener, span_start, span_end, max_unit_pages, out)
}

/// Tile `[start_page, doc_end_page]` with study units at an explicit outline
/// depth: units open at every entry at `chapter_level` or shallower and run
/// to the next such entry, keeping oversized spans whole — the page cap and
/// manual-boundary prompts do not apply. Pass `None` for automatic chapter
/// detection (see [`detect_chapter_level`]) with the 50-page rule.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on incoherent input, on depths below 1,
/// or when no outline entry sits at the requested depth (the message lists
/// the available depths); [`Error::NeedsManual`] as in [`plan_units`] in
/// automatic mode only.
pub fn plan_units_at_level(
    boundaries: &[UnitBoundary],
    start_page: i64,
    doc_end_page: i64,
    max_unit_pages: i64,
    chapter_level: Option<i64>,
) -> Result<Vec<PlannedUnit>> {
    let entries = grouped_entries(boundaries, start_page, doc_end_page, max_unit_pages)?;
    let Some(first) = entries.first() else {
        return Err(Error::InvalidInput(
            "no outline boundaries to split".to_string(),
        ));
    };
    if first.page > start_page {
        return Err(Error::InvalidInput(format!(
            "first boundary '{}' on page {} leaves pages [{start_page}, {}) uncovered",
            first.heading, first.page, first.page
        )));
    }
    let mut out = Vec::new();
    // Units open at the chapter level and anything coarser (part dividers,
    // back matter): each such entry starts a unit running to the next one,
    // so small dividers become stub units instead of swallowing chapters.
    // Each span then splits recursively only if it exceeds the cap —
    // unless the level was chosen explicitly, which keeps spans whole.
    let strict = chapter_level.is_some();
    let chapter_level = resolve_chapter_level(&entries, boundaries, doc_end_page, chapter_level)?;
    let mut starts: Vec<usize> = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry.level <= chapter_level {
            starts.push(index);
        }
    }
    if starts.is_empty() {
        return Err(Error::InvalidInput(
            "no outline boundaries to split".to_string(),
        ));
    }
    for position in 0..starts.len() {
        let Some((span_start, span_end)) =
            span_bounds(&entries, &starts, position, start_page, doc_end_page)
        else {
            continue;
        };
        push_span_units(
            &entries,
            span_start,
            span_end,
            strict,
            max_unit_pages,
            &mut out,
        )?;
    }
    verify_coverage(&out, start_page, doc_end_page)?;
    Ok(out)
}

/// Recursively split one span. `entries` holds the grouped outline entries
/// with `page` in `[span_start, span_end]`, sorted, non-empty, with
/// `entries[0].page <= span_start`.
fn split_span(
    entries: &[Entry],
    span_start: i64,
    span_end: i64,
    max_unit_pages: i64,
    out: &mut Vec<PlannedUnit>,
) -> Result<()> {
    let Some(head) = entries.first() else {
        return Err(Error::InvalidInput("empty span".to_string()));
    };
    let size = span_end
        .checked_sub(span_start)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| Error::InvalidInput("span overflow".to_string()))?;
    if unit_fits_cap(span_start, span_end, max_unit_pages) {
        out.push(PlannedUnit {
            heading: head.heading.clone(),
            level: head.level,
            start_page: span_start,
            end_page: span_end,
        });
        return Ok(());
    }
    // Split at the shallowest deeper level present (immediate subchapters,
    // whatever depth number the outline uses).
    let deeper_min = entries
        .iter()
        .skip(1)
        .filter(|e| e.page > span_start && e.page <= span_end)
        .map(|e| e.level)
        .min();
    let Some(child_level) = deeper_min else {
        return Err(Error::NeedsManual(format!(
            "'{}' spans pages {span_start}–{span_end} ({size} pages, cap {max_unit_pages}) with no subchapter boundaries; supply --manual-boundaries",
            head.heading
        )));
    };
    // Sub-span openers: the head opens the first sub-span, each child entry
    // at `child_level` opens the next.
    let mut points: Vec<(String, i64, i64)> = Vec::new();
    points.push((head.heading.clone(), head.level, span_start));
    for entry in entries.iter().skip(1) {
        if entry.page > span_start && entry.page <= span_end && entry.level == child_level {
            points.push((entry.heading.clone(), entry.level, entry.page));
        }
    }
    if points.len() < 2 {
        return Err(Error::NeedsManual(format!(
            "'{}' spans pages {span_start}–{span_end} ({size} pages, cap {max_unit_pages}) with no usable subchapter split; supply --manual-boundaries",
            head.heading
        )));
    }
    for (index, (heading, level, sub_start)) in points.iter().enumerate() {
        let sub_end = points
            .get(index.saturating_add(1))
            .map_or(span_end, |(_, _, next_start)| {
                next_start.checked_sub(1).unwrap_or(span_end)
            });
        if *sub_start > sub_end {
            continue;
        }
        let owned: Vec<Entry> = entries
            .iter()
            .filter(|e| e.page >= *sub_start && e.page <= sub_end)
            .cloned()
            .collect();
        let opener = if owned.is_empty() {
            vec![Entry {
                heading: heading.clone(),
                level: *level,
                page: *sub_start,
            }]
        } else {
            owned
        };
        split_span(&opener, *sub_start, sub_end, max_unit_pages, out)?;
    }
    Ok(())
}

/// Confirm the plan tiles `[start_page, doc_end_page]` with no gaps or
/// overlaps — the corpus must always answer *"Where did this claim come
/// from?"*, which requires contiguous unit coverage.
fn verify_coverage(units: &[PlannedUnit], start_page: i64, doc_end_page: i64) -> Result<()> {
    let Some(first) = units.first() else {
        return Err(Error::InvalidInput("split produced no units".to_string()));
    };
    if first.start_page != start_page {
        return Err(Error::InvalidInput(format!(
            "split leaves pages [{start_page}, {}) uncovered",
            first.start_page
        )));
    }
    let mut cursor = start_page;
    for unit in units {
        if unit.start_page != cursor {
            return Err(Error::InvalidInput(format!(
                "split has a gap or overlap at page {}",
                unit.start_page
            )));
        }
        if unit.end_page < unit.start_page {
            return Err(Error::InvalidInput(format!(
                "unit '{}' has an inverted range",
                unit.heading
            )));
        }
        let Some(next) = unit.end_page.checked_add(1) else {
            return Err(Error::InvalidInput("unit range overflow".to_string()));
        };
        cursor = next;
    }
    let Some(last_end) = cursor.checked_sub(1) else {
        return Err(Error::InvalidInput("unit range underflow".to_string()));
    };
    if last_end != doc_end_page {
        return Err(Error::InvalidInput(format!(
            "split covers through page {last_end}, expected {doc_end_page}"
        )));
    }
    Ok(())
}

/// Parse `--manual-boundaries` text (`"start-end,start-end,..."`, one-based)
/// into units tiling `[leaf_start, leaf_end]`.
///
/// The first unit keeps the leaf `heading`; later ones are suffixed
/// `(part N)` so provenance stays explicit without inventing titles.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] unless the ranges are sorted, contiguous,
/// and exactly cover `[leaf_start, leaf_end]`.
pub fn parse_manual_boundaries(
    text: &str,
    leaf_start: i64,
    leaf_end: i64,
    heading: &str,
    level: i64,
) -> Result<Vec<PlannedUnit>> {
    let mut units = Vec::new();
    for chunk in text.split(',') {
        let chunk = chunk.trim();
        let (start_text, end_text) = chunk.split_once('-').ok_or_else(|| {
            Error::InvalidInput(format!("bad boundary '{chunk}': expected start-end"))
        })?;
        let start: i64 = start_text.trim().parse().map_err(|_| {
            Error::InvalidInput(format!("bad boundary '{chunk}': not a page number"))
        })?;
        let end: i64 = end_text.trim().parse().map_err(|_| {
            Error::InvalidInput(format!("bad boundary '{chunk}': not a page number"))
        })?;
        if start < leaf_start || end > leaf_end || end < start {
            return Err(Error::InvalidInput(format!(
                "boundary '{chunk}' is outside the {leaf_start}–{leaf_end} span"
            )));
        }
        units.push((start, end));
    }
    if units.is_empty() {
        return Err(Error::InvalidInput(
            "no manual boundaries supplied".to_string(),
        ));
    }
    units.sort_unstable();
    let mut cursor = leaf_start;
    for (start, end) in &units {
        if *start != cursor {
            return Err(Error::InvalidInput(format!(
                "manual boundaries must contiguously cover {leaf_start}–{leaf_end} (gap or overlap at page {start})"
            )));
        }
        let Some(next) = end.checked_add(1) else {
            return Err(Error::InvalidInput("boundary overflow".to_string()));
        };
        cursor = next;
    }
    let Some(last_end) = cursor.checked_sub(1) else {
        return Err(Error::InvalidInput("boundary underflow".to_string()));
    };
    if last_end != leaf_end {
        return Err(Error::InvalidInput(format!(
            "manual boundaries cover through page {last_end}, expected {leaf_end}"
        )));
    }
    let mut out = Vec::with_capacity(units.len());
    for (index, (start, end)) in units.iter().enumerate() {
        let Some(part) = index.checked_add(1) else {
            return Err(Error::InvalidInput("too many parts".to_string()));
        };
        let title = if units.len() == 1 {
            heading.to_string()
        } else {
            format!("{heading} (part {part})")
        };
        out.push(PlannedUnit {
            heading: title,
            level,
            start_page: *start,
            end_page: *end,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(heading: &str, page: i64, level: i64) -> UnitBoundary {
        UnitBoundary {
            heading: heading.to_string(),
            page,
            level,
        }
    }

    #[test]
    fn small_book_stays_whole() {
        let bounds = vec![boundary("Ch 1", 20, 2), boundary("Ch 2", 28, 2)];
        let units = plan_units(&bounds, 20, 35, 50).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].start_page, 20);
        assert_eq!(units[0].end_page, 27);
        assert_eq!(units[1].start_page, 28);
        assert_eq!(units[1].end_page, 35);
    }

    #[test]
    fn oversized_chapter_splits_at_sections() {
        let bounds = vec![
            boundary("Ch 5", 100, 2),
            boundary("5.1", 110, 3),
            boundary("5.2", 130, 3),
            boundary("Ch 6", 171, 2),
        ];
        let units = plan_units(&bounds, 100, 200, 50).unwrap();
        assert_eq!(units.len(), 4);
        assert_eq!((units[0].start_page, units[0].end_page), (100, 109));
        assert_eq!((units[1].start_page, units[1].end_page), (110, 129));
        assert_eq!((units[2].start_page, units[2].end_page), (130, 170));
        assert_eq!((units[3].start_page, units[3].end_page), (171, 200));
        assert_eq!(units[0].heading, "Ch 5");
        assert_eq!(units[1].heading, "5.1");
        assert_eq!(units[2].heading, "5.2");
        assert_eq!(units[3].heading, "Ch 6");
    }

    #[test]
    fn recursion_descends_past_fitting_children() {
        // 5.2 itself is 60 pages; its subsections split it further.
        let bounds = vec![
            boundary("Ch 5", 100, 2),
            boundary("5.1", 110, 3),
            boundary("5.2", 130, 3),
            boundary("5.2.1", 150, 4),
            boundary("Ch 6", 200, 2),
        ];
        let units = plan_units(&bounds, 100, 210, 50).unwrap();
        for unit in &units {
            let size = unit.end_page - unit.start_page + 1;
            assert!(size <= 50, "oversized: {unit:?}");
        }
        // Coverage is contiguous from 100 to 210.
        assert_eq!(units.first().unwrap().start_page, 100);
        assert_eq!(units.last().unwrap().end_page, 210);
    }

    #[test]
    fn sectionless_leaf_demands_manual() {
        let bounds = vec![boundary("Big Ch", 100, 2), boundary("Next", 201, 2)];
        let err = plan_units(&bounds, 100, 250, 50).unwrap_err();
        assert!(matches!(err, Error::NeedsManual(_)));
    }

    #[test]
    fn subsection_never_split_numerically() {
        // A 40-page section inside an oversized chapter stays whole even
        // though a numerical split could balance sizes better.
        let bounds = vec![
            boundary("Ch", 100, 2),
            boundary("S1", 110, 3),
            boundary("S2", 150, 3),
            boundary("Next", 200, 2),
        ];
        let units = plan_units(&bounds, 100, 210, 50).unwrap();
        let kept = units.iter().find(|u| u.heading == "S1").unwrap();
        assert_eq!((kept.start_page, kept.end_page), (110, 149));
    }

    #[test]
    fn same_page_duplicates_share_a_unit() {
        let bounds = vec![
            boundary("5.7 Binary representions", 87, 3),
            boundary("5.7.1 Unsigned integers", 87, 4),
            boundary("5.7.2 Bit sets", 88, 4),
            boundary("Next", 120, 3),
        ];
        let units = plan_units(&bounds, 87, 130, 50).unwrap();
        assert_eq!(units[0].start_page, 87);
        assert_eq!(units[0].heading, "5.7 Binary representions");
    }

    #[test]
    fn rejects_empty_and_incoherent_input() {
        assert!(plan_units(&[], 20, 30, 50).is_err());
        assert!(plan_units(&[boundary("A", 20, 2)], 20, 30, 0).is_err());
        assert!(plan_units(&[boundary("A", 5, 2)], 20, 30, 50).is_err());
        // First boundary after start_page leaves a gap.
        assert!(plan_units(&[boundary("A", 25, 2)], 20, 30, 50).is_err());
    }

    #[test]
    fn manual_boundaries_must_tile_exactly() {
        let units = parse_manual_boundaries("100-140,141-180", 100, 180, "Big Ch", 2).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].heading, "Big Ch (part 1)");
        assert_eq!(units[1].heading, "Big Ch (part 2)");
        // Gap.
        assert!(parse_manual_boundaries("100-139,141-180", 100, 180, "Big Ch", 2).is_err());
        // Overlap.
        assert!(parse_manual_boundaries("100-141,141-180", 100, 180, "Big Ch", 2).is_err());
        // Under-coverage.
        assert!(parse_manual_boundaries("100-179", 100, 180, "Big Ch", 2).is_err());
        // Out of span.
        assert!(parse_manual_boundaries("90-180", 100, 180, "Big Ch", 2).is_err());
        // Garbage.
        assert!(parse_manual_boundaries("abc", 100, 180, "Big Ch", 2).is_err());
    }

    #[test]
    fn plan_units_tiles_chapter_spans() {
        let bounds = vec![boundary("Ch 1", 20, 2), boundary("Ch 2", 28, 2)];
        let units = plan_units(&bounds, 20, 35, 50).unwrap();
        assert_eq!(units.len(), 2);
    }

    /// Modern-C-shaped outline: part dividers (level 1) must not swallow
    /// their chapters (level 2); sections (level 3) must not become units.
    fn part_book() -> Vec<UnitBoundary> {
        vec![
            boundary("Level 0", 18, 1),
            boundary("ch 1", 20, 2),
            boundary("1.1", 21, 3),
            boundary("1.2", 22, 3),
            boundary("ch 2", 28, 2),
            boundary("2.1", 29, 3),
            boundary("Level 1", 38, 1),
            boundary("ch 3", 44, 2),
            boundary("3.1", 45, 3),
            boundary("3.2", 47, 3),
        ]
    }

    #[test]
    fn detects_deepest_chapter_level() {
        assert_eq!(detect_chapter_level(&part_book(), 60), Some(2));
    }

    #[test]
    fn detection_falls_back_when_everything_tiny() {
        let bounds = vec![
            boundary("c1", 1, 1),
            boundary("c2", 5, 1),
            boundary("c3", 9, 1),
            boundary("c4", 13, 1),
        ];
        assert_eq!(detect_chapter_level(&bounds, 16), None);
    }

    #[test]
    fn auto_mode_splits_fitting_part_divider() {
        let units = plan_units(&part_book(), 18, 60, 50).unwrap();
        let headings: Vec<&str> = units.iter().map(|u| u.heading.as_str()).collect();
        assert_eq!(headings, vec!["Level 0", "ch 1", "ch 2", "Level 1", "ch 3"]);
        assert_eq!((units[0].start_page, units[0].end_page), (18, 19));
        assert_eq!((units[1].start_page, units[1].end_page), (20, 27));
    }

    #[test]
    fn explicit_level_tiles_strictly_ignoring_cap() {
        let bounds = vec![
            boundary("Part", 20, 1),
            boundary("Ch 1", 22, 2),
            boundary("Ch 2", 30, 2),
        ];
        let units = plan_units_at_level(&bounds, 20, 100, 50, Some(2)).unwrap();
        assert_eq!(units.len(), 3);
        // Part stub, then chapters — the 71-page chapter stays whole
        // despite the 50-page cap.
        assert_eq!((units[0].start_page, units[0].end_page), (20, 21));
        assert_eq!((units[1].start_page, units[1].end_page), (22, 29));
        assert_eq!((units[2].start_page, units[2].end_page), (30, 100));
    }

    #[test]
    fn explicit_level_without_entries_names_available_depths() {
        let bounds = vec![boundary("Ch 1", 20, 2)];
        let err = plan_units_at_level(&bounds, 20, 35, 50, Some(5)).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(ref msg) if msg.contains("available depths: 2")),
            "unexpected: {err}"
        );
    }

    #[test]
    fn explicit_level_below_one_rejected() {
        let bounds = vec![boundary("Ch 1", 20, 2)];
        assert!(plan_units_at_level(&bounds, 20, 35, 50, Some(0)).is_err());
    }

    #[test]
    fn describe_levels_shows_table_and_effective_level() {
        let table = describe_levels(&part_book(), 18, 60, None);
        assert!(table.contains("level 1: 2 entries"), "table:\n{table}");
        assert!(table.contains("level 2: 3 entries"), "table:\n{table}");
        assert!(table.contains("level 3: 5 entries"), "table:\n{table}");
        assert!(
            table.contains("Detected chapter level: 2"),
            "table:\n{table}"
        );
        let explicit = describe_levels(&part_book(), 18, 60, Some(3));
        assert!(
            explicit.contains("Using chapter level: 3 (explicit; page cap ignored)"),
            "table:\n{explicit}"
        );
    }
}
