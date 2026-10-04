//! Evidence-based progress metrics (§13).
//!
//! No mastery scores: every number is observed evidence — task counts, answer
//! fractions, misconception tallies, trailing-7-day pace, and streaks from
//! completion dates. [`percent`] returns `None` (displayed `n/a`) when there
//! is no evidence yet, never a zero dressed up as a measurement. Skipped
//! chapters never reach this engine (callers filter them, §4.1); `DISPUTED`
//! misconceptions are purged evidence and count toward neither side.
//!
//! The engine is pure: callers snapshot store rows into [`ChapterEvidence`]
//! (plus activity dates and punctuality pairs) and render the [`Dashboard`].

use chrono::NaiveDate;

use crate::domain::ChapterStatus;

/// Observed evidence for one live (non-skipped) chapter on its current
/// attempt. MCQ fractions count answered items only; assignment scores use
/// the effective award (`final_score` once disputed, else `score`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterEvidence {
    /// Chapter heading for display.
    pub title: String,
    /// Lifecycle state (`Completed` marks a finished unit).
    pub status: ChapterStatus,
    /// Page count (`end - start + 1`).
    pub pages: u64,
    /// Date the chapter completed, if it has (latest live `DONE` task).
    pub completed_on: Option<NaiveDate>,
    /// Correct / answered pretest items.
    pub pretest_correct: u64,
    /// Correct / answered pretest items.
    pub pretest_answered: u64,
    /// Correct / answered retest items.
    pub retest_correct: u64,
    /// Correct / answered retest items.
    pub retest_answered: u64,
    /// Effective points earned across graded assignment questions.
    pub assignment_earned: u64,
    /// Rubric points possible across graded assignment questions.
    pub assignment_possible: u64,
    /// Open rows (`ACTIVE` + `IMPROVING`).
    pub active_misconceptions: u64,
    /// Rows resolved on earlier evidence (`RESOLVED`).
    pub resolved_misconceptions: u64,
}

/// Streaks and activity derived from distinct completion dates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Consistency {
    /// Distinct days with at least one completed task.
    pub days_active: usize,
    /// Consecutive active days ending today (or yesterday — a streak stays
    /// alive until a full idle day passes).
    pub current_streak: usize,
    /// Longest run of consecutive active days ever.
    pub longest_streak: usize,
}

/// Trailing-7-day pace plus a completion estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pace {
    /// Units completed in `[today-6, today]` (a 7-day window, so the count
    /// itself is the per-week rate).
    pub units_per_week: f64,
    /// Pages completed in the same window.
    pub pages_per_week: f64,
    /// Projected finish date at the current rate (`None` = unknown: nothing
    /// completed in the window; `Some(today)` = nothing left).
    pub eta: Option<NaiveDate>,
}

/// Aggregated dashboard over all live chapters.
#[derive(Debug, Clone, PartialEq)]
pub struct Dashboard {
    /// Chapters with status `Completed`.
    pub units_completed: usize,
    /// Live chapters not yet completed.
    pub units_remaining: usize,
    /// Pages in completed units.
    pub pages_completed: u64,
    /// Pages in remaining live units.
    pub pages_remaining: u64,
    /// Live `DONE` tasks / live tasks total.
    pub tasks_done: usize,
    /// Live `DONE` tasks / live tasks total.
    pub tasks_total: usize,
    /// Skipped chapters (excluded from every denominator here).
    pub skipped: usize,
    /// Pretest percent over answered items, if any.
    pub pretest: Option<f64>,
    /// Retest percent over answered items, if any.
    pub retest: Option<f64>,
    /// Assignment percent over graded points, if any.
    pub assignment: Option<f64>,
    /// Raw pretest sums for display (`13/21`).
    pub pretest_fraction: (u64, u64),
    /// Raw retest sums for display.
    pub retest_fraction: (u64, u64),
    /// Raw assignment sums for display.
    pub assignment_fraction: (u64, u64),
    /// Open misconception rows across live chapters.
    pub active_misconceptions: u64,
    /// Resolved misconception rows across live chapters.
    pub resolved_misconceptions: u64,
    /// Resolved share of all non-purged rows, if any.
    pub resolution: Option<f64>,
    /// Trailing pace and estimate.
    pub pace: Pace,
    /// Activity streaks.
    pub consistency: Consistency,
    /// Share of done tasks finished on or before their scheduled date, if any.
    pub on_time: Option<f64>,
    /// Pending tasks with `scheduled_for < today`.
    pub overdue: usize,
}

/// Everything [`summarize`] needs beyond the chapter snapshots: task totals,
/// distinct completion dates, `(scheduled, completed)` pairs for punctuality,
/// the overdue count, and the injected `today`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashboardInput<'a> {
    /// One snapshot per live (non-skipped) chapter.
    pub chapters: &'a [ChapterEvidence],
    /// Live `DONE` tasks.
    pub tasks_done: usize,
    /// Live tasks total.
    pub tasks_total: usize,
    /// Skipped chapters (display only — excluded from denominators).
    pub skipped: usize,
    /// Completion dates (one per done task; deduped inside).
    pub active_days: &'a [NaiveDate],
    /// `(scheduled_for, completed_on)` per done task with a parseable date.
    pub punctual: &'a [(NaiveDate, NaiveDate)],
    /// Pending tasks with `scheduled_for < today`.
    pub overdue: usize,
    /// Injected today (never read from the wall clock).
    pub today: NaiveDate,
}

/// Integer-to-float for evidence counts. Narrows through `u32` (saturating at
/// `u32::MAX`, unreachable for real evidence) to stay total under
/// `as_conversions`.
fn num_to_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

/// Observed percent (`earned / possible * 100`), or `None` when there is no
/// evidence (`possible == 0`). Callers display `None` as `n/a`.
#[must_use]
pub fn percent(earned: u64, possible: u64) -> Option<f64> {
    if possible == 0 {
        return None;
    }
    Some(num_to_f64(earned) / num_to_f64(possible) * 100.0)
}

/// Display a percent: whole-number `%`, or `n/a` without evidence.
#[must_use]
pub fn format_percent(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.0}%"))
}

/// Display an evidence fraction with its percent: `5/8 (63%)`, `0/0 (n/a)`.
#[must_use]
pub fn format_fraction(earned: u64, possible: u64) -> String {
    format!("{earned}/{possible} ({})", format_percent(percent(earned, possible)))
}

/// Streaks from completion dates. Future dates are ignored (defensive —
/// writers only store `today`), duplicates collapse to distinct active days.
#[must_use]
pub fn consistency(active_days: &[NaiveDate], today: NaiveDate) -> Consistency {
    let mut days: Vec<NaiveDate> = active_days
        .iter()
        .copied()
        .filter(|d| *d <= today)
        .collect();
    days.sort();
    days.dedup();
    let mut longest = 0_usize;
    let mut run = 0_usize;
    let mut prev: Option<NaiveDate> = None;
    for day in &days {
        let continues =
            prev.is_some_and(|p| p.checked_add_days(chrono::Days::new(1)) == Some(*day));
        if continues {
            run = run.saturating_add(1);
        } else {
            run = 1;
        }
        longest = longest.max(run);
        prev = Some(*day);
    }
    // The streak is alive when the latest active day is today or yesterday.
    let mut current = 0_usize;
    let yesterday = today.checked_sub_days(chrono::Days::new(1));
    if let Some(mut cursor) = days.last().copied().filter(|last| {
        *last == today || yesterday.is_some_and(|prior| *last == prior)
    }) {
        for day in days.iter().rev() {
            if *day != cursor {
                break;
            }
            current = current.saturating_add(1);
            let Some(prior) = cursor.checked_sub_days(chrono::Days::new(1)) else {
                break;
            };
            cursor = prior;
        }
    }
    Consistency {
        days_active: days.len(),
        current_streak: current,
        longest_streak: longest,
    }
}

/// Share of done tasks finished on or before their scheduled date, or `None`
/// without dated completions.
#[must_use]
pub fn on_time_percent(punctual: &[(NaiveDate, NaiveDate)]) -> Option<f64> {
    if punctual.is_empty() {
        return None;
    }
    let on_time = punctual
        .iter()
        .filter(|(scheduled, done)| done <= scheduled)
        .count();
    let total = u64::try_from(punctual.len()).unwrap_or(u64::MAX);
    let hits = u64::try_from(on_time).unwrap_or(u64::MAX);
    percent(hits, total)
}

/// Trailing-7-day pace over `(completion date, pages)` pairs plus an ETA for
/// `remaining_units`. `None` ETA means unknown (no window completions);
/// `Some(today)` means nothing is left.
#[must_use]
pub fn pace(
    completions: &[(NaiveDate, u64)],
    today: NaiveDate,
    remaining_units: usize,
) -> Pace {
    let window_start = today
        .checked_sub_days(chrono::Days::new(6))
        .unwrap_or(today);
    let mut units = 0_u64;
    let mut pages = 0_u64;
    for (date, unit_pages) in completions {
        if *date >= window_start && *date <= today {
            units = units.saturating_add(1);
            pages = pages.saturating_add(*unit_pages);
        }
    }
    let eta = if remaining_units == 0 {
        Some(today)
    } else if units == 0 {
        None
    } else {
        let remaining = u64::try_from(remaining_units).unwrap_or(u64::MAX);
        // ceil(remaining * 7 / units) in integers; any overflow → unknown.
        let days_needed = remaining
            .checked_mul(7)
            .and_then(|scaled| scaled.checked_add(units))
            .and_then(|scaled| scaled.checked_sub(1))
            .and_then(|scaled| scaled.checked_div(units));
        days_needed.and_then(|days| today.checked_add_days(chrono::Days::new(days)))
    };
    Pace {
        units_per_week: num_to_f64(units),
        pages_per_week: num_to_f64(pages),
        eta,
    }
}

/// Resolved share of all non-purged rows, or `None` when no rows exist yet.
#[must_use]
pub fn resolution_rate(active: u64, resolved: u64) -> Option<f64> {
    let total = active.saturating_add(resolved);
    if total == 0 {
        return None;
    }
    percent(resolved, total)
}

/// Aggregate chapter snapshots plus activity into one [`Dashboard`].
#[must_use]
pub fn summarize(input: &DashboardInput) -> Dashboard {
    let mut units_completed = 0_usize;
    let mut units_remaining = 0_usize;
    let mut pages_completed = 0_u64;
    let mut pages_remaining = 0_u64;
    let mut pretest_correct = 0_u64;
    let mut pretest_answered = 0_u64;
    let mut retest_correct = 0_u64;
    let mut retest_answered = 0_u64;
    let mut assignment_earned = 0_u64;
    let mut assignment_possible = 0_u64;
    let mut active = 0_u64;
    let mut resolved = 0_u64;
    let mut completions = Vec::new();
    for chapter in input.chapters {
        if chapter.status == ChapterStatus::Completed {
            units_completed = units_completed.saturating_add(1);
            pages_completed = pages_completed.saturating_add(chapter.pages);
            if let Some(date) = chapter.completed_on {
                completions.push((date, chapter.pages));
            }
        } else {
            units_remaining = units_remaining.saturating_add(1);
            pages_remaining = pages_remaining.saturating_add(chapter.pages);
        }
        pretest_correct = pretest_correct.saturating_add(chapter.pretest_correct);
        pretest_answered = pretest_answered.saturating_add(chapter.pretest_answered);
        retest_correct = retest_correct.saturating_add(chapter.retest_correct);
        retest_answered = retest_answered.saturating_add(chapter.retest_answered);
        assignment_earned = assignment_earned.saturating_add(chapter.assignment_earned);
        assignment_possible = assignment_possible.saturating_add(chapter.assignment_possible);
        active = active.saturating_add(chapter.active_misconceptions);
        resolved = resolved.saturating_add(chapter.resolved_misconceptions);
    }
    Dashboard {
        units_completed,
        units_remaining,
        pages_completed,
        pages_remaining,
        tasks_done: input.tasks_done,
        tasks_total: input.tasks_total,
        skipped: input.skipped,
        pretest: percent(pretest_correct, pretest_answered),
        retest: percent(retest_correct, retest_answered),
        assignment: percent(assignment_earned, assignment_possible),
        pretest_fraction: (pretest_correct, pretest_answered),
        retest_fraction: (retest_correct, retest_answered),
        assignment_fraction: (assignment_earned, assignment_possible),
        active_misconceptions: active,
        resolved_misconceptions: resolved,
        resolution: resolution_rate(active, resolved),
        pace: pace(&completions, input.today, units_remaining),
        consistency: consistency(input.active_days, input.today),
        on_time: on_time_percent(input.punctual),
        overdue: input.overdue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    fn evidence(title: &str, status: ChapterStatus) -> ChapterEvidence {
        ChapterEvidence {
            title: title.to_string(),
            status,
            pages: 10,
            completed_on: None,
            pretest_correct: 0,
            pretest_answered: 0,
            retest_correct: 0,
            retest_answered: 0,
            assignment_earned: 0,
            assignment_possible: 0,
            active_misconceptions: 0,
            resolved_misconceptions: 0,
        }
    }

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn percent_needs_evidence() {
        assert_eq!(percent(0, 0), None);
        assert_eq!(percent(5, 0), None);
        approx(percent(9, 10).unwrap(), 90.0);
        approx(percent(0, 8).unwrap(), 0.0);
    }

    #[test]
    fn format_percent_rounds_or_defers() {
        assert_eq!(format_percent(None), "n/a");
        assert_eq!(format_percent(Some(86.4)), "86%");
        assert_eq!(format_percent(Some(100.0)), "100%");
        assert_eq!(format_fraction(2, 3), "2/3 (67%)");
        assert_eq!(format_fraction(0, 0), "0/0 (n/a)");
    }

    #[test]
    fn consistency_empty_has_no_streak() {
        let found = consistency(&[], day(2026, 9, 27));
        assert_eq!(
            found,
            Consistency {
                days_active: 0,
                current_streak: 0,
                longest_streak: 0
            }
        );
    }

    #[test]
    fn consistency_counts_run_ending_today() {
        let days = vec![day(2026, 9, 24), day(2026, 9, 26), day(2026, 9, 27)];
        let found = consistency(&days, day(2026, 9, 27));
        assert_eq!(found.days_active, 3);
        assert_eq!(found.current_streak, 2);
        assert_eq!(found.longest_streak, 2);
    }

    #[test]
    fn consistency_stays_alive_through_yesterday() {
        let days = vec![day(2026, 9, 25), day(2026, 9, 26)];
        let found = consistency(&days, day(2026, 9, 27));
        assert_eq!(found.current_streak, 2);
        assert_eq!(found.longest_streak, 2);
    }

    #[test]
    fn consistency_breaks_after_a_full_idle_day() {
        let days = vec![day(2026, 9, 20), day(2026, 9, 21), day(2026, 9, 25)];
        let found = consistency(&days, day(2026, 9, 27));
        assert_eq!(found.current_streak, 0);
        assert_eq!(found.longest_streak, 2);
        assert_eq!(found.days_active, 3);
    }

    #[test]
    fn consistency_ignores_future_days_and_duplicates() {
        let days = vec![
            day(2026, 9, 27),
            day(2026, 9, 27),
            day(2026, 9, 28),
            day(2026, 9, 26),
        ];
        let found = consistency(&days, day(2026, 9, 27));
        assert_eq!(found.days_active, 2);
        assert_eq!(found.current_streak, 2);
    }

    #[test]
    fn on_time_needs_dated_completions() {
        assert_eq!(on_time_percent(&[]), None);
        let samples = vec![
            (day(2026, 9, 20), day(2026, 9, 20)),
            (day(2026, 9, 21), day(2026, 9, 20)),
            (day(2026, 9, 22), day(2026, 9, 25)),
        ];
        approx(on_time_percent(&samples).unwrap(), 200.0 / 3.0);
    }

    #[test]
    fn pace_counts_trailing_window_only() {
        let today = day(2026, 9, 27);
        let completions = vec![
            (day(2026, 9, 27), 20),
            (day(2026, 9, 21), 10),
            (day(2026, 9, 14), 50),
        ];
        let found = pace(&completions, today, 7);
        approx(found.units_per_week, 2.0);
        approx(found.pages_per_week, 30.0);
        // 7 units left at 2/week → ceil(7*7/2) = 25 days.
        assert_eq!(found.eta, Some(day(2026, 10, 22)));
    }

    #[test]
    fn pace_without_window_completions_has_unknown_eta() {
        let today = day(2026, 9, 27);
        let found = pace(&[], today, 5);
        approx(found.units_per_week, 0.0);
        assert_eq!(found.eta, None);
    }

    #[test]
    fn pace_with_nothing_left_finishes_today() {
        let today = day(2026, 9, 27);
        let found = pace(&[], today, 0);
        assert_eq!(found.eta, Some(today));
    }

    #[test]
    fn resolution_needs_rows() {
        assert_eq!(resolution_rate(0, 0), None);
        approx(resolution_rate(1, 3).unwrap(), 75.0);
        approx(resolution_rate(2, 0).unwrap(), 0.0);
    }

    #[test]
    fn summarize_aggregates_chapters_and_activity() {
        let today = day(2026, 9, 27);
        let mut done = evidence("Pointers", ChapterStatus::Completed);
        done.completed_on = Some(day(2026, 9, 26));
        done.pretest_correct = 5;
        done.pretest_answered = 8;
        done.retest_correct = 7;
        done.retest_answered = 8;
        done.assignment_earned = 14;
        done.assignment_possible = 16;
        done.active_misconceptions = 1;
        done.resolved_misconceptions = 2;
        let mut open = evidence("Lifetimes", ChapterStatus::ReadComplete);
        open.pretest_correct = 4;
        open.pretest_answered = 8;
        let input = DashboardInput {
            chapters: &[done, open],
            tasks_done: 6,
            tasks_total: 10,
            skipped: 1,
            active_days: &[day(2026, 9, 26), day(2026, 9, 27)],
            punctual: &[(day(2026, 9, 26), day(2026, 9, 26))],
            overdue: 2,
            today,
        };
        let board = summarize(&input);
        assert_eq!(board.units_completed, 1);
        assert_eq!(board.units_remaining, 1);
        assert_eq!(board.pages_completed, 10);
        assert_eq!(board.pages_remaining, 10);
        assert_eq!(board.pretest_fraction, (9, 16));
        approx(board.pretest.unwrap(), 56.25);
        approx(board.retest.unwrap(), 87.5);
        approx(board.assignment.unwrap(), 87.5);
        assert_eq!(board.active_misconceptions, 1);
        assert_eq!(board.resolved_misconceptions, 2);
        approx(board.resolution.unwrap(), 200.0 / 3.0);
        assert_eq!(board.consistency.current_streak, 2);
        approx(board.on_time.unwrap(), 100.0);
        assert_eq!(board.overdue, 2);
        approx(board.pace.units_per_week, 1.0);
    }

    #[test]
    fn summarize_empty_store_shows_no_evidence() {
        let today = day(2026, 9, 27);
        let input = DashboardInput {
            chapters: &[],
            tasks_done: 0,
            tasks_total: 0,
            skipped: 0,
            active_days: &[],
            punctual: &[],
            overdue: 0,
            today,
        };
        let board = summarize(&input);
        assert_eq!(board.units_completed, 0);
        assert_eq!(board.pretest, None);
        assert_eq!(board.retest, None);
        assert_eq!(board.assignment, None);
        assert_eq!(board.resolution, None);
        assert_eq!(board.on_time, None);
        assert_eq!(board.pace.eta, Some(today));
    }
}
