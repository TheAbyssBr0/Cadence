//! Scheduling engine (§5–§6).
//!
//! Pure logic over task slices: classification into `OVERDUE` / `DUE TODAY` /
//! `FUTURE`, §6 execution priority, and the 3-day sliding-window projection.
//! Time is injected as `NaiveDate` — the scheduler never reads the wall clock.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::domain::{Chapter, ChapterStatus, Task, TaskStatus, TaskType};
use crate::error::{Error, Result};
use crate::store::{NewTask, Store};

/// How a pending task relates to `today`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskBucket {
    /// `scheduled_date <= today` and still pending (§6 treats `== today` as
    /// due-today for display; overdue strictly means `< today`).
    Overdue,
    /// Due today.
    DueToday,
    /// Scheduled after today.
    Future,
    /// Already finished.
    Done,
}

/// Classify one task against an injected `today`.
#[must_use]
pub fn classify_task(task: &Task, today: NaiveDate) -> TaskBucket {
    if task.status == TaskStatus::Done {
        return TaskBucket::Done;
    }
    match task.scheduled_for.cmp(&today) {
        std::cmp::Ordering::Less => TaskBucket::Overdue,
        std::cmp::Ordering::Equal => TaskBucket::DueToday,
        std::cmp::Ordering::Greater => TaskBucket::Future,
    }
}

/// Rank for §6 execution priority: overdue retests → overdue assignments →
/// today's retests → today's assignments → notes (chapter closure) →
/// today's pretest+read (new learning) → future work. Returns a tuple ordered
/// lexicographically (lower first). Notes close out their chapter before new
/// chapters open: a `NOTES` due today always beats a `PRETEST`/`READ` due
/// today, so grading flows straight into notes instead of starting the next
/// chapter's pre-read.
#[must_use]
pub fn execution_rank(task: &Task, today: NaiveDate) -> (u8, NaiveDate, i64) {
    let overdue = u8::from(task.status == TaskStatus::Pending && task.scheduled_for < today);
    // `overdue_first`: 0 for overdue, 1 for the rest, so overdue sorts first.
    let overdue_first = 1_u8.wrapping_sub(overdue);
    let type_rank = match task.task_type {
        TaskType::Retest => 0,
        TaskType::AssignmentWrite => 1,
        TaskType::Notes => 2,
        TaskType::Pretest => 3,
        TaskType::Read => 4,
    };
    // Combine into a single leading byte: overdue class dominates type rank.
    let leading = overdue_first.saturating_mul(10).saturating_add(type_rank);
    (leading, task.scheduled_for, task.sequence)
}

/// Today's executable queue: pending tasks with `scheduled_for <= today`,
/// ordered by §6 priority. Future tasks are excluded; completed tasks are
/// excluded.
#[must_use]
pub fn today_queue(tasks: &[Task], today: NaiveDate) -> Vec<Task> {
    let mut out: Vec<Task> = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Pending && t.scheduled_for <= today)
        .cloned()
        .collect();
    out.sort_by_key(|t| execution_rank(t, today));
    out
}

/// Overdue subset of [`today_queue`], same ordering.
#[must_use]
pub fn overdue_tasks(tasks: &[Task], today: NaiveDate) -> Vec<Task> {
    let mut out: Vec<Task> = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Pending && t.scheduled_for < today)
        .cloned()
        .collect();
    out.sort_by_key(|t| execution_rank(t, today));
    out
}

/// Whether `pull` is allowed: all mandatory work for today is complete, i.e.
/// no pending task with `scheduled_for <= today`.
#[must_use]
pub fn pull_available(tasks: &[Task], today: NaiveDate) -> bool {
    !tasks
        .iter()
        .any(|t| t.status == TaskStatus::Pending && t.scheduled_for <= today)
}

/// Next pullable future task (§6): the earliest future `PRETEST` — the next
/// chapter's first step. Retests, assignments, and notes keep their scheduled
/// days (their spacing is the pedagogy); only new learning pulls forward.
/// Earliest date wins, lowest creation sequence breaks ties.
/// Returns `None` when no future pretest exists.
#[must_use]
pub fn next_pull_candidate(tasks: &[Task], today: NaiveDate) -> Option<Task> {
    tasks
        .iter()
        .filter(|t| {
            t.status == TaskStatus::Pending
                && t.task_type == TaskType::Pretest
                && t.scheduled_for > today
        })
        .min_by_key(|t| (t.scheduled_for, t.sequence))
        .cloned()
}

/// Pull the next chapter forward to `today` (§6).
///
/// The gate runs over live rows only: skipped chapters and abandoned
/// attempts are audit trail and never block pulling (§4.1). Stale future
/// candidates (skipped chapter, attempt mismatch, orphaned chapter) are
/// skipped, never pulled.
///
/// Returns `Ok(None)` when no future pretest exists, otherwise the pulled
/// pretest with its previous date. Multiple pulls cascade forward through
/// queued pretests: pulled work is mandatory, so each call advances exactly
/// one task.
///
/// # Errors
///
/// Returns [`Error::InvalidTransition`] while mandatory work remains,
/// and propagates [`Error::Store`] on backend failure.
pub fn pull_next(
    store: &mut dyn Store,
    today: NaiveDate,
    created_at: &str,
) -> Result<Option<(Task, NaiveDate)>> {
    let mut chapters = Vec::new();
    for book in store.list_books()? {
        chapters.extend(store.list_chapters(book.id)?);
    }
    let live = |task: &Task| {
        chapters
            .iter()
            .find(|c| c.id == task.chapter_id)
            .is_some_and(|c| !c.status.is_skipped() && task.attempt_no == c.attempt_no)
    };
    let tasks = store.list_tasks()?;
    let live_tasks: Vec<Task> = tasks.iter().filter(|t| live(t)).cloned().collect();
    if !pull_available(&live_tasks, today) {
        return Err(Error::InvalidTransition(
            "mandatory work remains — clear today's queue before pulling future work".to_string(),
        ));
    }
    let Some(next) = next_pull_candidate(&live_tasks, today) else {
        return Ok(None);
    };
    let previous = next.scheduled_for;
    let pulled = store.reschedule_task(next.id, today)?;
    let evidence = format!("{previous} -> {today}");
    store.log_event(
        "TASK_PULLED",
        Some(pulled.chapter_id),
        Some(pulled.id),
        Some(&evidence),
        created_at,
    )?;
    Ok(Some((pulled, previous)))
}

/// Open the reading stage (§4): `PRETEST_COMPLETE` → `READ_AVAILABLE`.
///
/// Scheduling (not task completion) performs this transition: finishing the
/// pretest immediately unlocks reading, and the `READ` task created alongside
/// it moves the chapter to `READ_COMPLETE` when finished.
///
/// # Errors
///
/// Returns [`Error::InvalidTransition`] unless `status` is `PretestComplete`.
pub fn open_reading(status: ChapterStatus) -> Result<ChapterStatus> {
    if status == ChapterStatus::PretestComplete {
        return Ok(ChapterStatus::ReadAvailable);
    }
    Err(Error::InvalidTransition(format!(
        "cannot open reading while chapter is {}",
        status.as_str()
    )))
}

/// Whether a chapter has finished its Day N read (retest unlocked).
const fn reading_done_or_beyond(status: ChapterStatus) -> bool {
    matches!(
        status,
        ChapterStatus::ReadComplete
            | ChapterStatus::RetestComplete
            | ChapterStatus::AssignmentComplete
            | ChapterStatus::Completed
    )
}

/// Next free creation `sequence`: one past the maximum in use (starts at 1).
fn next_sequence(tasks: &[Task]) -> i64 {
    let mut max: i64 = 0;
    for task in tasks {
        if task.sequence > max {
            max = task.sequence;
        }
    }
    max.saturating_add(1).max(1)
}

/// Calendar addition that never panics: overflow falls back to `date`
/// (unreachable for real schedules; keeps the engine infallible).
fn add_days(date: NaiveDate, days: u64) -> NaiveDate {
    date.checked_add_days(chrono::Days::new(days))
        .unwrap_or(date)
}

/// Whether a `PENDING` task of `kind` already exists for this chapter attempt
/// (in `tasks`, which should include rows created earlier in the same call).
fn has_pending_for(tasks: &[Task], chapter_id: i64, attempt_no: i64, kind: TaskType) -> bool {
    tasks.iter().any(|t| {
        t.chapter_id == chapter_id
            && t.attempt_no == attempt_no
            && t.task_type == kind
            && t.status == TaskStatus::Pending
    })
}

/// Nearest non-skipped predecessor status before `position` in an
/// `index_in_book`-ordered status view; `None` at the head or behind only
/// skipped chapters (skips are transparent, §4.1).
fn live_predecessor(statuses: &[ChapterStatus], position: usize) -> Option<ChapterStatus> {
    for back in (0..position).rev() {
        let prev = statuses
            .get(back)
            .copied()
            .unwrap_or(ChapterStatus::Skipped);
        if prev == ChapterStatus::Skipped {
            continue;
        }
        return Some(prev);
    }
    None
}
/// Count skipped chapters in a listing (metrics `Skipped: N`, §4.1).
#[must_use]
pub fn count_skipped(chapters: &[Chapter]) -> usize {
    chapters.iter().filter(|c| c.status.is_skipped()).count()
}

/// Pre-chapter confirmation prompt (§4.1): new chapters are caught where they
/// are noticed. Three-way: proceed, skip the chapter outright, or exit the
/// loop with state saved — `n` never skips (only an explicit `s` does).
#[must_use]
pub fn format_chapter_start_prompt(title: &str, start_page: i64, end_page: i64) -> String {
    format!("Starting '{title}' (pages {start_page}–{end_page})? [Y/s(kip)/n] > ")
}

/// Idempotently top up scheduler state for one book (§4–§6).
///
/// For each chapter in `index_in_book` order:
/// * unlocks the next `LOCKED` chapter once all earlier non-skipped chapters
///   finished reading (`READ_COMPLETE` or beyond; skipped chapters are
///   transparent, §4.1);
/// * auto-opens reading for chapters stranded at `PRETEST_COMPLETE`
///   (crash between pretest completion and `READ` creation);
/// * creates the missing `PENDING` task for [`ChapterStatus::next_task`]
///   when no pending task of that kind exists for the chapter's current
///   attempt — due `today`, except a chained unlock's first `PRETEST`, which
///   is due tomorrow (pullable today).
///
/// Rescue tasks are due `today` (never pushed to the future): overdue work
/// surfaces through the normal catch-up queue. A chapter chained-unlocked
/// behind a finished predecessor (not the head) schedules its first
/// `PRETEST` for tomorrow: Day N stays closed once its work is done, and an
/// explicit `pull` brings the pretest forward when the user wants more.
/// The Day N/N+1/N+2 stagger for fresh work otherwise comes from
/// [`complete_and_advance`], not from here.
///
/// Returns the tasks created by this call (empty when already topped up).
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
/// Unlock one `LOCKED` chapter whose gating predecessor finished reading.
/// Returns whether this chapter chained behind finished work (its first
/// pretest belongs to tomorrow, not to the day just closed).
fn unlock_chapter(
    store: &mut dyn Store,
    chapter: &Chapter,
    statuses: &mut [ChapterStatus],
    position: usize,
    created_at: &str,
) -> Result<bool> {
    if statuses
        .get(position)
        .copied()
        .unwrap_or(ChapterStatus::Locked)
        != ChapterStatus::Locked
    {
        return Ok(false);
    }
    // Look back past skipped chapters for the gating predecessor.
    let predecessor = live_predecessor(statuses, position);
    let blocked = predecessor.is_some_and(|prev| !reading_done_or_beyond(prev));
    if blocked {
        return Ok(false);
    }
    store.set_chapter_status(chapter.id, ChapterStatus::PretestReady)?;
    store.log_event(
        "CHAPTER_UNLOCKED",
        Some(chapter.id),
        None,
        Some(chapter.title.as_str()),
        created_at,
    )?;
    if let Some(slot) = statuses.get_mut(position) {
        *slot = ChapterStatus::PretestReady;
    }
    // A non-head unlock chains behind finished work: its first
    // pretest belongs to tomorrow, not to the day just closed.
    Ok(predecessor.is_some())
}

pub fn ensure_tasks(
    store: &mut dyn Store,
    book_id: i64,
    today: NaiveDate,
    created_at: &str,
) -> Result<Vec<Task>> {
    let chapters = store.list_chapters(book_id)?;
    let mut known = store.list_tasks()?;
    let mut sequence = next_sequence(&known);
    let mut created = Vec::new();
    // Local status view so unlocks earlier in this pass gate later chapters.
    let mut statuses: Vec<ChapterStatus> = chapters.iter().map(|c| c.status).collect();
    for (position, chapter) in chapters.iter().enumerate() {
        let current = statuses
            .get(position)
            .copied()
            .unwrap_or(ChapterStatus::Locked);
        // Unlock rule: the head chapter and any chapter whose earlier
        // non-skipped chapters all finished reading become available.
        // Chained unlocks (a finished non-skipped predecessor) start tomorrow
        // so a closed Day N stays closed; head chapters start today.
        let unlocked_behind_predecessor = if current == ChapterStatus::Locked {
            unlock_chapter(store, chapter, &mut statuses, position, created_at)?
        } else {
            false
        };
        // Crash-safety: a chapter stranded at PRETEST_COMPLETE never got its
        // reading stage opened; open it now before creating the READ task.
        let current = statuses
            .get(position)
            .copied()
            .unwrap_or(ChapterStatus::Locked);
        if current == ChapterStatus::PretestComplete {
            let opened = open_reading(current)?;
            store.set_chapter_status(chapter.id, opened)?;
            store.log_event(
                "READING_OPENED",
                Some(chapter.id),
                None,
                Some(chapter.title.as_str()),
                created_at,
            )?;
            if let Some(slot) = statuses.get_mut(position) {
                *slot = opened;
            }
        }
        let current = statuses
            .get(position)
            .copied()
            .unwrap_or(ChapterStatus::Locked);
        let Some(next) = current.next_task() else {
            continue;
        };
        if has_pending_for(&known, chapter.id, chapter.attempt_no, next) {
            continue;
        }
        // Chained first-pretests start tomorrow (pullable today); every
        // other missing task is a same-chapter rescue due today.
        let due = if unlocked_behind_predecessor && next == TaskType::Pretest {
            add_days(today, 1)
        } else {
            today
        };
        let task = store.create_task(&NewTask {
            book_id,
            chapter_id: chapter.id,
            task_type: next,
            scheduled_for: due,
            sequence,
            attempt_no: chapter.attempt_no,
        })?;
        sequence = sequence.saturating_add(1).max(1);
        store.log_event(
            "TASK_SCHEDULED",
            Some(chapter.id),
            Some(task.id),
            Some(next.as_str()),
            created_at,
        )?;
        known.push(task.clone());
        created.push(task);
    }
    Ok(created)
}

/// After `NOTES` completes, the chapter is `COMPLETED`: unlock the next
/// `LOCKED` sibling with its first `PRETEST` due tomorrow (pullable today;
/// skipped chapters stay skipped). Returns the fresh pretest, if any.
fn unlock_next_after_notes(
    store: &mut dyn Store,
    chapter: &Chapter,
    today: NaiveDate,
    completed_at: &str,
) -> Result<Option<Task>> {
    let siblings = store.list_chapters(chapter.book_id)?;
    let position = siblings.iter().position(|c| c.id == chapter.id);
    if let Some(index) = position {
        for sibling in siblings.iter().skip(index.saturating_add(1)) {
            if sibling.status != ChapterStatus::Locked {
                continue;
            }
            store.set_chapter_status(sibling.id, ChapterStatus::PretestReady)?;
            store.log_event(
                "CHAPTER_UNLOCKED",
                Some(sibling.id),
                None,
                Some(sibling.title.as_str()),
                completed_at,
            )?;
            let fresh = store.list_tasks()?;
            let created = store.create_task(&NewTask {
                book_id: sibling.book_id,
                chapter_id: sibling.id,
                task_type: TaskType::Pretest,
                scheduled_for: add_days(today, 1),
                sequence: next_sequence(&fresh),
                attempt_no: sibling.attempt_no,
            })?;
            store.log_event(
                "TASK_SCHEDULED",
                Some(sibling.id),
                Some(created.id),
                Some(TaskType::Pretest.as_str()),
                completed_at,
            )?;
            return Ok(Some(created));
        }
    }
    Ok(None)
}

/// Complete one task and advance the pipeline (§4–§5).
///
/// Validates the transition structurally ([`ChapterStatus::complete`]), marks
/// the task `DONE`, moves the chapter forward (pretest completion auto-opens
/// reading via [`open_reading`]), and schedules the follow-up with the 3-day
/// sliding-window offsets: `READ` due today (Day N), `RETEST` due tomorrow
/// (Day N+1), `ASSIGNMENT_WRITE` due tomorrow (Day N+2 relative to Day N),
/// `NOTES` due today. Completing `NOTES` finishes the chapter (`COMPLETED`)
/// and unlocks the next `LOCKED` chapter with a fresh `PRETEST` due tomorrow
/// (pullable today, so a closed Day N stays closed).
///
/// Completing an already-`DONE` task returns [`Error::AlreadyCompleted`]
/// without mutating state. Stale tasks from abandoned attempts (attempt
/// mismatch) are rejected with [`Error::InvalidTransition`].
///
/// Returns the follow-up task, if one was scheduled.
///
/// # Errors
///
/// Returns [`Error::NotFound`] for unknown tasks/chapters,
/// [`Error::AlreadyCompleted`] for reruns, [`Error::InvalidTransition`] for
/// out-of-order completions, and [`Error::Store`] on backend failure.
pub fn complete_and_advance(
    store: &mut dyn Store,
    task_id: i64,
    today: NaiveDate,
    completed_at: &str,
) -> Result<Option<Task>> {
    let task = store.get_task(task_id)?;
    if task.status == TaskStatus::Done {
        return Err(Error::AlreadyCompleted(format!("task {task_id}")));
    }
    let chapter = store.get_chapter(task.chapter_id)?;
    if task.attempt_no != chapter.attempt_no {
        return Err(Error::InvalidTransition(format!(
            "task {task_id} belongs to attempt {} but chapter {} is on attempt {}",
            task.attempt_no, chapter.id, chapter.attempt_no
        )));
    }
    // Pretest completion also opens reading (Day N pretest → read).
    let advanced = if task.task_type == TaskType::Pretest {
        let intermediate = chapter.status.complete(task.task_type)?;
        open_reading(intermediate)?
    } else {
        chapter.status.complete(task.task_type)?
    };
    let done = store.complete_task(task_id, completed_at)?;
    store.set_chapter_status(chapter.id, advanced)?;
    store.log_event(
        "TASK_COMPLETED",
        Some(chapter.id),
        Some(done.id),
        Some(task.task_type.as_str()),
        completed_at,
    )?;
    let known = store.list_tasks()?;
    let sequence = next_sequence(&known);
    let followup: Option<(TaskType, NaiveDate)> = match task.task_type {
        TaskType::Pretest => Some((TaskType::Read, today)),
        TaskType::Read => Some((TaskType::Retest, add_days(today, 1))),
        TaskType::Retest => Some((TaskType::AssignmentWrite, add_days(today, 1))),
        TaskType::AssignmentWrite => Some((TaskType::Notes, today)),
        TaskType::Notes => None,
    };
    if let Some((kind, due)) = followup {
        let next = store.create_task(&NewTask {
            book_id: task.book_id,
            chapter_id: chapter.id,
            task_type: kind,
            scheduled_for: due,
            sequence,
            attempt_no: chapter.attempt_no,
        })?;
        store.log_event(
            "TASK_SCHEDULED",
            Some(chapter.id),
            Some(next.id),
            Some(kind.as_str()),
            completed_at,
        )?;
        return Ok(Some(next));
    }
    // NOTES done: chapter is COMPLETED; unlock the next LOCKED chapter with
    // its first PRETEST due tomorrow (pullable today; skipped chapters stay
    // skipped).
    unlock_next_after_notes(store, &chapter, today, completed_at)
}

/// Skip a chapter (§4.1): status → `SKIPPED`, pending tasks deleted.
/// Completed work stays as an audit trail; misconceptions stay `ACTIVE`.
///
/// Idempotent: skipping an already-skipped chapter returns `Ok(0)`.
///
/// Returns the number of pending tasks removed.
///
/// # Errors
///
/// Returns [`Error::NotFound`] for unknown chapters and [`Error::Store`] on
/// backend failure.
pub fn skip_chapter(store: &mut dyn Store, chapter_id: i64, created_at: &str) -> Result<usize> {
    let chapter = store.get_chapter(chapter_id)?;
    if chapter.status == ChapterStatus::Skipped {
        return Ok(0);
    }
    store.set_chapter_status(chapter_id, ChapterStatus::Skipped)?;
    let removed = store.delete_pending_tasks_for_chapter(chapter_id)?;
    store.log_event(
        "CHAPTER_SKIPPED",
        Some(chapter_id),
        None,
        Some(chapter.title.as_str()),
        created_at,
    )?;
    Ok(removed)
}

/// Return a skipped chapter to the pipeline (§4.1): always restarts fresh —
/// status → `PRETEST_READY`, `attempt_no` increments, and a new `PRETEST` due
/// `today` is scheduled. Nothing from the abandoned attempt resumes (metrics
/// filter on the new attempt).
///
/// # Errors
///
/// Returns [`Error::InvalidTransition`] unless the chapter is `SKIPPED`,
/// [`Error::NotFound`] for unknown chapters, and [`Error::Store`] on backend
/// or `attempt_no` overflow failure.
pub fn unskip_chapter(
    store: &mut dyn Store,
    chapter_id: i64,
    today: NaiveDate,
    created_at: &str,
) -> Result<Task> {
    let chapter = store.get_chapter(chapter_id)?;
    if chapter.status != ChapterStatus::Skipped {
        return Err(Error::InvalidTransition(format!(
            "cannot unskip chapter {} while it is {}",
            chapter_id,
            chapter.status.as_str()
        )));
    }
    let attempt = chapter
        .attempt_no
        .checked_add(1)
        .ok_or_else(|| Error::Store("chapter attempt_no overflow".to_string()))?;
    store.set_chapter_attempt(chapter_id, attempt)?;
    store.set_chapter_status(chapter_id, ChapterStatus::PretestReady)?;
    store.log_event(
        "CHAPTER_UNSKIPPED",
        Some(chapter_id),
        None,
        Some(chapter.title.as_str()),
        created_at,
    )?;
    let known = store.list_tasks()?;
    let task = store.create_task(&NewTask {
        book_id: chapter.book_id,
        chapter_id,
        task_type: TaskType::Pretest,
        scheduled_for: today,
        sequence: next_sequence(&known),
        attempt_no: attempt,
    })?;
    store.log_event(
        "TASK_SCHEDULED",
        Some(chapter_id),
        Some(task.id),
        Some(TaskType::Pretest.as_str()),
        created_at,
    )?;
    Ok(task)
}

/// One row of the 3-day sliding-window projection (§5) for chapter `index`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowRow {
    /// Zero-based chapter offset (0 = chapter K).
    pub chapter_offset: i64,
    /// Day offset from N (0, 1, 2).
    pub day_offset: i64,
    /// Planned activity label.
    pub activity: String,
}

/// Project the §5 window: Day N pretest+read, Day N+1 retest, Day N+2
/// assignment+grading+notes, for `chapters` consecutive chapters.
///
/// # Errors
///
/// This function is infallible by construction but returns `Result` to keep
/// the engine interface uniform; it always returns `Ok`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "infallible by construction; Result keeps the engine interface uniform"
)]
pub fn project_window(
    chapters: i64,
    start: NaiveDate,
) -> crate::error::Result<Vec<(NaiveDate, WindowRow)>> {
    let mut out = Vec::new();
    for offset in 0..chapters {
        let Some(day_n) =
            start.checked_add_days(chrono::Days::new(u64::try_from(offset).unwrap_or_default()))
        else {
            continue;
        };
        let Some(day_n1) = day_n.checked_add_days(chrono::Days::new(1)) else {
            continue;
        };
        let Some(day_n2) = day_n.checked_add_days(chrono::Days::new(2)) else {
            continue;
        };
        out.push((
            day_n,
            WindowRow {
                chapter_offset: offset,
                day_offset: 0,
                activity: "Pretest + Read".to_string(),
            },
        ));
        out.push((
            day_n1,
            WindowRow {
                chapter_offset: offset,
                day_offset: 1,
                activity: "Retest".to_string(),
            },
        ));
        out.push((
            day_n2,
            WindowRow {
                chapter_offset: offset,
                day_offset: 2,
                activity: "Assignment + Grading + Notes".to_string(),
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{TaskStatus, TaskType};
    use crate::store::{MemoryStore, NewBook, NewChapter};

    fn task(kind: TaskType, date: &str, status: TaskStatus, seq: i64) -> Task {
        Task {
            id: seq,
            book_id: 1,
            chapter_id: seq,
            task_type: kind,
            scheduled_for: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            status,
            completed_at: None,
            sequence: seq,
            attempt_no: 1,
        }
    }

    #[test]
    fn classification_buckets() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let overdue = task(TaskType::Retest, "2026-01-09", TaskStatus::Pending, 1);
        let due = task(TaskType::Retest, "2026-01-10", TaskStatus::Pending, 2);
        let future = task(TaskType::Retest, "2026-01-11", TaskStatus::Pending, 3);
        let done = task(TaskType::Retest, "2026-01-09", TaskStatus::Done, 4);
        assert_eq!(classify_task(&overdue, today), TaskBucket::Overdue);
        assert_eq!(classify_task(&due, today), TaskBucket::DueToday);
        assert_eq!(classify_task(&future, today), TaskBucket::Future);
        assert_eq!(classify_task(&done, today), TaskBucket::Done);
    }

    #[test]
    fn priority_orders_overdue_retest_first() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let tasks = vec![
            task(TaskType::Pretest, "2026-01-10", TaskStatus::Pending, 1),
            task(
                TaskType::AssignmentWrite,
                "2026-01-09",
                TaskStatus::Pending,
                2,
            ),
            task(TaskType::Retest, "2026-01-09", TaskStatus::Pending, 3),
            task(TaskType::Retest, "2026-01-10", TaskStatus::Pending, 4),
        ];
        let queue = today_queue(&tasks, today);
        assert_eq!(queue[0].task_type, TaskType::Retest);
        assert_eq!(queue[0].scheduled_for.to_string(), "2026-01-09");
        assert_eq!(queue[1].task_type, TaskType::AssignmentWrite);
    }

    #[test]
    fn overdue_excludes_today_done_and_future() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let tasks = vec![
            task(TaskType::Retest, "2026-01-09", TaskStatus::Pending, 1),
            task(TaskType::Retest, "2026-01-10", TaskStatus::Pending, 2),
            task(TaskType::Retest, "2026-01-08", TaskStatus::Done, 3),
            task(TaskType::Retest, "2026-01-11", TaskStatus::Pending, 4),
        ];
        let over = overdue_tasks(&tasks, today);
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].sequence, 1);
    }

    #[test]
    fn sequence_continues_past_max() {
        assert_eq!(next_sequence(&[]), 1);
        let tasks = vec![
            task(TaskType::Pretest, "2026-01-10", TaskStatus::Pending, 3),
            task(TaskType::Pretest, "2026-01-10", TaskStatus::Pending, 7),
        ];
        assert_eq!(next_sequence(&tasks), 8);
    }

    #[test]
    fn pending_matching_requires_all_fields() {
        let tasks = vec![task(
            TaskType::Pretest,
            "2026-01-10",
            TaskStatus::Pending,
            7,
        )];
        assert!(has_pending_for(&tasks, 7, 1, TaskType::Pretest));
        assert!(!has_pending_for(&tasks, 8, 1, TaskType::Pretest));
        assert!(!has_pending_for(&tasks, 7, 2, TaskType::Pretest));
        assert!(!has_pending_for(&tasks, 7, 1, TaskType::Retest));
        let done = vec![task(TaskType::Pretest, "2026-01-10", TaskStatus::Done, 7)];
        assert!(!has_pending_for(&done, 7, 1, TaskType::Pretest));
    }

    #[test]
    fn notes_beats_pretest_same_day() {
        // Regression: after grading, NOTES (chapter closure) must run before
        // the next chapter's PRETEST, not after it.
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let tasks = vec![
            task(TaskType::Pretest, "2026-01-10", TaskStatus::Pending, 1),
            task(TaskType::Notes, "2026-01-10", TaskStatus::Pending, 2),
            task(TaskType::Read, "2026-01-10", TaskStatus::Pending, 3),
        ];
        let queue = today_queue(&tasks, today);
        assert_eq!(queue[0].task_type, TaskType::Notes);
        assert_eq!(queue[1].task_type, TaskType::Pretest);
        assert_eq!(queue[2].task_type, TaskType::Read);
        // Assessments still outrank notes; overdue still dominates type.
        let mixed = vec![
            task(TaskType::Notes, "2026-01-10", TaskStatus::Pending, 1),
            task(TaskType::Retest, "2026-01-10", TaskStatus::Pending, 2),
            task(
                TaskType::AssignmentWrite,
                "2026-01-10",
                TaskStatus::Pending,
                3,
            ),
        ];
        let ordered = today_queue(&mixed, today);
        assert_eq!(ordered[0].task_type, TaskType::Retest);
        assert_eq!(ordered[1].task_type, TaskType::AssignmentWrite);
        assert_eq!(ordered[2].task_type, TaskType::Notes);
    }

    #[test]
    fn pull_requires_empty_mandatory_queue() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let pending = vec![task(
            TaskType::Pretest,
            "2026-01-10",
            TaskStatus::Pending,
            1,
        )];
        assert!(!pull_available(&pending, today));
        let clear = vec![task(
            TaskType::Pretest,
            "2026-01-11",
            TaskStatus::Pending,
            1,
        )];
        assert!(pull_available(&clear, today));
    }

    #[test]
    fn pull_candidate_selects_pretest_only() {
        // Pull starts new learning early: an earlier retest or assignment
        // never beats a later pretest — their spacing is the pedagogy.
        let today = date("2026-01-10");
        let tasks = vec![
            task(TaskType::Retest, "2026-01-11", TaskStatus::Pending, 1),
            task(
                TaskType::AssignmentWrite,
                "2026-01-11",
                TaskStatus::Pending,
                2,
            ),
            task(TaskType::Pretest, "2026-01-13", TaskStatus::Pending, 3),
        ];
        let next = next_pull_candidate(&tasks, today).unwrap();
        assert_eq!(next.task_type, TaskType::Pretest);
        assert_eq!(next.sequence, 3);
        // No future pretest at all: future retests alone pull nothing.
        let spaced = vec![
            task(TaskType::Retest, "2026-01-11", TaskStatus::Pending, 1),
            task(
                TaskType::AssignmentWrite,
                "2026-01-12",
                TaskStatus::Pending,
                2,
            ),
            task(TaskType::Notes, "2026-01-12", TaskStatus::Pending, 3),
        ];
        assert!(next_pull_candidate(&spaced, today).is_none());
    }

    #[test]
    fn pull_candidate_breaks_ties_by_date_then_sequence() {
        let today = date("2026-01-10");
        let tasks = vec![
            task(TaskType::Pretest, "2026-01-13", TaskStatus::Pending, 1),
            task(TaskType::Pretest, "2026-01-12", TaskStatus::Pending, 2),
            task(TaskType::Pretest, "2026-01-12", TaskStatus::Pending, 3),
        ];
        let next = next_pull_candidate(&tasks, today).unwrap();
        assert_eq!(next.sequence, 2);
    }

    #[test]
    fn pull_candidate_ignores_non_future_rows() {
        let today = date("2026-01-10");
        let tasks = vec![
            task(TaskType::Pretest, "2026-01-10", TaskStatus::Pending, 1),
            task(TaskType::Pretest, "2026-01-09", TaskStatus::Pending, 2),
            task(TaskType::Pretest, "2026-01-11", TaskStatus::Done, 3),
        ];
        assert!(next_pull_candidate(&tasks, today).is_none());
    }

    #[test]
    fn pull_next_rejects_mandatory_queue() {
        let mut store = MemoryStore::new();
        let (book, _, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let err = pull_next(&mut store, today, "2026-01-10").unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
    }

    #[test]
    fn pull_next_moves_next_pretest_to_today_then_gates() {
        let mut store = MemoryStore::new();
        let (book, _, second) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let pretest = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let read = complete_and_advance(&mut store, pretest[0].id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        let retest = complete_and_advance(&mut store, read.id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        assert_eq!(retest.scheduled_for, date("2026-01-11"));
        // Day N is closed: the chained pretest waits tomorrow while the
        // retest keeps its spaced day — pull takes the pretest, never the
        // retest.
        let chained = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(chained.len(), 1);
        assert_eq!(chained[0].task_type, TaskType::Pretest);
        let (pulled, previous) = pull_next(&mut store, today, "2026-01-10").unwrap().unwrap();
        assert_eq!(pulled.task_type, TaskType::Pretest);
        assert_eq!(pulled.chapter_id, second);
        assert_eq!(previous, date("2026-01-11"));
        assert_eq!(pulled.scheduled_for, today);
        // Pulled work is mandatory: a second pull gates until it is done.
        let gated = pull_next(&mut store, today, "2026-01-10").unwrap_err();
        assert!(matches!(gated, Error::InvalidTransition(_)));
    }

    #[test]
    fn pull_next_returns_none_without_future_pretest() {
        let mut store = MemoryStore::new();
        let today = date("2026-01-10");
        store
            .create_book(
                &NewBook {
                    title: "Empty".to_string(),
                    filepath: "/e.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 1,
                },
                "2026-01-10",
            )
            .unwrap();
        assert!(
            pull_next(&mut store, today, "2026-01-10")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn pull_next_leaves_spaced_retest_alone() {
        // Day-N pretest+read done, next chapter still locked behind the gate:
        // the only future work is the spaced retest — pull finds no pretest
        // and moves nothing.
        let mut store = MemoryStore::new();
        let (book, _, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let pretest = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let read = complete_and_advance(&mut store, pretest[0].id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        let retest = complete_and_advance(&mut store, read.id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        assert_eq!(retest.task_type, TaskType::Retest);
        assert!(
            pull_next(&mut store, today, "2026-01-10")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            store.get_task(retest.id).unwrap().scheduled_for,
            date("2026-01-11")
        );
    }

    #[test]
    fn pull_next_skips_stale_and_skipped_candidates() {
        use crate::store::NewTask;
        let mut store = MemoryStore::new();
        let (book, first, second) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let future = date("2026-01-12");
        // Abandoned attempt on a live chapter + pending work on a skipped
        // chapter: both invisible to pull (§4.1).
        store
            .create_task(&NewTask {
                book_id: book,
                chapter_id: second,
                task_type: TaskType::Retest,
                scheduled_for: future,
                sequence: 1,
                attempt_no: 99,
            })
            .unwrap();
        skip_chapter(&mut store, first, "2026-01-10").unwrap();
        store
            .create_task(&NewTask {
                book_id: book,
                chapter_id: first,
                task_type: TaskType::AssignmentWrite,
                scheduled_for: future,
                sequence: 2,
                attempt_no: 1,
            })
            .unwrap();
        // Even a future pretest on the skipped chapter stays invisible.
        store
            .create_task(&NewTask {
                book_id: book,
                chapter_id: first,
                task_type: TaskType::Pretest,
                scheduled_for: future,
                sequence: 3,
                attempt_no: 1,
            })
            .unwrap();
        assert!(
            pull_next(&mut store, today, "2026-01-10")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn window_projects_three_days_per_chapter() {
        let start = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let rows = project_window(2, start).unwrap();
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].1.activity, "Pretest + Read");
        assert_eq!(rows[2].1.activity, "Assignment + Grading + Notes");
    }

    fn two_chapter_book(store: &mut MemoryStore) -> (i64, i64, i64) {
        let book = store
            .create_book(
                &NewBook {
                    title: "T".to_string(),
                    filepath: "/t.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 1,
                },
                "2026-01-10",
            )
            .unwrap();
        let first = store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 1,
                end_page: 20,
                file_path: "u1.json".to_string(),
                status: ChapterStatus::PretestReady,
            })
            .unwrap();
        let second = store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 1,
                level: 1,
                title: "Ch 2".to_string(),
                start_page: 21,
                end_page: 40,
                file_path: "u2.json".to_string(),
                status: ChapterStatus::Locked,
            })
            .unwrap();
        (book.id, first.id, second.id)
    }

    fn date(text: &str) -> NaiveDate {
        NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn open_reading_only_from_pretest_complete() {
        assert_eq!(
            open_reading(ChapterStatus::PretestComplete).unwrap(),
            ChapterStatus::ReadAvailable
        );
        assert!(open_reading(ChapterStatus::ReadAvailable).is_err());
        assert!(open_reading(ChapterStatus::Locked).is_err());
    }

    #[test]
    fn ensure_creates_initial_pretest_idempotently() {
        let mut store = MemoryStore::new();
        let (book, first, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let made = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(made.len(), 1);
        assert_eq!(made[0].task_type, TaskType::Pretest);
        assert_eq!(made[0].scheduled_for, today);
        assert_eq!(made[0].chapter_id, first);
        // Rerun creates nothing.
        let again = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(again.len(), 0);
    }

    #[test]
    fn ensure_opens_stranded_pretest_complete() {
        let mut store = MemoryStore::new();
        let (book, first, _) = two_chapter_book(&mut store);
        store
            .set_chapter_status(first, ChapterStatus::PretestComplete)
            .unwrap();
        let today = date("2026-01-10");
        let made = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(
            store.get_chapter(first).unwrap().status,
            ChapterStatus::ReadAvailable
        );
        assert_eq!(made.len(), 1);
        assert_eq!(made[0].task_type, TaskType::Read);
    }

    #[test]
    fn advance_pretest_opens_reading_same_day() {
        let mut store = MemoryStore::new();
        let (book, first, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let made = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let next = complete_and_advance(&mut store, made[0].id, today, "2026-01-10").unwrap();
        assert_eq!(
            store.get_chapter(first).unwrap().status,
            ChapterStatus::ReadAvailable
        );
        let read = next.unwrap();
        assert_eq!(read.task_type, TaskType::Read);
        assert_eq!(read.scheduled_for, today);
    }

    #[test]
    fn advance_read_staggers_retest_tomorrow() {
        let mut store = MemoryStore::new();
        let (book, _, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let pretest = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let read = complete_and_advance(&mut store, pretest[0].id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        let retest = complete_and_advance(&mut store, read.id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        assert_eq!(retest.task_type, TaskType::Retest);
        assert_eq!(retest.scheduled_for, date("2026-01-11"));
    }

    #[test]
    fn advance_full_chain_unlocks_next_chapter() {
        let mut store = MemoryStore::new();
        let (book, first, second) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let pretest = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        let read = complete_and_advance(&mut store, pretest[0].id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        let retest = complete_and_advance(&mut store, read.id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        // Retest is due tomorrow; complete it then.
        let assign = complete_and_advance(&mut store, retest.id, date("2026-01-11"), "2026-01-11")
            .unwrap()
            .unwrap();
        assert_eq!(assign.task_type, TaskType::AssignmentWrite);
        assert_eq!(assign.scheduled_for, date("2026-01-12"));
        let notes = complete_and_advance(&mut store, assign.id, date("2026-01-12"), "2026-01-12")
            .unwrap()
            .unwrap();
        assert_eq!(notes.task_type, TaskType::Notes);
        let unlocked = complete_and_advance(&mut store, notes.id, date("2026-01-12"), "2026-01-12")
            .unwrap()
            .unwrap();
        assert_eq!(
            store.get_chapter(first).unwrap().status,
            ChapterStatus::Completed
        );
        assert_eq!(
            store.get_chapter(second).unwrap().status,
            ChapterStatus::PretestReady
        );
        assert_eq!(unlocked.task_type, TaskType::Pretest);
        assert_eq!(unlocked.chapter_id, second);
        // The freed chapter starts tomorrow (pullable today); Day N+2 stays
        // closed once its notes are done.
        assert_eq!(unlocked.scheduled_for, date("2026-01-13"));
    }

    #[test]
    fn ensure_chains_next_pretest_tomorrow() {
        // Day N pretest+read closes the day: the next chapter's first pretest
        // lands tomorrow, leaving today queueable-clear and pullable.
        let mut store = MemoryStore::new();
        let (book, first, second) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let pretest = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(pretest[0].scheduled_for, today);
        let read = complete_and_advance(&mut store, pretest[0].id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        assert_eq!(read.scheduled_for, today);
        let retest = complete_and_advance(&mut store, read.id, today, "2026-01-10")
            .unwrap()
            .unwrap();
        assert_eq!(retest.scheduled_for, date("2026-01-11"));
        let chained = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(chained.len(), 1);
        assert_eq!(chained[0].task_type, TaskType::Pretest);
        assert_eq!(chained[0].chapter_id, second);
        assert_eq!(chained[0].scheduled_for, date("2026-01-11"));
        assert_eq!(
            store.get_chapter(first).unwrap().status,
            ChapterStatus::ReadComplete
        );
        // Nothing due today remains for the finished chapter; rerun creates
        // nothing further.
        let again = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(again.len(), 0);
    }

    #[test]
    fn completing_done_task_is_rejected() {
        let mut store = MemoryStore::new();
        let (book, _, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let made = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        complete_and_advance(&mut store, made[0].id, today, "2026-01-10").unwrap();
        let err = complete_and_advance(&mut store, made[0].id, today, "2026-01-10").unwrap_err();
        assert!(matches!(err, Error::AlreadyCompleted(_)));
    }

    #[test]
    fn skip_deletes_pending_and_unskip_restarts_fresh() {
        let mut store = MemoryStore::new();
        let (book, first, _) = two_chapter_book(&mut store);
        let today = date("2026-01-10");
        let made = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(made.len(), 1);
        let removed = skip_chapter(&mut store, first, "2026-01-10").unwrap();
        assert_eq!(removed, 1);
        assert_eq!(
            store.get_chapter(first).unwrap().status,
            ChapterStatus::Skipped
        );
        // Idempotent reskip.
        assert_eq!(skip_chapter(&mut store, first, "2026-01-10").unwrap(), 0);
        // Skipped chapters get no tasks; the next non-skipped unit unlocks
        // and compacts forward (K+1 = next non-skipped, §4.1).
        let chapters = store.list_chapters(book).unwrap();
        assert_eq!(count_skipped(&chapters), 1);
        let none: Vec<crate::domain::Chapter> = Vec::new();
        assert_eq!(count_skipped(&none), 0);
        let compacted = ensure_tasks(&mut store, book, today, "2026-01-10").unwrap();
        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].task_type, TaskType::Pretest);
        // No live predecessor remains, so the compacted head starts today.
        assert_eq!(compacted[0].scheduled_for, today);
        // Unskip restarts fresh on attempt 2.
        let fresh = unskip_chapter(&mut store, first, today, "2026-01-10").unwrap();
        assert_eq!(fresh.task_type, TaskType::Pretest);
        assert_eq!(fresh.attempt_no, 2);
        assert_eq!(store.get_chapter(first).unwrap().attempt_no, 2);
        // Unskipping a live chapter fails loudly.
        assert!(unskip_chapter(&mut store, first, today, "2026-01-10").is_err());
    }

    #[test]
    fn chapter_start_prompt_marks_options() {
        let prompt = format_chapter_start_prompt("Level 1 Acquaintance", 38, 43);
        assert!(prompt.contains("[Y/s(kip)/n]"));
        assert!(prompt.contains("Level 1 Acquaintance"));
    }
}
