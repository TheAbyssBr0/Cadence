//! Persistence: [`Store`] trait plus in-memory and SQLite implementations.
//!
//! Engines receive a `&dyn Store` (or generic `Store`) via constructors and
//! never touch SQLite or the filesystem directly. Time is passed in by
//! callers; the store only persists what it is given.
//!
//! The SQLite implementation acquires an OS advisory lock (`cadence.lock`)
//! before opening the database so two processes never interleave writes.
//! Competing processes fail fast with [`Error::AlreadyOpen`].

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use fs2::FileExt;
use rusqlite::{Connection, OptionalExtension, params};

use crate::domain::{Book, Chapter, ChapterStatus, Task, TaskStatus, TaskType};
use crate::error::{Error, Result};
use crate::misconceptions::INITIAL_CONFIDENCE;

/// Current schema version stored in the `version` table.
pub const SCHEMA_VERSION: i64 = 6;

/// SQL applied on fresh databases (and used by tests to assert migration state).
pub const SCHEMA_SQL: &str = r"
CREATE TABLE IF NOT EXISTS version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS books (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  filepath TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  start_page INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chapters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id INTEGER NOT NULL REFERENCES books(id),
  index_in_book INTEGER NOT NULL,
  level INTEGER NOT NULL,
  title TEXT NOT NULL,
  start_page INTEGER NOT NULL,
  end_page INTEGER NOT NULL,
  file_path TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_id INTEGER NOT NULL REFERENCES books(id),
  chapter_id INTEGER NOT NULL REFERENCES chapters(id),
  type TEXT NOT NULL,
  scheduled_for TEXT NOT NULL,
  status TEXT NOT NULL,
  completed_at TEXT,
  sequence INTEGER NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS event_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  chapter_id INTEGER,
  task_id INTEGER,
  evidence TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL,
  raw_response TEXT,
  parsed_response TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS llm_cache (
  cache_hash TEXT PRIMARY KEY,
  operation TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  request_json TEXT NOT NULL,
  response_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mcq_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapters(id),
  phase TEXT NOT NULL,
  question_text TEXT NOT NULL,
  options_json TEXT NOT NULL,
  correct_index INTEGER NOT NULL,
  trap_index INTEGER NOT NULL,
  explanation_text TEXT NOT NULL,
  source_refs TEXT NOT NULL,
  topic TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS mcq_responses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id),
  selected_index INTEGER NOT NULL,
  is_correct INTEGER NOT NULL,
  selected_trap INTEGER NOT NULL,
  answered_at TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS misconceptions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapters(id),
  concept_description TEXT NOT NULL,
  description TEXT NOT NULL,
  evidence TEXT NOT NULL,
  source_task TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS assignment_questions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapters(id),
  position INTEGER NOT NULL,
  kind TEXT NOT NULL,
  parts_json TEXT NOT NULL,
  rubric_json TEXT NOT NULL,
  target_misconception_ids TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS assignment_responses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id INTEGER NOT NULL REFERENCES assignment_questions(id),
  answer_text TEXT NOT NULL,
  answered_at TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS grades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id INTEGER NOT NULL REFERENCES assignment_questions(id),
  score INTEGER NOT NULL,
  max_score INTEGER NOT NULL,
  classification TEXT NOT NULL,
  criteria_results_json TEXT NOT NULL,
  feedback TEXT NOT NULL,
  disputed INTEGER NOT NULL DEFAULT 0,
  original_score INTEGER,
  dispute_text TEXT,
  final_score INTEGER,
  adjudication_json TEXT,
  grader_version TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS disputes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  grade_id INTEGER NOT NULL REFERENCES grades(id),
  dispute_text TEXT NOT NULL,
  decision TEXT NOT NULL,
  final_score INTEGER NOT NULL,
  adjudicator_model TEXT NOT NULL,
  timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapters(id),
  content_markdown TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1
);
";

/// New-book input (no id yet).
#[derive(Debug, Clone)]
pub struct NewBook {
    /// Display title.
    pub title: String,
    /// Original PDF path.
    pub filepath: String,
    /// Hex SHA-256 of the PDF bytes.
    pub file_hash: String,
    /// One-based physical page where study content starts.
    pub start_page: i64,
}

/// New-chapter input (no id yet).
#[derive(Debug, Clone)]
pub struct NewChapter {
    /// Owning book id.
    pub book_id: i64,
    /// Zero-based position within the book.
    pub index_in_book: i64,
    /// Outline depth.
    pub level: i64,
    /// Heading text.
    pub title: String,
    /// One-based inclusive pages.
    pub start_page: i64,
    /// One-based inclusive pages.
    pub end_page: i64,
    /// JSON file holding sanitized text + provenance.
    pub file_path: String,
    /// Initial lifecycle state.
    pub status: ChapterStatus,
}

/// New-task input (no id yet).
#[derive(Debug, Clone)]
pub struct NewTask {
    /// Owning book id.
    pub book_id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Kind of work.
    pub task_type: TaskType,
    /// Calendar due date.
    pub scheduled_for: NaiveDate,
    /// Creation order for stable tie-breaking.
    pub sequence: i64,
    /// Attempt number copied from the chapter at creation (§4.1).
    pub attempt_no: i64,
}

/// New LLM job input (no id yet). One row per request (§16): retries update
/// the same row's `attempt_count` so the audit trail survives crashes.
#[derive(Debug, Clone)]
pub struct NewLlmJob {
    /// Stage label (`smoke`, `pretest`, `retest`, ...).
    pub operation: String,
    /// Provider id (`ai-gateway`).
    pub provider: String,
    /// Model id (`poolside/laguna-s-2.1-free`, ...).
    pub model: String,
    /// Hex SHA-256 of the prompt + source material.
    pub input_hash: String,
    /// Prompt template version (bump on prompt changes).
    pub prompt_version: String,
}

/// One durable LLM job row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmJob {
    /// Row id.
    pub id: i64,
    /// Stage label.
    pub operation: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Hex SHA-256 of the prompt + source material.
    pub input_hash: String,
    /// Prompt template version.
    pub prompt_version: String,
    /// `PENDING` | `RUNNING` | `OK` | `FAILED`.
    pub status: String,
    /// Attempts so far (retries increment this).
    pub attempt_count: i64,
    /// Last raw response text, if any.
    pub raw_response: Option<String>,
    /// Last validated response text, if any.
    pub parsed_response: Option<String>,
    /// Last error text, if any.
    pub error: Option<String>,
}

/// One validated cache entry. Only validated responses are stored; reads must
/// re-run the caller's validator before use (shape + source membership).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmCacheEntry {
    /// `SHA256(provider + model + task_type + prompt_version +
    /// source_content_hash + parameters)` (§16).
    pub cache_hash: String,
    /// Stage label.
    pub operation: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Prompt template version.
    pub prompt_version: String,
    /// Canonical request JSON (prompt + params).
    pub request_json: String,
    /// Validated response JSON/text.
    pub response_json: String,
    /// `OK` (only validated rows are written).
    pub status: String,
    /// ISO date string.
    pub created_at: String,
}

/// One stored MCQ (§7.1): 4 generated options (1 correct, 1 trap,
/// 2 distractors) pre-shuffle; `E = "I don't know"` is hardcoded at display
/// time and never stored. `correct_index` / `trap_index` are pre-shuffle
/// positions into `options_json` (0–3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McqItem {
    /// Row id.
    pub id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// `pretest` | `retest` | `review`.
    pub phase: String,
    /// Question stem.
    pub question_text: String,
    /// JSON array of exactly 4 option strings (pre-shuffle).
    pub options_json: String,
    /// Index of the correct option pre-shuffle (0–3).
    pub correct_index: i64,
    /// Index of the designated trap pre-shuffle (0–3, never correct).
    pub trap_index: i64,
    /// 2–5 sentence explanation.
    pub explanation_text: String,
    /// JSON source references (`pages`, `sections`).
    pub source_refs: String,
    /// Topic label.
    pub topic: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// New-MCQ input (no id yet).
#[derive(Debug, Clone)]
pub struct NewMcqItem {
    /// Owning chapter id.
    pub chapter_id: i64,
    /// `pretest` | `retest` | `review`.
    pub phase: String,
    /// Question stem.
    pub question_text: String,
    /// JSON array of exactly 4 option strings.
    pub options_json: String,
    /// Correct option pre-shuffle (0–3).
    pub correct_index: i64,
    /// Trap option pre-shuffle (0–3).
    pub trap_index: i64,
    /// Explanation text.
    pub explanation_text: String,
    /// JSON source references.
    pub source_refs: String,
    /// Topic label.
    pub topic: String,
    /// Attempt number.
    pub attempt_no: i64,
}

/// One recorded MCQ answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McqResponse {
    /// Row id.
    pub id: i64,
    /// Answered item.
    pub mcq_item_id: i64,
    /// Displayed index selected (0–4, where 4 is always `"I don't know"`).
    pub selected_index: i64,
    /// Whether the selection was correct.
    pub is_correct: bool,
    /// Whether the designated trap was selected.
    pub selected_trap: bool,
    /// ISO timestamp.
    pub answered_at: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// One tracked misconception (§12).
#[derive(Debug, Clone)]
pub struct Misconception {
    /// Row id.
    pub id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Short concept label.
    pub concept_description: String,
    /// Fuller description.
    pub description: String,
    /// User's wrong answer / explanation.
    pub evidence: String,
    /// `RETEST` | `ASSIGNMENT`.
    pub source_task: String,
    /// `ACTIVE` | `IMPROVING` | `RESOLVED` | `DISPUTED`.
    pub status: String,
    /// Confidence 0.0–1.0.
    pub confidence: f64,
    /// ISO timestamps.
    pub created_at: String,
    /// ISO timestamps.
    pub updated_at: String,
    /// ISO timestamp when resolved, if ever.
    pub resolved_at: Option<String>,
}

/// One stored assignment question (§7.2): a written multi-part question or
/// the single coding question, with its creation-time rubric. `kind` is
/// `written` | `coding`; `position` is zero-based display order; rubrics and
/// model solutions are frozen at creation time, decoupled from grading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentQuestion {
    /// Row id.
    pub id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Zero-based position within the set.
    pub position: i64,
    /// `written` | `coding`.
    pub kind: String,
    /// JSON array of part strings (`a) …`, `b) …`, …).
    pub parts_json: String,
    /// JSON rubric (`criteria`, `max_score`, `model_solution`).
    pub rubric_json: String,
    /// JSON array of misconception ids this question re-probes.
    pub target_misconception_ids: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// New-assignment-question input (no id yet).
#[derive(Debug, Clone)]
pub struct NewAssignmentQuestion {
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Zero-based position within the set.
    pub position: i64,
    /// `written` | `coding`.
    pub kind: String,
    /// JSON array of part strings.
    pub parts_json: String,
    /// JSON rubric.
    pub rubric_json: String,
    /// JSON array of misconception ids re-probed.
    pub target_misconception_ids: String,
    /// Attempt number.
    pub attempt_no: i64,
}

/// One written assignment answer (closed-book, via `nvim`, §7.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentResponse {
    /// Row id.
    pub id: i64,
    /// Answered question.
    pub question_id: i64,
    /// Full answer text as saved from the editor buffer.
    pub answer_text: String,
    /// ISO timestamp.
    pub answered_at: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// New-grade input (no id yet): one rubric judgment for an answer (§7.3/§10).
/// The `score` award is the original forever — disputes write `final_score`
/// and never overwrite it (§9 audit trail).
#[derive(Debug, Clone)]
pub struct NewGrade {
    /// Graded assignment question.
    pub question_id: i64,
    /// Points awarded (`0 <= score <= max_score`).
    pub score: i64,
    /// Rubric total.
    pub max_score: i64,
    /// §10 verdict label (`CORRECT`, `INCORRECT`, ...).
    pub classification: String,
    /// JSON per-criterion breakdown.
    pub criteria_results_json: String,
    /// Technical feedback for the student.
    pub feedback: String,
    /// Grader version (prompt version that produced the grade).
    pub grader_version: String,
    /// ISO timestamp.
    pub created_at: String,
}

/// One stored grade (§17): the original award plus the dispute trail.
/// `disputed` starts false; [`Store::record_dispute`] flips it and fills the
/// `original_*` / `final_*` fields without touching `score`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grade {
    /// Row id.
    pub id: i64,
    /// Graded assignment question.
    pub question_id: i64,
    /// Original points awarded (never overwritten after a dispute).
    pub score: i64,
    /// Rubric total.
    pub max_score: i64,
    /// §10 verdict label.
    pub classification: String,
    /// JSON per-criterion breakdown.
    pub criteria_results_json: String,
    /// Technical feedback for the student.
    pub feedback: String,
    /// True once a dispute audit has been recorded.
    pub disputed: bool,
    /// Score at first dispute time (the preserved original).
    pub original_score: Option<i64>,
    /// Student's dispute text, if disputed.
    pub dispute_text: Option<String>,
    /// Corrected total after dispute, if disputed.
    pub final_score: Option<i64>,
    /// Validated auditor JSON payload, if disputed.
    pub adjudication_json: Option<String>,
    /// Grader version.
    pub grader_version: String,
    /// ISO timestamp.
    pub created_at: String,
}

/// New-dispute input (no id yet): one §9 audit outcome for a grade.
#[derive(Debug, Clone)]
pub struct NewDispute {
    /// Audited grade.
    pub grade_id: i64,
    /// Student's dispute text.
    pub text: String,
    /// `REVISED` | `UPHELD` | `QUESTION_DEFECTIVE`.
    pub decision: String,
    /// Corrected total (`0 <= final_score <= grade max`).
    pub final_score: i64,
    /// Validated auditor JSON payload (stored on the grade row).
    pub adjudication_json: String,
    /// Model that produced the audit.
    pub adjudicator_model: String,
    /// ISO timestamp.
    pub timestamp: String,
}

/// One stored dispute audit (§17): the inspectable trail behind a grade
/// correction. The original grade is never overwritten — see [`Grade`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispute {
    /// Row id.
    pub id: i64,
    /// Audited grade.
    pub grade_id: i64,
    /// Student's dispute text.
    pub text: String,
    /// `REVISED` | `UPHELD` | `QUESTION_DEFECTIVE`.
    pub decision: String,
    /// Corrected total.
    pub final_score: i64,
    /// Model that produced the audit.
    pub adjudicator_model: String,
    /// ISO timestamp.
    pub timestamp: String,
}

/// New-notes input (no id yet): one post-grading synthesis document (§11).
/// Like MCQs and assignments the row carries `attempt_no` (§4.1): an unskip
/// restarts the chapter on a fresh attempt and synthesizes fresh notes, while
/// the abandoned attempt's notes stay as an audit trail.
#[derive(Debug, Clone)]
pub struct NewNote {
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Full personalized markdown (`Chapter_<N>_Notes.md` content, §11).
    pub content_markdown: String,
    /// ISO timestamp.
    pub generated_at: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// One stored chapter-notes document (§17).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    /// Row id.
    pub id: i64,
    /// Owning chapter id.
    pub chapter_id: i64,
    /// Full personalized markdown (§11).
    pub content_markdown: String,
    /// ISO timestamp.
    pub generated_at: String,
    /// Attempt number (§4.1).
    pub attempt_no: i64,
}

/// Persistence abstraction. Implementations must make every state transition
/// atomic from the caller's perspective (SQLite uses transactions internally;
/// the memory store mutates a single map entry).
pub trait Store {
    /// Persist a new book and return it with its assigned id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn create_book(&mut self, input: &NewBook, created_at: &str) -> Result<Book>;
    /// Fetch a book by id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent, [`Error::Store`] on failure.
    fn get_book(&self, id: i64) -> Result<Book>;
    /// All books in insertion (`id`) order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_books(&self) -> Result<Vec<Book>>;
    /// Persist a new chapter and return it with its assigned id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn create_chapter(&mut self, input: &NewChapter) -> Result<Chapter>;
    /// Fetch a chapter by id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn get_chapter(&self, id: i64) -> Result<Chapter>;
    /// All chapters of a book in `index_in_book` order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_chapters(&self, book_id: i64) -> Result<Vec<Chapter>>;
    /// Update a chapter's lifecycle status.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn set_chapter_status(&mut self, id: i64, status: ChapterStatus) -> Result<()>;
    /// Update a chapter's attempt number (§4.1 unskip increments it).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn set_chapter_attempt(&mut self, id: i64, attempt_no: i64) -> Result<()>;
    /// Delete all `PENDING` tasks for a chapter (skip path). Returns the
    /// number of rows removed; completed work stays as an audit trail.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn delete_pending_tasks_for_chapter(&mut self, chapter_id: i64) -> Result<usize>;
    /// Persist a new `PENDING` task.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn create_task(&mut self, input: &NewTask) -> Result<Task>;
    /// Fetch a task by id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn get_task(&self, id: i64) -> Result<Task>;
    /// All tasks, ordered by `sequence`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_tasks(&self) -> Result<Vec<Task>>;
    /// Mark a `PENDING` task `DONE`. Completing an already-`DONE` task
    /// returns [`Error::AlreadyCompleted`] without mutating state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn complete_task(&mut self, id: i64, completed_at: &str) -> Result<Task>;
    /// Move a `PENDING` task to a new calendar date (pull path, §6).
    /// Rescheduling an already-`DONE` task returns
    /// [`Error::AlreadyCompleted`] without mutating state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn reschedule_task(&mut self, id: i64, scheduled_for: NaiveDate) -> Result<Task>;
    /// Append an event-log row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn log_event(
        &mut self,
        event_type: &str,
        chapter_id: Option<i64>,
        task_id: Option<i64>,
        evidence: Option<&str>,
        created_at: &str,
    ) -> Result<()>;
    /// Insert a new `PENDING` LLM job row and return it with its id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn create_llm_job(&mut self, input: &NewLlmJob) -> Result<LlmJob>;
    /// Record one attempt against a job row (status + attempt count + latest
    /// payloads). Terminal states are `OK` and `FAILED`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn record_llm_attempt(
        &mut self,
        id: i64,
        attempt_count: i64,
        status: &str,
        raw_response: Option<&str>,
        parsed_response: Option<&str>,
        error: Option<&str>,
    ) -> Result<()>;
    /// Fetch a cache entry by its identity hash.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn get_llm_cache(&self, cache_hash: &str) -> Result<Option<LlmCacheEntry>>;
    /// Insert or replace a validated cache entry (callers validate first;
    /// malformed responses are never cached).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn put_llm_cache(&mut self, entry: &LlmCacheEntry) -> Result<()>;
    /// Persist validated MCQ items for a chapter/phase/attempt.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn save_mcq_items(&mut self, items: &[NewMcqItem]) -> Result<Vec<McqItem>>;
    /// Discard stored MCQ items for a chapter/phase/attempt (fresh `--generation`
    /// experiments replace the previous set). Responses attached to the removed
    /// items are deleted too so no orphans linger. Returns rows removed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn delete_mcq_items_for(&mut self, chapter_id: i64, phase: &str, attempt_no: i64) -> Result<usize>;
    /// All MCQ items for a chapter/phase/attempt, in insertion order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_mcq_items(&self, chapter_id: i64, phase: &str, attempt_no: i64) -> Result<Vec<McqItem>>;
    /// Record one MCQ answer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn record_mcq_response(
        &mut self,
        mcq_item_id: i64,
        selected_index: i64,
        is_correct: bool,
        selected_trap: bool,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<McqResponse>;
    /// All responses for one MCQ item (insertion order).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_mcq_responses(&self, mcq_item_id: i64) -> Result<Vec<McqResponse>>;
    /// Log a misconception (§12). Only incorrect retest answers (excluding
    /// `"I don't know"`) reach this path; callers enforce that rule.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    #[allow(clippy::too_many_arguments)]
    fn create_misconception(
        &mut self,
        chapter_id: i64,
        concept: &str,
        description: &str,
        evidence: &str,
        source_task: &str,
        created_at: &str,
    ) -> Result<Misconception>;
    /// All misconceptions for a chapter (insertion order).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_misconceptions(&self, chapter_id: i64) -> Result<Vec<Misconception>>;
    /// Apply one lifecycle step to a misconception (§12): confidence,
    /// status, and timestamps. `resolved_at` replaces the stored value, so
    /// callers preserve the existing timestamp except when the step resolves
    /// the row (see `misconceptions::Transition::just_resolved`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for an unknown id, [`Error::InvalidInput`]
    /// for out-of-range confidence or an unknown status, [`Error::Store`] on
    /// backend failure.
    fn update_misconception(
        &mut self,
        id: i64,
        confidence: f64,
        status: &str,
        updated_at: &str,
        resolved_at: Option<&str>,
    ) -> Result<Misconception>;
    /// Persist validated assignment questions for a chapter/attempt (§7.2).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn save_assignment_questions(
        &mut self,
        items: &[NewAssignmentQuestion],
    ) -> Result<Vec<AssignmentQuestion>>;
    /// All assignment questions for a chapter/attempt, in `position` order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_assignment_questions(
        &self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<Vec<AssignmentQuestion>>;
    /// Discard stored assignment questions for a chapter/attempt (fresh
    /// experiments replace the previous set). Responses attached to the
    /// removed questions go with them so no orphans linger. Returns rows
    /// removed.
    ///
    /// Retained API: no production flow regenerates assignments yet (reruns
    /// resume stored rows), but the contract is covered by unit tests.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    #[allow(dead_code)]
    fn delete_assignment_questions_for(
        &mut self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<usize>;
    /// Record one written answer from the editor buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn record_assignment_response(
        &mut self,
        question_id: i64,
        answer_text: &str,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<AssignmentResponse>;
    /// All responses for one assignment question (insertion order).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_assignment_responses(&self, question_id: i64) -> Result<Vec<AssignmentResponse>>;
    /// Persist one rubric grade for an assignment answer (§7.3/§10). The award
    /// is the original of record; later disputes correct via `final_score`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the score is out of bounds or the
    /// classification is blank, [`Error::Store`] on backend failure.
    fn save_grade(&mut self, input: &NewGrade) -> Result<Grade>;
    /// Fetch a grade by id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when absent.
    fn get_grade(&self, id: i64) -> Result<Grade>;
    /// All grades for one assignment question (insertion order; the latest is
    /// the grade of record).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_grades_for_question(&self, question_id: i64) -> Result<Vec<Grade>>;
    /// Record one §9 dispute audit: flips the grade to disputed (preserving
    /// the first original score), stores the auditor payload, and appends a
    /// `disputes` trail row. Returns the new dispute row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for an unknown grade,
    /// [`Error::InvalidInput`] for a bad decision, out-of-bounds score, or
    /// blank dispute text.
    fn record_dispute(&mut self, input: &NewDispute) -> Result<Dispute>;
    /// All dispute audits for one grade (insertion order).
    ///
    /// Retained API: production reports the latest audit inline, but the
    /// full trail query is covered by unit tests.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    #[allow(dead_code)]
    fn list_disputes_for_grade(&self, grade_id: i64) -> Result<Vec<Dispute>>;
    /// Persist one post-grading chapter-notes document (§11).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the markdown is blank,
    /// [`Error::Store`] on backend failure.
    fn save_note(&mut self, input: &NewNote) -> Result<Note>;
    /// All notes for a chapter/attempt, in insertion order (the latest is
    /// the notes of record; generate-or-resume reuses it on rerun).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn list_notes(&self, chapter_id: i64, attempt_no: i64) -> Result<Vec<Note>>;
    /// Factory reset for single-book focus (§3): delete every library record
    /// (books, chapters, tasks, events, jobs, MCQ, misconceptions,
    /// assignments, grades, disputes, notes) and restart id sequences, so a
    /// newly ingested book starts from a clean slate. The schema `version`
    /// row and the content-addressed `llm_cache` survive (neither references
    /// a book); unit-text files under `books/` are removed by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on backend failure.
    fn clear_library(&mut self) -> Result<()>;
}

/// In-memory implementation for unit tests and `dev` commands (never
/// production storage).
#[derive(Debug, Default)]
pub struct MemoryStore {
    books: HashMap<i64, Book>,
    chapters: HashMap<i64, Chapter>,
    tasks: HashMap<i64, Task>,
    llm_jobs: HashMap<i64, LlmJob>,
    llm_cache: HashMap<String, LlmCacheEntry>,
    mcq_items: HashMap<i64, McqItem>,
    mcq_responses: HashMap<i64, McqResponse>,
    misconceptions: HashMap<i64, Misconception>,
    assignment_questions: HashMap<i64, AssignmentQuestion>,
    assignment_responses: HashMap<i64, AssignmentResponse>,
    grades: HashMap<i64, Grade>,
    disputes: HashMap<i64, Dispute>,
    notes: HashMap<i64, Note>,
    next_id: i64,
}

impl MemoryStore {
    /// Empty store; ids start at 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            books: HashMap::new(),
            chapters: HashMap::new(),
            tasks: HashMap::new(),
            llm_jobs: HashMap::new(),
            llm_cache: HashMap::new(),
            mcq_items: HashMap::new(),
            mcq_responses: HashMap::new(),
            misconceptions: HashMap::new(),
            assignment_questions: HashMap::new(),
            assignment_responses: HashMap::new(),
            grades: HashMap::new(),
            disputes: HashMap::new(),
            notes: HashMap::new(),
            next_id: 1,
        }
    }

    /// Allocate the next row id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on `i64` overflow (practically unreachable).
    fn alloc_id(&mut self) -> Result<i64> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            Error::Store("id sequence overflow".to_string())
        })?;
        Ok(id)
    }
}

/// Reject grade rows that could silently misrecord an award (shared by both
/// backends so file and memory stores enforce the same bounds).
fn check_grade_input(input: &NewGrade) -> Result<()> {
    if input.max_score < 1 {
        return Err(Error::InvalidInput(
            "grade max_score must be >= 1".to_string(),
        ));
    }
    if input.score < 0 || input.score > input.max_score {
        return Err(Error::InvalidInput(format!(
            "grade score {} outside 0-{}",
            input.score, input.max_score
        )));
    }
    if input.classification.trim().is_empty() {
        return Err(Error::InvalidInput(
            "grade needs a classification".to_string(),
        ));
    }
    Ok(())
}

/// Reject dispute rows with unknown decisions or blank text (shared by both
/// backends; score bounds need the grade row, so each backend checks those).
fn check_dispute_input(input: &NewDispute) -> Result<()> {
    if input.decision != "REVISED"
        && input.decision != "UPHELD"
        && input.decision != "QUESTION_DEFECTIVE"
    {
        return Err(Error::InvalidInput(format!(
            "dispute decision must be REVISED|UPHELD|QUESTION_DEFECTIVE, got '{}'",
            input.decision
        )));
    }
    if input.text.trim().is_empty() {
        return Err(Error::InvalidInput(
            "dispute needs dispute text".to_string(),
        ));
    }
    if input.adjudication_json.trim().is_empty() {
        return Err(Error::InvalidInput(
            "dispute needs the auditor payload".to_string(),
        ));
    }
    Ok(())
}

/// Reject blank chapter notes (shared by both backends): an empty synthesis
/// is a generation failure, never a storable artifact.
fn check_note_input(input: &NewNote) -> Result<()> {
    if input.content_markdown.trim().is_empty() {
        return Err(Error::InvalidInput(
            "notes need non-empty markdown".to_string(),
        ));
    }
    Ok(())
}

/// Reject bad lifecycle steps (shared by both backends): confidence must be
/// a real number in `0.0–1.0` and status one of the four §12 labels.
fn check_misconception_update(confidence: f64, status: &str) -> Result<()> {
    if !(0.0..=1.0).contains(&confidence) {
        return Err(Error::InvalidInput(format!(
            "misconception confidence {confidence} outside 0.0-1.0"
        )));
    }
    if status != "ACTIVE" && status != "IMPROVING" && status != "RESOLVED" && status != "DISPUTED" {
        return Err(Error::InvalidInput(format!(
            "misconception status must be ACTIVE|IMPROVING|RESOLVED|DISPUTED, got '{status}'"
        )));
    }
    Ok(())
}

impl Store for MemoryStore {
    fn create_book(&mut self, input: &NewBook, created_at: &str) -> Result<Book> {
        let id = self.alloc_id()?;
        let book = Book {
            id,
            title: input.title.clone(),
            filepath: input.filepath.clone(),
            file_hash: input.file_hash.clone(),
            start_page: input.start_page,
            created_at: created_at.to_string(),
        };
        self.books.insert(id, book.clone());
        Ok(book)
    }

    fn get_book(&self, id: i64) -> Result<Book> {
        self.books
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("book {id}")))
    }

    fn list_books(&self) -> Result<Vec<Book>> {
        let mut out: Vec<Book> = self.books.values().cloned().collect();
        out.sort_by_key(|b| b.id);
        Ok(out)
    }

    fn create_chapter(&mut self, input: &NewChapter) -> Result<Chapter> {
        let id = self.alloc_id()?;
        let chapter = Chapter {
            id,
            book_id: input.book_id,
            index_in_book: input.index_in_book,
            level: input.level,
            title: input.title.clone(),
            start_page: input.start_page,
            end_page: input.end_page,
            file_path: input.file_path.clone(),
            status: input.status,
            attempt_no: 1,
        };
        self.chapters.insert(id, chapter.clone());
        Ok(chapter)
    }

    fn get_chapter(&self, id: i64) -> Result<Chapter> {
        self.chapters
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("chapter {id}")))
    }

    fn list_chapters(&self, book_id: i64) -> Result<Vec<Chapter>> {
        let mut out: Vec<Chapter> = self
            .chapters
            .values()
            .filter(|c| c.book_id == book_id)
            .cloned()
            .collect();
        out.sort_by(|a, b| {
            a.index_in_book
                .cmp(&b.index_in_book)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(out)
    }

    fn set_chapter_status(&mut self, id: i64, status: ChapterStatus) -> Result<()> {
        let chapter = self
            .chapters
            .get_mut(&id)
            .ok_or_else(|| Error::NotFound(format!("chapter {id}")))?;
        chapter.status = status;
        Ok(())
    }

    fn set_chapter_attempt(&mut self, id: i64, attempt_no: i64) -> Result<()> {
        if attempt_no < 1 {
            return Err(Error::InvalidInput(
                "attempt_no must be >= 1".to_string(),
            ));
        }
        let chapter = self
            .chapters
            .get_mut(&id)
            .ok_or_else(|| Error::NotFound(format!("chapter {id}")))?;
        chapter.attempt_no = attempt_no;
        Ok(())
    }

    fn delete_pending_tasks_for_chapter(&mut self, chapter_id: i64) -> Result<usize> {
        let pending: Vec<i64> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.chapter_id == chapter_id && t.status == TaskStatus::Pending)
            .map(|(id, _)| *id)
            .collect();
        let count = pending.len();
        for id in pending {
            self.tasks.remove(&id);
        }
        Ok(count)
    }

    fn create_task(&mut self, input: &NewTask) -> Result<Task> {
        if input.attempt_no < 1 {
            return Err(Error::InvalidInput(
                "task attempt_no must be >= 1".to_string(),
            ));
        }
        let id = self.alloc_id()?;
        let task = Task {
            id,
            book_id: input.book_id,
            chapter_id: input.chapter_id,
            task_type: input.task_type,
            scheduled_for: input.scheduled_for,
            status: TaskStatus::Pending,
            completed_at: None,
            sequence: input.sequence,
            attempt_no: input.attempt_no,
        };
        self.tasks.insert(id, task.clone());
        Ok(task)
    }

    fn get_task(&self, id: i64) -> Result<Task> {
        self.tasks
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("task {id}")))
    }

    fn list_tasks(&self) -> Result<Vec<Task>> {
        let mut out: Vec<Task> = self.tasks.values().cloned().collect();
        out.sort_by(|a, b| a.sequence.cmp(&b.sequence).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    fn complete_task(&mut self, id: i64, completed_at: &str) -> Result<Task> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| Error::NotFound(format!("task {id}")))?;
        if task.status == TaskStatus::Done {
            return Err(Error::AlreadyCompleted(format!("task {id}")));
        }
        task.status = TaskStatus::Done;
        task.completed_at = Some(completed_at.to_string());
        Ok(task.clone())
    }

    fn reschedule_task(&mut self, id: i64, scheduled_for: NaiveDate) -> Result<Task> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| Error::NotFound(format!("task {id}")))?;
        if task.status == TaskStatus::Done {
            return Err(Error::AlreadyCompleted(format!("task {id}")));
        }
        task.scheduled_for = scheduled_for;
        Ok(task.clone())
    }

    fn log_event(
        &mut self,
        _event_type: &str,
        _chapter_id: Option<i64>,
        _task_id: Option<i64>,
        _evidence: Option<&str>,
        _created_at: &str,
    ) -> Result<()> {
        Ok(())
    }

    fn create_llm_job(&mut self, input: &NewLlmJob) -> Result<LlmJob> {
        let id = self.alloc_id()?;
        let job = LlmJob {
            id,
            operation: input.operation.clone(),
            provider: input.provider.clone(),
            model: input.model.clone(),
            input_hash: input.input_hash.clone(),
            prompt_version: input.prompt_version.clone(),
            status: "PENDING".to_string(),
            attempt_count: 0,
            raw_response: None,
            parsed_response: None,
            error: None,
        };
        self.llm_jobs.insert(id, job.clone());
        Ok(job)
    }

    fn record_llm_attempt(
        &mut self,
        id: i64,
        attempt_count: i64,
        status: &str,
        raw_response: Option<&str>,
        parsed_response: Option<&str>,
        error: Option<&str>,
    ) -> Result<()> {
        let job = self
            .llm_jobs
            .get_mut(&id)
            .ok_or_else(|| Error::NotFound(format!("llm_job {id}")))?;
        job.attempt_count = attempt_count;
        job.status = status.to_string();
        job.raw_response = raw_response.map(ToString::to_string);
        job.parsed_response = parsed_response.map(ToString::to_string);
        job.error = error.map(ToString::to_string);
        Ok(())
    }

    fn get_llm_cache(&self, cache_hash: &str) -> Result<Option<LlmCacheEntry>> {
        Ok(self.llm_cache.get(cache_hash).cloned())
    }

    fn put_llm_cache(&mut self, entry: &LlmCacheEntry) -> Result<()> {
        self.llm_cache
            .insert(entry.cache_hash.clone(), entry.clone());
        Ok(())
    }

    fn save_mcq_items(&mut self, items: &[NewMcqItem]) -> Result<Vec<McqItem>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if item.attempt_no < 1 {
                return Err(Error::InvalidInput(
                    "mcq attempt_no must be >= 1".to_string(),
                ));
            }
            let id = self.alloc_id()?;
            let row = McqItem {
                id,
                chapter_id: item.chapter_id,
                phase: item.phase.clone(),
                question_text: item.question_text.clone(),
                options_json: item.options_json.clone(),
                correct_index: item.correct_index,
                trap_index: item.trap_index,
                explanation_text: item.explanation_text.clone(),
                source_refs: item.source_refs.clone(),
                topic: item.topic.clone(),
                attempt_no: item.attempt_no,
            };
            self.mcq_items.insert(id, row.clone());
            out.push(row);
        }
        Ok(out)
    }

    fn list_mcq_items(
        &self,
        chapter_id: i64,
        phase: &str,
        attempt_no: i64,
    ) -> Result<Vec<McqItem>> {
        let mut out: Vec<McqItem> = self
            .mcq_items
            .values()
            .filter(|m| m.chapter_id == chapter_id && m.phase == phase && m.attempt_no == attempt_no)
            .cloned()
            .collect();
        out.sort_by_key(|m| m.id);
        Ok(out)
    }

    fn delete_mcq_items_for(
        &mut self,
        chapter_id: i64,
        phase: &str,
        attempt_no: i64,
    ) -> Result<usize> {
        let ids: Vec<i64> = self
            .mcq_items
            .values()
            .filter(|m| m.chapter_id == chapter_id && m.phase == phase && m.attempt_no == attempt_no)
            .map(|m| m.id)
            .collect();
        let count = ids.len();
        for id in &ids {
            self.mcq_items.remove(id);
        }
        self.mcq_responses.retain(|_, r| !ids.contains(&r.mcq_item_id));
        Ok(count)
    }

    fn record_mcq_response(
        &mut self,
        mcq_item_id: i64,
        selected_index: i64,
        is_correct: bool,
        selected_trap: bool,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<McqResponse> {
        if !self.mcq_items.contains_key(&mcq_item_id) {
            return Err(Error::NotFound(format!("mcq_item {mcq_item_id}")));
        }
        let id = self.alloc_id()?;
        let row = McqResponse {
            id,
            mcq_item_id,
            selected_index,
            is_correct,
            selected_trap,
            answered_at: answered_at.to_string(),
            attempt_no,
        };
        self.mcq_responses.insert(id, row.clone());
        Ok(row)
    }

    fn list_mcq_responses(&self, mcq_item_id: i64) -> Result<Vec<McqResponse>> {
        let mut out: Vec<McqResponse> = self
            .mcq_responses
            .values()
            .filter(|r| r.mcq_item_id == mcq_item_id)
            .cloned()
            .collect();
        out.sort_by_key(|r| r.id);
        Ok(out)
    }

    fn create_misconception(
        &mut self,
        chapter_id: i64,
        concept: &str,
        description: &str,
        evidence: &str,
        source_task: &str,
        created_at: &str,
    ) -> Result<Misconception> {
        if concept.trim().is_empty() || description.trim().is_empty() {
            return Err(Error::InvalidInput(
                "misconception needs a concept and description".to_string(),
            ));
        }
        let id = self.alloc_id()?;
        let row = Misconception {
            id,
            chapter_id,
            concept_description: concept.to_string(),
            description: description.to_string(),
            evidence: evidence.to_string(),
            source_task: source_task.to_string(),
            status: "ACTIVE".to_string(),
            confidence: INITIAL_CONFIDENCE,
            created_at: created_at.to_string(),
            updated_at: created_at.to_string(),
            resolved_at: None,
        };
        self.misconceptions.insert(id, row.clone());
        Ok(row)
    }

    fn list_misconceptions(&self, chapter_id: i64) -> Result<Vec<Misconception>> {
        let mut out: Vec<Misconception> = self
            .misconceptions
            .values()
            .filter(|m| m.chapter_id == chapter_id)
            .cloned()
            .collect();
        out.sort_by_key(|m| m.id);
        Ok(out)
    }

    fn update_misconception(
        &mut self,
        id: i64,
        confidence: f64,
        status: &str,
        updated_at: &str,
        resolved_at: Option<&str>,
    ) -> Result<Misconception> {
        check_misconception_update(confidence, status)?;
        let Some(row) = self.misconceptions.get_mut(&id) else {
            return Err(Error::NotFound(format!("misconception {id}")));
        };
        row.confidence = confidence;
        row.status = status.to_string();
        row.updated_at = updated_at.to_string();
        row.resolved_at = resolved_at.map(str::to_string);
        Ok(row.clone())
    }

    fn save_assignment_questions(
        &mut self,
        items: &[NewAssignmentQuestion],
    ) -> Result<Vec<AssignmentQuestion>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if item.attempt_no < 1 {
                return Err(Error::InvalidInput(
                    "assignment attempt_no must be >= 1".to_string(),
                ));
            }
            if item.kind != "written" && item.kind != "coding" {
                return Err(Error::InvalidInput(format!(
                    "assignment kind must be written|coding, got '{}'",
                    item.kind
                )));
            }
            let id = self.alloc_id()?;
            let row = AssignmentQuestion {
                id,
                chapter_id: item.chapter_id,
                position: item.position,
                kind: item.kind.clone(),
                parts_json: item.parts_json.clone(),
                rubric_json: item.rubric_json.clone(),
                target_misconception_ids: item.target_misconception_ids.clone(),
                attempt_no: item.attempt_no,
            };
            self.assignment_questions.insert(id, row.clone());
            out.push(row);
        }
        Ok(out)
    }

    fn list_assignment_questions(
        &self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<Vec<AssignmentQuestion>> {
        let mut out: Vec<AssignmentQuestion> = self
            .assignment_questions
            .values()
            .filter(|q| q.chapter_id == chapter_id && q.attempt_no == attempt_no)
            .cloned()
            .collect();
        out.sort_by(|a, b| a.position.cmp(&b.position).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    fn delete_assignment_questions_for(
        &mut self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<usize> {
        let ids: Vec<i64> = self
            .assignment_questions
            .values()
            .filter(|q| q.chapter_id == chapter_id && q.attempt_no == attempt_no)
            .map(|q| q.id)
            .collect();
        let count = ids.len();
        for id in &ids {
            self.assignment_questions.remove(id);
        }
        self.assignment_responses
            .retain(|_, r| !ids.contains(&r.question_id));
        Ok(count)
    }

    fn record_assignment_response(
        &mut self,
        question_id: i64,
        answer_text: &str,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<AssignmentResponse> {
        if !self.assignment_questions.contains_key(&question_id) {
            return Err(Error::NotFound(format!(
                "assignment_question {question_id}"
            )));
        }
        let id = self.alloc_id()?;
        let row = AssignmentResponse {
            id,
            question_id,
            answer_text: answer_text.to_string(),
            answered_at: answered_at.to_string(),
            attempt_no,
        };
        self.assignment_responses.insert(id, row.clone());
        Ok(row)
    }

    fn list_assignment_responses(&self, question_id: i64) -> Result<Vec<AssignmentResponse>> {
        let mut out: Vec<AssignmentResponse> = self
            .assignment_responses
            .values()
            .filter(|r| r.question_id == question_id)
            .cloned()
            .collect();
        out.sort_by_key(|r| r.id);
        Ok(out)
    }

    fn save_grade(&mut self, input: &NewGrade) -> Result<Grade> {
        check_grade_input(input)?;
        let id = self.alloc_id()?;
        let row = Grade {
            id,
            question_id: input.question_id,
            score: input.score,
            max_score: input.max_score,
            classification: input.classification.clone(),
            criteria_results_json: input.criteria_results_json.clone(),
            feedback: input.feedback.clone(),
            disputed: false,
            original_score: None,
            dispute_text: None,
            final_score: None,
            adjudication_json: None,
            grader_version: input.grader_version.clone(),
            created_at: input.created_at.clone(),
        };
        self.grades.insert(id, row.clone());
        Ok(row)
    }

    fn get_grade(&self, id: i64) -> Result<Grade> {
        self.grades
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("grade {id}")))
    }

    fn list_grades_for_question(&self, question_id: i64) -> Result<Vec<Grade>> {
        let mut out: Vec<Grade> = self
            .grades
            .values()
            .filter(|g| g.question_id == question_id)
            .cloned()
            .collect();
        out.sort_by_key(|g| g.id);
        Ok(out)
    }

    fn record_dispute(&mut self, input: &NewDispute) -> Result<Dispute> {
        check_dispute_input(input)?;
        let grade = self
            .grades
            .get(&input.grade_id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("grade {}", input.grade_id)))?;
        if input.final_score < 0 || input.final_score > grade.max_score {
            return Err(Error::InvalidInput(format!(
                "dispute final_score {} outside 0-{}",
                input.final_score, grade.max_score
            )));
        }
        let preserved = grade.original_score.unwrap_or(grade.score);
        let updated = Grade {
            disputed: true,
            original_score: Some(preserved),
            dispute_text: Some(input.text.clone()),
            final_score: Some(input.final_score),
            adjudication_json: Some(input.adjudication_json.clone()),
            ..grade
        };
        self.grades.insert(updated.id, updated);
        let id = self.alloc_id()?;
        let row = Dispute {
            id,
            grade_id: input.grade_id,
            text: input.text.clone(),
            decision: input.decision.clone(),
            final_score: input.final_score,
            adjudicator_model: input.adjudicator_model.clone(),
            timestamp: input.timestamp.clone(),
        };
        self.disputes.insert(id, row.clone());
        Ok(row)
    }

    fn list_disputes_for_grade(&self, grade_id: i64) -> Result<Vec<Dispute>> {
        let mut out: Vec<Dispute> = self
            .disputes
            .values()
            .filter(|d| d.grade_id == grade_id)
            .cloned()
            .collect();
        out.sort_by_key(|d| d.id);
        Ok(out)
    }

    fn save_note(&mut self, input: &NewNote) -> Result<Note> {
        check_note_input(input)?;
        let id = self.alloc_id()?;
        let row = Note {
            id,
            chapter_id: input.chapter_id,
            content_markdown: input.content_markdown.clone(),
            generated_at: input.generated_at.clone(),
            attempt_no: input.attempt_no,
        };
        self.notes.insert(id, row.clone());
        Ok(row)
    }

    fn list_notes(&self, chapter_id: i64, attempt_no: i64) -> Result<Vec<Note>> {
        let mut out: Vec<Note> = self
            .notes
            .values()
            .filter(|n| n.chapter_id == chapter_id && n.attempt_no == attempt_no)
            .cloned()
            .collect();
        out.sort_by_key(|n| n.id);
        Ok(out)
    }

    fn clear_library(&mut self) -> Result<()> {
        self.books.clear();
        self.chapters.clear();
        self.tasks.clear();
        self.llm_jobs.clear();
        self.mcq_items.clear();
        self.mcq_responses.clear();
        self.misconceptions.clear();
        self.assignment_questions.clear();
        self.assignment_responses.clear();
        self.grades.clear();
        self.disputes.clear();
        self.notes.clear();
        self.next_id = 1;
        Ok(())
    }
}

/// SQLite implementation. Holds the advisory lock file guard for its lifetime;
/// dropping it releases the lock (including on crash via OS cleanup).
#[derive(Debug)]
pub struct SqliteStore {
    conn: Connection,
    _lock: File,
    _path: PathBuf,
}

impl SqliteStore {
    /// Open (creating parent directories as needed) at `db_path`, acquiring
    /// `lock_path` exclusively first. Uses a transaction for schema setup and
    /// version assertion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyOpen`] when another process holds the lock,
    /// [`Error::Store`] on SQLite failures.
    pub fn open(db_path: &Path, lock_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        if let Some(parent) = lock_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(lock_path)?;
        lock.try_lock_exclusive().map_err(|_| {
            Error::AlreadyOpen(format!("database locked: {}", lock_path.display()))
        })?;
        let conn = Connection::open(db_path)?;
        conn.execute_batch(SCHEMA_SQL)?;
        Self::ensure_version(&conn)?;
        Ok(Self {
            conn,
            _lock: lock,
            _path: db_path.to_path_buf(),
        })
    }

    /// Open an in-memory SQLite database (no lock file; for tests).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on SQLite failures and [`Error::Io`] when the
    /// scratch lock file cannot be created.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let lock_path = std::env::temp_dir().join(format!(
            "cadence-test-{}.lock",
            std::process::id()
        ));
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA_SQL)?;
        Self::ensure_version(&conn)?;
        Ok(Self {
            conn,
            _lock: lock,
            _path: PathBuf::from(":memory:"),
        })
    }

    /// Insert the version row on fresh DBs; migrate older schemas forward;
    /// reject newer-than-known schemas.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] when the stored version is newer than
    /// [`SCHEMA_VERSION`] or a migration step fails.
    fn ensure_version(conn: &Connection) -> Result<()> {
        let existing: Option<i64> = conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .optional()?;
        match existing {
            None => {
                conn.execute(
                    "INSERT INTO version (id, schema_version) VALUES (1, ?1)",
                    params![SCHEMA_VERSION],
                )?;
                Ok(())
            }
            Some(v) if v == SCHEMA_VERSION => Ok(()),
            Some(v) if v < SCHEMA_VERSION => {
                Self::migrate(conn, v)?;
                Ok(())
            }
            Some(v) => Err(Error::Store(format!(
                "database schema v{v} is newer than supported v{SCHEMA_VERSION}"
            ))),
        }
    }

    /// Check whether `column` exists on `table` (idempotent migrations).
    fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
        for row in rows {
            if row? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Apply forward migrations (`from` → [`SCHEMA_VERSION`]). Each step is
    /// idempotent (column-existence checks + `IF NOT EXISTS`) so a crashed
    /// migration can rerun safely.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] on SQLite failures.
    #[allow(clippy::too_many_lines)]
    fn migrate(conn: &Connection, from: i64) -> Result<()> {
        if from <= 1 {
            // v1 → v2: attempt tracking + MCQ + misconception tables (§4.1, §7.1, §12).
            if !Self::column_exists(conn, "chapters", "attempt_no")? {
                conn.execute_batch(
                    "ALTER TABLE chapters ADD COLUMN attempt_no INTEGER NOT NULL DEFAULT 1;",
                )?;
            }
            if !Self::column_exists(conn, "tasks", "attempt_no")? {
                conn.execute_batch(
                    "ALTER TABLE tasks ADD COLUMN attempt_no INTEGER NOT NULL DEFAULT 1;",
                )?;
            }
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS mcq_items (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   chapter_id INTEGER NOT NULL REFERENCES chapters(id),
                   phase TEXT NOT NULL,
                   question_text TEXT NOT NULL,
                   options_json TEXT NOT NULL,
                   correct_index INTEGER NOT NULL,
                   trap_index INTEGER NOT NULL,
                   explanation_text TEXT NOT NULL,
                   source_refs TEXT NOT NULL,
                   topic TEXT NOT NULL,
                   attempt_no INTEGER NOT NULL DEFAULT 1
                 );
                 CREATE TABLE IF NOT EXISTS mcq_responses (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id),
                   selected_index INTEGER NOT NULL,
                   is_correct INTEGER NOT NULL,
                   selected_trap INTEGER NOT NULL,
                   answered_at TEXT NOT NULL,
                   attempt_no INTEGER NOT NULL DEFAULT 1
                 );
                 CREATE TABLE IF NOT EXISTS misconceptions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   chapter_id INTEGER NOT NULL REFERENCES chapters(id),
                   concept_description TEXT NOT NULL,
                   description TEXT NOT NULL,
                   evidence TEXT NOT NULL,
                   source_task TEXT NOT NULL,
                   status TEXT NOT NULL,
                   confidence REAL NOT NULL,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL,
                   resolved_at TEXT
                 );
                  UPDATE version SET schema_version = 2 WHERE id = 1;",
            )?;
        }
        if from <= 2 {
            // v2 → v3: written-assignment tables (§7.2). Rubrics are created
            // with the questions, so they share one row per question.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS assignment_questions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   chapter_id INTEGER NOT NULL REFERENCES chapters(id),
                   position INTEGER NOT NULL,
                   kind TEXT NOT NULL,
                   parts_json TEXT NOT NULL,
                   rubric_json TEXT NOT NULL,
                   target_misconception_ids TEXT NOT NULL,
                   attempt_no INTEGER NOT NULL DEFAULT 1
                 );
                 CREATE TABLE IF NOT EXISTS assignment_responses (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   question_id INTEGER NOT NULL REFERENCES assignment_questions(id),
                   answer_text TEXT NOT NULL,
                   answered_at TEXT NOT NULL,
                   attempt_no INTEGER NOT NULL DEFAULT 1
                 );
                 UPDATE version SET schema_version = 3 WHERE id = 1;",
            )?;
        }
        if from <= 3 {
            // v3 → v4: dispute audit trail (§9). The `grades` row keeps the
            // original award (`score`) forever; a dispute writes `final_score`
            // plus the auditor payload, and appends a `disputes` row.
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS grades (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   question_id INTEGER NOT NULL REFERENCES assignment_questions(id),
                   score INTEGER NOT NULL,
                   max_score INTEGER NOT NULL,
                   classification TEXT NOT NULL,
                   criteria_results_json TEXT NOT NULL,
                   feedback TEXT NOT NULL,
                   disputed INTEGER NOT NULL DEFAULT 0,
                   original_score INTEGER,
                   dispute_text TEXT,
                   final_score INTEGER,
                   adjudication_json TEXT,
                   grader_version TEXT NOT NULL,
                   created_at TEXT NOT NULL
                 );
                  CREATE TABLE IF NOT EXISTS disputes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   grade_id INTEGER NOT NULL REFERENCES grades(id),
                   dispute_text TEXT NOT NULL,
                   decision TEXT NOT NULL,
                   final_score INTEGER NOT NULL,
                   adjudicator_model TEXT NOT NULL,
                   timestamp TEXT NOT NULL
                 );
                 UPDATE version SET schema_version = 4 WHERE id = 1;",
            )?;
        }
        if from <= 4 {
            // v4 → v5: post-grading chapter notes (§11). One synthesis
            // document per chapter/attempt; `attempt_no` keeps unskipped
            // re-attempts isolated like MCQs and assignments (§4.1).
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS notes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   chapter_id INTEGER NOT NULL REFERENCES chapters(id),
                   content_markdown TEXT NOT NULL,
                   generated_at TEXT NOT NULL,
                   attempt_no INTEGER NOT NULL DEFAULT 1
                 );
                 UPDATE version SET schema_version = 5 WHERE id = 1;",
            )?;
        }
        if from <= 5 {
            // v5 → v6: repair drifted tables. Databases created by
            // intermediate builds carry tables missing columns that later
            // builds added to fresh-table DDL without a repair path
            // (`CREATE TABLE IF NOT EXISTS` skips them while the version
            // stamp still advances), so v6 queries fail on live data
            // (`no such column`). Every repair is additive with a lossless
            // default — no row is dropped or rewritten. Pre-`kind`
            // assignment rows predate the coding question, so `'written'`
            // is exact, not a guess; pre-`attempt_no` rows are first
            // attempts by construction.
            for (table, column, ddl) in [
                ("misconceptions", "concept_description", "TEXT NOT NULL DEFAULT ''"),
                ("misconceptions", "updated_at", "TEXT NOT NULL DEFAULT ''"),
                ("mcq_items", "attempt_no", "INTEGER NOT NULL DEFAULT 1"),
                ("mcq_responses", "attempt_no", "INTEGER NOT NULL DEFAULT 1"),
                (
                    "assignment_questions",
                    "kind",
                    "TEXT NOT NULL DEFAULT 'written'",
                ),
                (
                    "assignment_questions",
                    "attempt_no",
                    "INTEGER NOT NULL DEFAULT 1",
                ),
                (
                    "assignment_responses",
                    "attempt_no",
                    "INTEGER NOT NULL DEFAULT 1",
                ),
                ("notes", "attempt_no", "INTEGER NOT NULL DEFAULT 1"),
            ] {
                if !Self::column_exists(conn, table, column)? {
                    conn.execute_batch(&format!(
                        "ALTER TABLE {table} ADD COLUMN {column} {ddl};"
                    ))?;
                }
            }
            // `updated_at` did not exist when old rows were written: inherit
            // the creation stamp (never NULL, never blank).
            conn.execute_batch(
                "UPDATE misconceptions SET updated_at = created_at WHERE updated_at = '';
                 UPDATE version SET schema_version = 6 WHERE id = 1;",
            )?;
        }
        let current: Option<i64> = conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .optional()?;
        if current != Some(SCHEMA_VERSION) {
            return Err(Error::Store(format!(
                "migration from v{from} did not reach v{SCHEMA_VERSION}"
            )));
        }
        Ok(())
    }
}

impl Store for SqliteStore {
    fn create_book(&mut self, input: &NewBook, created_at: &str) -> Result<Book> {
        self.conn.execute(
            "INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                input.title,
                input.filepath,
                input.file_hash,
                input.start_page,
                created_at
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Book {
            id,
            title: input.title.clone(),
            filepath: input.filepath.clone(),
            file_hash: input.file_hash.clone(),
            start_page: input.start_page,
            created_at: created_at.to_string(),
        })
    }

    fn get_book(&self, id: i64) -> Result<Book> {
        self.conn
            .query_row(
                "SELECT id, title, filepath, file_hash, start_page, created_at FROM books WHERE id = ?1",
                params![id],
                |r| {
                    Ok(Book {
                        id: r.get(0)?,
                        title: r.get(1)?,
                        filepath: r.get(2)?,
                        file_hash: r.get(3)?,
                        start_page: r.get(4)?,
                        created_at: r.get(5)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("book {id}")))
    }

    fn list_books(&self) -> Result<Vec<Book>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, filepath, file_hash, start_page, created_at FROM books ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Book {
                id: r.get(0)?,
                title: r.get(1)?,
                filepath: r.get(2)?,
                file_hash: r.get(3)?,
                start_page: r.get(4)?,
                created_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn create_chapter(&mut self, input: &NewChapter) -> Result<Chapter> {
        self.conn.execute(
            "INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
            params![
                input.book_id,
                input.index_in_book,
                input.level,
                input.title,
                input.start_page,
                input.end_page,
                input.file_path,
                input.status.as_str()
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Chapter {
            id,
            book_id: input.book_id,
            index_in_book: input.index_in_book,
            level: input.level,
            title: input.title.clone(),
            start_page: input.start_page,
            end_page: input.end_page,
            file_path: input.file_path.clone(),
            status: input.status,
            attempt_no: 1,
        })
    }

    fn get_chapter(&self, id: i64) -> Result<Chapter> {
        let row: Option<(i64, i64, i64, i64, String, i64, i64, String, String, i64)> = self
            .conn
            .query_row(
                "SELECT id, book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no FROM chapters WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                        r.get(8)?,
                        r.get(9)?,
                    ))
                },
            )
            .optional()?;
        row.map_or_else(
            || Err(Error::NotFound(format!("chapter {id}"))),
            |(
                id,
                book_id,
                index_in_book,
                level,
                title,
                start_page,
                end_page,
                file_path,
                status,
                attempt_no,
            )| {
                Ok(Chapter {
                    id,
                    book_id,
                    index_in_book,
                    level,
                    title,
                    start_page,
                    end_page,
                    file_path,
                    status: ChapterStatus::parse(&status)?,
                    attempt_no,
                })
            },
        )
    }

    fn list_chapters(&self, book_id: i64) -> Result<Vec<Chapter>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no FROM chapters WHERE book_id = ?1 ORDER BY index_in_book ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![book_id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
                r.get::<_, i64>(9)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                id,
                book_id,
                index_in_book,
                level,
                title,
                start_page,
                end_page,
                file_path,
                status,
                attempt_no,
            ) = row?;
            out.push(Chapter {
                id,
                book_id,
                index_in_book,
                level,
                title,
                start_page,
                end_page,
                file_path,
                status: ChapterStatus::parse(&status)?,
                attempt_no,
            });
        }
        Ok(out)
    }

    fn set_chapter_status(&mut self, id: i64, status: ChapterStatus) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE chapters SET status = ?1 WHERE id = ?2",
            params![status.as_str(), id],
        )?;
        if changed == 0 {
            return Err(Error::NotFound(format!("chapter {id}")));
        }
        Ok(())
    }

    fn set_chapter_attempt(&mut self, id: i64, attempt_no: i64) -> Result<()> {
        if attempt_no < 1 {
            return Err(Error::InvalidInput(
                "attempt_no must be >= 1".to_string(),
            ));
        }
        let changed = self.conn.execute(
            "UPDATE chapters SET attempt_no = ?1 WHERE id = ?2",
            params![attempt_no, id],
        )?;
        if changed == 0 {
            return Err(Error::NotFound(format!("chapter {id}")));
        }
        Ok(())
    }

    fn delete_pending_tasks_for_chapter(&mut self, chapter_id: i64) -> Result<usize> {
        let changed = self.conn.execute(
            "DELETE FROM tasks WHERE chapter_id = ?1 AND status = ?2",
            params![chapter_id, TaskStatus::Pending.as_str()],
        )?;
        Ok(changed)
    }

    fn create_task(&mut self, input: &NewTask) -> Result<Task> {
        if input.attempt_no < 1 {
            return Err(Error::InvalidInput(
                "task attempt_no must be >= 1".to_string(),
            ));
        }
        let scheduled = input.scheduled_for.format("%Y-%m-%d").to_string();
        self.conn.execute(
            "INSERT INTO tasks (book_id, chapter_id, type, scheduled_for, status, completed_at, sequence, attempt_no) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7)",
            params![
                input.book_id,
                input.chapter_id,
                input.task_type.as_str(),
                scheduled,
                TaskStatus::Pending.as_str(),
                input.sequence,
                input.attempt_no
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Task {
            id,
            book_id: input.book_id,
            chapter_id: input.chapter_id,
            task_type: input.task_type,
            scheduled_for: input.scheduled_for,
            status: TaskStatus::Pending,
            completed_at: None,
            sequence: input.sequence,
            attempt_no: input.attempt_no,
        })
    }

    fn get_task(&self, id: i64) -> Result<Task> {
        let row: Option<(
            i64,
            i64,
            i64,
            String,
            String,
            String,
            Option<String>,
            i64,
            i64,
        )> = self
            .conn
            .query_row(
                "SELECT id, book_id, chapter_id, type, scheduled_for, status, completed_at, sequence, attempt_no FROM tasks WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                        r.get(8)?,
                    ))
                },
            )
            .optional()?;
        row.map_or_else(
            || Err(Error::NotFound(format!("task {id}"))),
            |(
                id,
                book_id,
                chapter_id,
                kind,
                scheduled_for,
                status,
                completed_at,
                sequence,
                attempt_no,
            )| {
                let parsed_date = NaiveDate::parse_from_str(&scheduled_for, "%Y-%m-%d")
                    .map_err(|e| Error::Store(e.to_string()))?;
                Ok(Task {
                    id,
                    book_id,
                    chapter_id,
                    task_type: TaskType::parse(&kind)?,
                    scheduled_for: parsed_date,
                    status: TaskStatus::parse(&status)?,
                    completed_at,
                    sequence,
                    attempt_no,
                })
            },
        )
    }

    fn list_tasks(&self) -> Result<Vec<Task>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, book_id, chapter_id, type, scheduled_for, status, completed_at, sequence, attempt_no FROM tasks ORDER BY sequence ASC, id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, i64>(7)?,
                r.get::<_, i64>(8)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                id,
                book_id,
                chapter_id,
                kind,
                scheduled_for,
                status,
                completed_at,
                sequence,
                attempt_no,
            ) = row?;
            let parsed_date = NaiveDate::parse_from_str(&scheduled_for, "%Y-%m-%d")
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                ))?;
            out.push(Task {
                id,
                book_id,
                chapter_id,
                task_type: TaskType::parse(&kind)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())),
                    ))?,
                scheduled_for: parsed_date,
                status: TaskStatus::parse(&status).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())),
                    )
                })?,
                completed_at,
                sequence,
                attempt_no,
            });
        }
        Ok(out)
    }

    fn complete_task(&mut self, id: i64, completed_at: &str) -> Result<Task> {
        let task = self.get_task(id)?;
        if task.status == TaskStatus::Done {
            return Err(Error::AlreadyCompleted(format!("task {id}")));
        }
        self.conn.execute(
            "UPDATE tasks SET status = ?1, completed_at = ?2 WHERE id = ?3",
            params![TaskStatus::Done.as_str(), completed_at, id],
        )?;
        let mut done = task;
        done.status = TaskStatus::Done;
        done.completed_at = Some(completed_at.to_string());
        Ok(done)
    }

    fn reschedule_task(&mut self, id: i64, scheduled_for: NaiveDate) -> Result<Task> {
        let task = self.get_task(id)?;
        if task.status == TaskStatus::Done {
            return Err(Error::AlreadyCompleted(format!("task {id}")));
        }
        let stamped = scheduled_for.format("%Y-%m-%d").to_string();
        self.conn.execute(
            "UPDATE tasks SET scheduled_for = ?1 WHERE id = ?2",
            params![stamped, id],
        )?;
        let mut moved_task = task;
        moved_task.scheduled_for = scheduled_for;
        Ok(moved_task)
    }

    fn log_event(
        &mut self,
        event_type: &str,
        chapter_id: Option<i64>,
        task_id: Option<i64>,
        evidence: Option<&str>,
        created_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO event_log (event_type, chapter_id, task_id, evidence, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![event_type, chapter_id, task_id, evidence, created_at],
        )?;
        Ok(())
    }

    fn create_llm_job(&mut self, input: &NewLlmJob) -> Result<LlmJob> {
        self.conn.execute(
            "INSERT INTO llm_jobs (operation, provider, model, input_hash, prompt_version, status, attempt_count, raw_response, parsed_response, error) VALUES (?1, ?2, ?3, ?4, ?5, 'PENDING', 0, NULL, NULL, NULL)",
            params![
                input.operation,
                input.provider,
                input.model,
                input.input_hash,
                input.prompt_version
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(LlmJob {
            id,
            operation: input.operation.clone(),
            provider: input.provider.clone(),
            model: input.model.clone(),
            input_hash: input.input_hash.clone(),
            prompt_version: input.prompt_version.clone(),
            status: "PENDING".to_string(),
            attempt_count: 0,
            raw_response: None,
            parsed_response: None,
            error: None,
        })
    }

    fn record_llm_attempt(
        &mut self,
        id: i64,
        attempt_count: i64,
        status: &str,
        raw_response: Option<&str>,
        parsed_response: Option<&str>,
        error: Option<&str>,
    ) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE llm_jobs SET status = ?1, attempt_count = ?2, raw_response = ?3, parsed_response = ?4, error = ?5 WHERE id = ?6",
            params![status, attempt_count, raw_response, parsed_response, error, id],
        )?;
        if changed == 0 {
            return Err(Error::NotFound(format!("llm_job {id}")));
        }
        Ok(())
    }

    fn get_llm_cache(&self, cache_hash: &str) -> Result<Option<LlmCacheEntry>> {
        let row: Option<(String, String, String, String, String, String, String, String, String)> =
            self.conn
                .query_row(
                    "SELECT cache_hash, operation, provider, model, prompt_version, request_json, response_json, status, created_at FROM llm_cache WHERE cache_hash = ?1",
                    params![cache_hash],
                    |r| {
                        Ok((
                            r.get(0)?,
                            r.get(1)?,
                            r.get(2)?,
                            r.get(3)?,
                            r.get(4)?,
                            r.get(5)?,
                            r.get(6)?,
                            r.get(7)?,
                            r.get(8)?,
                        ))
                    },
                )
                .optional()?;
        Ok(row.map(
            |(
                cache_hash,
                operation,
                provider,
                model,
                prompt_version,
                request_json,
                response_json,
                status,
                created_at,
            )| {
                LlmCacheEntry {
                    cache_hash,
                    operation,
                    provider,
                    model,
                    prompt_version,
                    request_json,
                    response_json,
                    status,
                    created_at,
                }
            },
        ))
    }

    fn put_llm_cache(&mut self, entry: &LlmCacheEntry) -> Result<()> {
        self.conn.execute(
            "INSERT INTO llm_cache (cache_hash, operation, provider, model, prompt_version, request_json, response_json, status, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) ON CONFLICT(cache_hash) DO UPDATE SET response_json = excluded.response_json, status = excluded.status, created_at = excluded.created_at",
            params![
                entry.cache_hash,
                entry.operation,
                entry.provider,
                entry.model,
                entry.prompt_version,
                entry.request_json,
                entry.response_json,
                entry.status,
                entry.created_at
            ],
        )?;
        Ok(())
    }

    fn save_mcq_items(&mut self, items: &[NewMcqItem]) -> Result<Vec<McqItem>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if item.attempt_no < 1 {
                return Err(Error::InvalidInput(
                    "mcq attempt_no must be >= 1".to_string(),
                ));
            }
            self.conn.execute(
                "INSERT INTO mcq_items (chapter_id, phase, question_text, options_json, correct_index, trap_index, explanation_text, source_refs, topic, attempt_no) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    item.chapter_id,
                    item.phase,
                    item.question_text,
                    item.options_json,
                    item.correct_index,
                    item.trap_index,
                    item.explanation_text,
                    item.source_refs,
                    item.topic,
                    item.attempt_no
                ],
            )?;
            let id = self.conn.last_insert_rowid();
            out.push(McqItem {
                id,
                chapter_id: item.chapter_id,
                phase: item.phase.clone(),
                question_text: item.question_text.clone(),
                options_json: item.options_json.clone(),
                correct_index: item.correct_index,
                trap_index: item.trap_index,
                explanation_text: item.explanation_text.clone(),
                source_refs: item.source_refs.clone(),
                topic: item.topic.clone(),
                attempt_no: item.attempt_no,
            });
        }
        Ok(out)
    }

    fn list_mcq_items(
        &self,
        chapter_id: i64,
        phase: &str,
        attempt_no: i64,
    ) -> Result<Vec<McqItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chapter_id, phase, question_text, options_json, correct_index, trap_index, explanation_text, source_refs, topic, attempt_no FROM mcq_items WHERE chapter_id = ?1 AND phase = ?2 AND attempt_no = ?3 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chapter_id, phase, attempt_no], |r| {
            Ok(McqItem {
                id: r.get(0)?,
                chapter_id: r.get(1)?,
                phase: r.get(2)?,
                question_text: r.get(3)?,
                options_json: r.get(4)?,
                correct_index: r.get(5)?,
                trap_index: r.get(6)?,
                explanation_text: r.get(7)?,
                source_refs: r.get(8)?,
                topic: r.get(9)?,
                attempt_no: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn delete_mcq_items_for(
        &mut self,
        chapter_id: i64,
        phase: &str,
        attempt_no: i64,
    ) -> Result<usize> {
        self.conn.execute(
            "DELETE FROM mcq_responses WHERE mcq_item_id IN (SELECT id FROM mcq_items WHERE chapter_id = ?1 AND phase = ?2 AND attempt_no = ?3)",
            params![chapter_id, phase, attempt_no],
        )?;
        let removed = self.conn.execute(
            "DELETE FROM mcq_items WHERE chapter_id = ?1 AND phase = ?2 AND attempt_no = ?3",
            params![chapter_id, phase, attempt_no],
        )?;
        Ok(removed)
    }

    fn record_mcq_response(
        &mut self,
        mcq_item_id: i64,
        selected_index: i64,
        is_correct: bool,
        selected_trap: bool,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<McqResponse> {
        self.conn.execute(
            "INSERT INTO mcq_responses (mcq_item_id, selected_index, is_correct, selected_trap, answered_at, attempt_no) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                mcq_item_id,
                selected_index,
                i64::from(is_correct),
                i64::from(selected_trap),
                answered_at,
                attempt_no
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(McqResponse {
            id,
            mcq_item_id,
            selected_index,
            is_correct,
            selected_trap,
            answered_at: answered_at.to_string(),
            attempt_no,
        })
    }

    fn list_mcq_responses(&self, mcq_item_id: i64) -> Result<Vec<McqResponse>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, mcq_item_id, selected_index, is_correct, selected_trap, answered_at, attempt_no FROM mcq_responses WHERE mcq_item_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![mcq_item_id], |r| {
            let is_correct: i64 = r.get(3)?;
            let selected_trap: i64 = r.get(4)?;
            Ok(McqResponse {
                id: r.get(0)?,
                mcq_item_id: r.get(1)?,
                selected_index: r.get(2)?,
                is_correct: is_correct != 0,
                selected_trap: selected_trap != 0,
                answered_at: r.get(5)?,
                attempt_no: r.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn create_misconception(
        &mut self,
        chapter_id: i64,
        concept: &str,
        description: &str,
        evidence: &str,
        source_task: &str,
        created_at: &str,
    ) -> Result<Misconception> {
        if concept.trim().is_empty() || description.trim().is_empty() {
            return Err(Error::InvalidInput(
                "misconception needs a concept and description".to_string(),
            ));
        }
        self.conn.execute(
            "INSERT INTO misconceptions (chapter_id, concept_description, description, evidence, source_task, status, confidence, created_at, updated_at, resolved_at) VALUES (?1, ?2, ?3, ?4, ?5, 'ACTIVE', ?6, ?7, ?7, NULL)",
            params![chapter_id, concept, description, evidence, source_task, INITIAL_CONFIDENCE, created_at],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Misconception {
            id,
            chapter_id,
            concept_description: concept.to_string(),
            description: description.to_string(),
            evidence: evidence.to_string(),
            source_task: source_task.to_string(),
            status: "ACTIVE".to_string(),
            confidence: INITIAL_CONFIDENCE,
            created_at: created_at.to_string(),
            updated_at: created_at.to_string(),
            resolved_at: None,
        })
    }

    fn list_misconceptions(&self, chapter_id: i64) -> Result<Vec<Misconception>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chapter_id, concept_description, description, evidence, source_task, status, confidence, created_at, updated_at, resolved_at FROM misconceptions WHERE chapter_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chapter_id], |r| {
            Ok(Misconception {
                id: r.get(0)?,
                chapter_id: r.get(1)?,
                concept_description: r.get(2)?,
                description: r.get(3)?,
                evidence: r.get(4)?,
                source_task: r.get(5)?,
                status: r.get(6)?,
                confidence: r.get(7)?,
                created_at: r.get(8)?,
                updated_at: r.get(9)?,
                resolved_at: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn update_misconception(
        &mut self,
        id: i64,
        confidence: f64,
        status: &str,
        updated_at: &str,
        resolved_at: Option<&str>,
    ) -> Result<Misconception> {
        check_misconception_update(confidence, status)?;
        let changed = self.conn.execute(
            "UPDATE misconceptions SET confidence = ?1, status = ?2, updated_at = ?3, resolved_at = ?4 WHERE id = ?5",
            params![confidence, status, updated_at, resolved_at, id],
        )?;
        if changed == 0 {
            return Err(Error::NotFound(format!("misconception {id}")));
        }
        self.conn
            .query_row(
                "SELECT id, chapter_id, concept_description, description, evidence, source_task, status, confidence, created_at, updated_at, resolved_at FROM misconceptions WHERE id = ?1",
                params![id],
                |r| {
                    Ok(Misconception {
                        id: r.get(0)?,
                        chapter_id: r.get(1)?,
                        concept_description: r.get(2)?,
                        description: r.get(3)?,
                        evidence: r.get(4)?,
                        source_task: r.get(5)?,
                        status: r.get(6)?,
                        confidence: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                        resolved_at: r.get(10)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("misconception {id}")))
    }

    fn save_assignment_questions(
        &mut self,
        items: &[NewAssignmentQuestion],
    ) -> Result<Vec<AssignmentQuestion>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if item.attempt_no < 1 {
                return Err(Error::InvalidInput(
                    "assignment attempt_no must be >= 1".to_string(),
                ));
            }
            if item.kind != "written" && item.kind != "coding" {
                return Err(Error::InvalidInput(format!(
                    "assignment kind must be written|coding, got '{}'",
                    item.kind
                )));
            }
            self.conn.execute(
                "INSERT INTO assignment_questions (chapter_id, position, kind, parts_json, rubric_json, target_misconception_ids, attempt_no) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    item.chapter_id,
                    item.position,
                    item.kind,
                    item.parts_json,
                    item.rubric_json,
                    item.target_misconception_ids,
                    item.attempt_no
                ],
            )?;
            let id = self.conn.last_insert_rowid();
            out.push(AssignmentQuestion {
                id,
                chapter_id: item.chapter_id,
                position: item.position,
                kind: item.kind.clone(),
                parts_json: item.parts_json.clone(),
                rubric_json: item.rubric_json.clone(),
                target_misconception_ids: item.target_misconception_ids.clone(),
                attempt_no: item.attempt_no,
            });
        }
        Ok(out)
    }

    fn list_assignment_questions(
        &self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<Vec<AssignmentQuestion>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chapter_id, position, kind, parts_json, rubric_json, target_misconception_ids, attempt_no FROM assignment_questions WHERE chapter_id = ?1 AND attempt_no = ?2 ORDER BY position ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![chapter_id, attempt_no], |r| {
            Ok(AssignmentQuestion {
                id: r.get(0)?,
                chapter_id: r.get(1)?,
                position: r.get(2)?,
                kind: r.get(3)?,
                parts_json: r.get(4)?,
                rubric_json: r.get(5)?,
                target_misconception_ids: r.get(6)?,
                attempt_no: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn delete_assignment_questions_for(
        &mut self,
        chapter_id: i64,
        attempt_no: i64,
    ) -> Result<usize> {
        self.conn.execute(
            "DELETE FROM assignment_responses WHERE question_id IN (SELECT id FROM assignment_questions WHERE chapter_id = ?1 AND attempt_no = ?2)",
            params![chapter_id, attempt_no],
        )?;
        let removed = self.conn.execute(
            "DELETE FROM assignment_questions WHERE chapter_id = ?1 AND attempt_no = ?2",
            params![chapter_id, attempt_no],
        )?;
        Ok(removed)
    }

    fn record_assignment_response(
        &mut self,
        question_id: i64,
        answer_text: &str,
        answered_at: &str,
        attempt_no: i64,
    ) -> Result<AssignmentResponse> {
        self.conn.execute(
            "INSERT INTO assignment_responses (question_id, answer_text, answered_at, attempt_no) VALUES (?1, ?2, ?3, ?4)",
            params![question_id, answer_text, answered_at, attempt_no],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(AssignmentResponse {
            id,
            question_id,
            answer_text: answer_text.to_string(),
            answered_at: answered_at.to_string(),
            attempt_no,
        })
    }

    fn list_assignment_responses(&self, question_id: i64) -> Result<Vec<AssignmentResponse>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, question_id, answer_text, answered_at, attempt_no FROM assignment_responses WHERE question_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![question_id], |r| {
            Ok(AssignmentResponse {
                id: r.get(0)?,
                question_id: r.get(1)?,
                answer_text: r.get(2)?,
                answered_at: r.get(3)?,
                attempt_no: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn save_grade(&mut self, input: &NewGrade) -> Result<Grade> {
        check_grade_input(input)?;
        self.conn.execute(
            "INSERT INTO grades (question_id, score, max_score, classification, criteria_results_json, feedback, disputed, original_score, dispute_text, final_score, adjudication_json, grader_version, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, NULL, NULL, NULL, NULL, ?7, ?8)",
            params![
                input.question_id,
                input.score,
                input.max_score,
                input.classification,
                input.criteria_results_json,
                input.feedback,
                input.grader_version,
                input.created_at
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Grade {
            id,
            question_id: input.question_id,
            score: input.score,
            max_score: input.max_score,
            classification: input.classification.clone(),
            criteria_results_json: input.criteria_results_json.clone(),
            feedback: input.feedback.clone(),
            disputed: false,
            original_score: None,
            dispute_text: None,
            final_score: None,
            adjudication_json: None,
            grader_version: input.grader_version.clone(),
            created_at: input.created_at.clone(),
        })
    }

    fn get_grade(&self, id: i64) -> Result<Grade> {
        self.conn
            .query_row(
                "SELECT id, question_id, score, max_score, classification, criteria_results_json, feedback, disputed, original_score, dispute_text, final_score, adjudication_json, grader_version, created_at FROM grades WHERE id = ?1",
                params![id],
                |r| {
                    let disputed: i64 = r.get(7)?;
                    Ok(Grade {
                        id: r.get(0)?,
                        question_id: r.get(1)?,
                        score: r.get(2)?,
                        max_score: r.get(3)?,
                        classification: r.get(4)?,
                        criteria_results_json: r.get(5)?,
                        feedback: r.get(6)?,
                        disputed: disputed != 0,
                        original_score: r.get(8)?,
                        dispute_text: r.get(9)?,
                        final_score: r.get(10)?,
                        adjudication_json: r.get(11)?,
                        grader_version: r.get(12)?,
                        created_at: r.get(13)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("grade {id}")))
    }

    fn list_grades_for_question(&self, question_id: i64) -> Result<Vec<Grade>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, question_id, score, max_score, classification, criteria_results_json, feedback, disputed, original_score, dispute_text, final_score, adjudication_json, grader_version, created_at FROM grades WHERE question_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![question_id], |r| {
            let disputed: i64 = r.get(7)?;
            Ok(Grade {
                id: r.get(0)?,
                question_id: r.get(1)?,
                score: r.get(2)?,
                max_score: r.get(3)?,
                classification: r.get(4)?,
                criteria_results_json: r.get(5)?,
                feedback: r.get(6)?,
                disputed: disputed != 0,
                original_score: r.get(8)?,
                dispute_text: r.get(9)?,
                final_score: r.get(10)?,
                adjudication_json: r.get(11)?,
                grader_version: r.get(12)?,
                created_at: r.get(13)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn record_dispute(&mut self, input: &NewDispute) -> Result<Dispute> {
        check_dispute_input(input)?;
        let grade = self.get_grade(input.grade_id)?;
        if input.final_score < 0 || input.final_score > grade.max_score {
            return Err(Error::InvalidInput(format!(
                "dispute final_score {} outside 0-{}",
                input.final_score, grade.max_score
            )));
        }
        // `COALESCE` preserves the first original on re-dispute: the `score`
        // column itself is never overwritten, so the audit trail stays intact.
        self.conn.execute(
            "UPDATE grades SET disputed = 1, original_score = COALESCE(original_score, score), dispute_text = ?1, final_score = ?2, adjudication_json = ?3 WHERE id = ?4",
            params![
                input.text,
                input.final_score,
                input.adjudication_json,
                input.grade_id
            ],
        )?;
        self.conn.execute(
            "INSERT INTO disputes (grade_id, dispute_text, decision, final_score, adjudicator_model, timestamp) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                input.grade_id,
                input.text,
                input.decision,
                input.final_score,
                input.adjudicator_model,
                input.timestamp
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Dispute {
            id,
            grade_id: input.grade_id,
            text: input.text.clone(),
            decision: input.decision.clone(),
            final_score: input.final_score,
            adjudicator_model: input.adjudicator_model.clone(),
            timestamp: input.timestamp.clone(),
        })
    }

    fn list_disputes_for_grade(&self, grade_id: i64) -> Result<Vec<Dispute>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, grade_id, dispute_text, decision, final_score, adjudicator_model, timestamp FROM disputes WHERE grade_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![grade_id], |r| {
            Ok(Dispute {
                id: r.get(0)?,
                grade_id: r.get(1)?,
                text: r.get(2)?,
                decision: r.get(3)?,
                final_score: r.get(4)?,
                adjudicator_model: r.get(5)?,
                timestamp: r.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn save_note(&mut self, input: &NewNote) -> Result<Note> {
        check_note_input(input)?;
        self.conn.execute(
            "INSERT INTO notes (chapter_id, content_markdown, generated_at, attempt_no) VALUES (?1, ?2, ?3, ?4)",
            params![
                input.chapter_id,
                input.content_markdown,
                input.generated_at,
                input.attempt_no
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Note {
            id,
            chapter_id: input.chapter_id,
            content_markdown: input.content_markdown.clone(),
            generated_at: input.generated_at.clone(),
            attempt_no: input.attempt_no,
        })
    }

    fn list_notes(&self, chapter_id: i64, attempt_no: i64) -> Result<Vec<Note>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, chapter_id, content_markdown, generated_at, attempt_no FROM notes WHERE chapter_id = ?1 AND attempt_no = ?2 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![chapter_id, attempt_no], |r| {
            Ok(Note {
                id: r.get(0)?,
                chapter_id: r.get(1)?,
                content_markdown: r.get(2)?,
                generated_at: r.get(3)?,
                attempt_no: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn clear_library(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM disputes;
             DELETE FROM grades;
             DELETE FROM assignment_responses;
             DELETE FROM assignment_questions;
             DELETE FROM mcq_responses;
             DELETE FROM mcq_items;
             DELETE FROM misconceptions;
             DELETE FROM notes;
             DELETE FROM llm_jobs;
             DELETE FROM tasks;
             DELETE FROM event_log;
             DELETE FROM chapters;
             DELETE FROM books;
             DELETE FROM sqlite_sequence WHERE name IN ('books', 'chapters', 'tasks', 'event_log', 'llm_jobs', 'mcq_items', 'mcq_responses', 'misconceptions', 'assignment_questions', 'assignment_responses', 'grades', 'disputes', 'notes');",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn sample_book() -> NewBook {
        NewBook {
            title: "Modern C".to_string(),
            filepath: "/books/modern-c.pdf".to_string(),
            file_hash: "abc123".to_string(),
            start_page: 25,
        }
    }

    fn seed_library(store: &mut impl Store) {
        let book = store.create_book(&sample_book(), "2026-01-01").unwrap();
        let chapter = store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 25,
                end_page: 60,
                file_path: "unit_1.json".to_string(),
                status: ChapterStatus::PretestReady,
            })
            .unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        store
            .create_task(&NewTask {
                book_id: book.id,
                chapter_id: chapter.id,
                task_type: TaskType::Pretest,
                scheduled_for: date,
                sequence: 1,
                attempt_no: 1,
            })
            .unwrap();
        store
            .save_note(&NewNote {
                chapter_id: chapter.id,
                content_markdown: "# Notes".to_string(),
                generated_at: "2026-01-05".to_string(),
                attempt_no: 1,
            })
            .unwrap();
        store
            .create_llm_job(&NewLlmJob {
                operation: "pretest".to_string(),
                provider: "custom".to_string(),
                model: "m".to_string(),
                input_hash: "h".to_string(),
                prompt_version: "v3".to_string(),
            })
            .unwrap();
        store
            .put_llm_cache(&LlmCacheEntry {
                cache_hash: "c".to_string(),
                operation: "pretest".to_string(),
                provider: "custom".to_string(),
                model: "m".to_string(),
                prompt_version: "v3".to_string(),
                request_json: "{}".to_string(),
                response_json: "{}".to_string(),
                status: "OK".to_string(),
                created_at: "2026-01-05".to_string(),
            })
            .unwrap();
    }

    #[test]
    fn clear_library_empties_memory_store_but_keeps_cache() {
        let mut store = MemoryStore::new();
        seed_library(&mut store);
        assert_eq!(store.list_books().unwrap().len(), 1);
        store.clear_library().unwrap();
        assert_eq!(store.list_books().unwrap().len(), 0);
        assert_eq!(store.list_tasks().unwrap().len(), 0);
        assert!(store.get_llm_cache("c").unwrap().is_some());
        let book = store.create_book(&sample_book(), "2026-01-02").unwrap();
        assert_eq!(book.id, 1);
    }

    #[test]
    fn clear_library_empties_sqlite_but_keeps_version_and_cache() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        seed_library(&mut store);
        assert_eq!(store.list_books().unwrap().len(), 1);
        store.clear_library().unwrap();
        assert_eq!(store.list_books().unwrap().len(), 0);
        assert_eq!(store.list_tasks().unwrap().len(), 0);
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        assert!(store.get_llm_cache("c").unwrap().is_some());
        let book = store.create_book(&sample_book(), "2026-01-02").unwrap();
        assert_eq!(book.id, 1);
    }

    #[test]
    fn memory_round_trip() {
        let mut store = MemoryStore::new();
        let book = store.create_book(&sample_book(), "2026-01-01").unwrap();
        assert_eq!(store.get_book(book.id).unwrap(), book);
        let chapter = store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 25,
                end_page: 60,
                file_path: "unit_1.json".to_string(),
                status: ChapterStatus::PretestReady,
            })
            .unwrap();
        assert_eq!(store.get_chapter(chapter.id).unwrap(), chapter);
        store
            .set_chapter_status(chapter.id, ChapterStatus::PretestComplete)
            .unwrap();
        assert_eq!(
            store.get_chapter(chapter.id).unwrap().status,
            ChapterStatus::PretestComplete
        );
        let date = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        let task = store
            .create_task(&NewTask {
                book_id: book.id,
                chapter_id: chapter.id,
                task_type: TaskType::Pretest,
                scheduled_for: date,
                sequence: 1,
                attempt_no: chapter.attempt_no,
            })
            .unwrap();
        assert_eq!(store.get_task(task.id).unwrap(), task);
        let done = store.complete_task(task.id, "2026-01-05").unwrap();
        assert_eq!(done.status, TaskStatus::Done);
        assert!(store.complete_task(task.id, "2026-01-06").is_err());
    }

    #[test]
    fn lists_books_in_order_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            assert_eq!(store.list_books().unwrap().len(), 0);
            let first = store.create_book(&sample_book(), "2026-01-01").unwrap();
            let second = store
                .create_book(
                    &NewBook {
                        title: "Second".to_string(),
                        ..sample_book()
                    },
                    "2026-01-02",
                )
                .unwrap();
            let listed = store.list_books().unwrap();
            assert_eq!(listed.len(), 2);
            assert_eq!(listed[0].id, first.id);
            assert_eq!(listed[1].id, second.id);
            assert_eq!(listed[1].title, "Second");
        }
    }

    #[test]
    fn sqlite_round_trip_and_locking() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let book = store.create_book(&sample_book(), "2026-01-01").unwrap();
        assert_eq!(store.get_book(book.id).unwrap().title, "Modern C");
        let missing = store.get_book(9999).unwrap_err();
        assert!(matches!(missing, Error::NotFound(_)));
    }

    #[test]
    fn sqlite_tasks_round_trip() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let book = store.create_book(&sample_book(), "2026-01-01").unwrap();
        let chapter = store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 25,
                end_page: 60,
                file_path: "unit_1.json".to_string(),
                status: ChapterStatus::PretestReady,
            })
            .unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        let task = store
            .create_task(&NewTask {
                book_id: book.id,
                chapter_id: chapter.id,
                task_type: TaskType::Retest,
                scheduled_for: date,
                sequence: 7,
                attempt_no: chapter.attempt_no,
            })
            .unwrap();
        let fetched = store.get_task(task.id).unwrap();
        assert_eq!(fetched.task_type, TaskType::Retest);
        assert_eq!(fetched.scheduled_for, date);
        let listed = store.list_tasks().unwrap();
        assert_eq!(listed.len(), 1);
    }

    #[test]
    fn llm_job_and_cache_round_trip_memory() {
        let mut store = MemoryStore::new();
        let job = store
            .create_llm_job(&NewLlmJob {
                operation: "smoke".to_string(),
                provider: "ai-gateway".to_string(),
                model: "m".to_string(),
                input_hash: "abc".to_string(),
                prompt_version: "v1".to_string(),
            })
            .unwrap();
        assert_eq!(job.status, "PENDING");
        store
            .record_llm_attempt(job.id, 2, "OK", Some("raw"), Some("ok"), None)
            .unwrap();
        assert!(store.get_llm_cache("nope").unwrap().is_none());
        store
            .put_llm_cache(&LlmCacheEntry {
                cache_hash: "h1".to_string(),
                operation: "smoke".to_string(),
                provider: "ai-gateway".to_string(),
                model: "m".to_string(),
                prompt_version: "v1".to_string(),
                request_json: "{}".to_string(),
                response_json: "ok".to_string(),
                status: "OK".to_string(),
                created_at: "2026-01-01".to_string(),
            })
            .unwrap();
        let hit = store.get_llm_cache("h1").unwrap().unwrap();
        assert_eq!(hit.response_json, "ok");
        assert!(store.record_llm_attempt(9999, 1, "FAILED", None, None, Some("x")).is_err());
    }

    #[test]
    fn llm_job_and_cache_round_trip_sqlite() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let job = store
            .create_llm_job(&NewLlmJob {
                operation: "smoke".to_string(),
                provider: "ai-gateway".to_string(),
                model: "m".to_string(),
                input_hash: "abc".to_string(),
                prompt_version: "v1".to_string(),
            })
            .unwrap();
        store
            .record_llm_attempt(job.id, 1, "FAILED", None, None, Some("boom"))
            .unwrap();
        store
            .put_llm_cache(&LlmCacheEntry {
                cache_hash: "h9".to_string(),
                operation: "smoke".to_string(),
                provider: "ai-gateway".to_string(),
                model: "m".to_string(),
                prompt_version: "v1".to_string(),
                request_json: "{}".to_string(),
                response_json: "ok".to_string(),
                status: "OK".to_string(),
                created_at: "2026-01-01".to_string(),
            })
            .unwrap();
        // Upsert replaces the payload on hash collision.
        store
            .put_llm_cache(&LlmCacheEntry {
                cache_hash: "h9".to_string(),
                operation: "smoke".to_string(),
                provider: "ai-gateway".to_string(),
                model: "m".to_string(),
                prompt_version: "v1".to_string(),
                request_json: "{}".to_string(),
                response_json: "ok2".to_string(),
                status: "OK".to_string(),
                created_at: "2026-01-02".to_string(),
            })
            .unwrap();
        let hit = store.get_llm_cache("h9").unwrap().unwrap();
        assert_eq!(hit.response_json, "ok2");
    }

    #[test]
    fn schema_version_row_present() {
        let store = SqliteStore::open_in_memory().unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    fn sample_mcq_items(chapter_id: i64) -> Vec<NewMcqItem> {
        vec![
            NewMcqItem {
                chapter_id,
                phase: "pretest".to_string(),
                question_text: "What does &x yield?".to_string(),
                options_json:
                    "[\"addr\",\"value\",\"null\",\"dangling\"]".to_string(),
                correct_index: 0,
                trap_index: 1,
                explanation_text: "The & operator takes addresses plainly.".to_string(),
                source_refs:
                    "{\"pages\":[12],\"sections\":[\"Addresses\"]}".to_string(),
                topic: "addresses".to_string(),
                attempt_no: 1,
            },
            NewMcqItem {
                chapter_id,
                phase: "pretest".to_string(),
                question_text: "What does *p read?".to_string(),
                options_json:
                    "[\"pointee\",\"address\",\"null\",\"type\"]".to_string(),
                correct_index: 0,
                trap_index: 2,
                explanation_text: "Dereference reads the pointed-to value here.".to_string(),
                source_refs:
                    "{\"pages\":[13],\"sections\":[\"Deref\"]}".to_string(),
                topic: "deref".to_string(),
                attempt_no: 1,
            },
        ]
    }

    fn sample_assignment_questions(chapter_id: i64, attempt_no: i64) -> Vec<NewAssignmentQuestion> {
        let rubric = "{\"criteria\":[{\"name\":\"correctness\",\"max_score\":5,\"what_good_looks_like\":\"right\"}],\"max_score\":5,\"model_solution\":\"sol\"}";
        (0..3)
            .map(|position| NewAssignmentQuestion {
                chapter_id,
                position,
                kind: "written".to_string(),
                parts_json: "[\"a) ...\",\"b) ...\"]".to_string(),
                rubric_json: rubric.to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no,
            })
            .chain(std::iter::once(NewAssignmentQuestion {
                chapter_id,
                position: 3,
                kind: "coding".to_string(),
                parts_json: "[\"Write a program that ...\"]".to_string(),
                rubric_json: rubric.to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no,
            }))
            .collect()
    }

    fn chapter_for(store: &mut dyn Store) -> Chapter {
        let book = store.create_book(&sample_book(), "2026-01-01").unwrap();
        store
            .create_chapter(&NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 25,
                end_page: 60,
                file_path: "unit_1.json".to_string(),
                status: ChapterStatus::PretestReady,
            })
            .unwrap()
    }

    #[test]
    fn mcq_round_trip_memory() {
        let mut store = MemoryStore::new();
        let chapter = chapter_for(&mut store);
        let saved = store.save_mcq_items(&sample_mcq_items(chapter.id)).unwrap();
        assert_eq!(saved.len(), 2);
        let listed = store.list_mcq_items(chapter.id, "pretest", 1).unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed[0].id < listed[1].id);
        // Other phases/attempts are isolated.
        assert_eq!(store.list_mcq_items(chapter.id, "retest", 1).unwrap().len(), 0);
        assert_eq!(store.list_mcq_items(chapter.id, "pretest", 2).unwrap().len(), 0);
        let response = store
            .record_mcq_response(saved[0].id, 0, true, false, "2026-01-05", 1)
            .unwrap();
        assert!(response.is_correct);
        let responses = store.list_mcq_responses(saved[0].id).unwrap();
        assert_eq!(responses.len(), 1);
        assert_eq!(store.list_mcq_responses(saved[1].id).unwrap().len(), 0);
        // Unknown items fail loudly, never silently.
        assert!(store.record_mcq_response(9999, 0, false, false, "2026-01-05", 1).is_err());
        // Bad attempt numbers are rejected.
        let mut bad = sample_mcq_items(chapter.id);
        bad[0].attempt_no = 0;
        assert!(store.save_mcq_items(&bad).is_err());
    }

    #[test]
    fn mcq_round_trip_sqlite() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let chapter = chapter_for(&mut store);
        let saved = store.save_mcq_items(&sample_mcq_items(chapter.id)).unwrap();
        assert_eq!(saved.len(), 2);
        let listed = store.list_mcq_items(chapter.id, "pretest", 1).unwrap();
        assert_eq!(listed[0].question_text, "What does &x yield?");
        store
            .record_mcq_response(saved[1].id, 2, false, true, "2026-01-06", 1)
            .unwrap();
        let responses = store.list_mcq_responses(saved[1].id).unwrap();
        assert_eq!(responses.len(), 1);
        assert!(!responses[0].is_correct);
        assert!(responses[0].selected_trap);
    }

    #[test]
    fn delete_mcq_items_for_replaces_experiments() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let saved = store.save_mcq_items(&sample_mcq_items(chapter.id)).unwrap();
            store
                .record_mcq_response(saved[0].id, 0, true, false, "2026-01-05", 1)
                .unwrap();
            let mut retest = sample_mcq_items(chapter.id);
            for item in &mut retest {
                item.phase = "retest".to_string();
            }
            store.save_mcq_items(&retest[..1]).unwrap();
            assert_eq!(
                store
                    .delete_mcq_items_for(chapter.id, "pretest", 1)
                    .unwrap(),
                2
            );
            assert_eq!(store.list_mcq_items(chapter.id, "pretest", 1).unwrap().len(), 0);
            // Attached responses go with the items; other phases are untouched.
            assert_eq!(store.list_mcq_responses(saved[0].id).unwrap().len(), 0);
            assert_eq!(store.list_mcq_items(chapter.id, "retest", 1).unwrap().len(), 1);
            assert_eq!(store.delete_mcq_items_for(chapter.id, "pretest", 1).unwrap(), 0);
        }
    }

    #[test]
    fn assignment_round_trip_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let saved = store
                .save_assignment_questions(&sample_assignment_questions(chapter.id, 1))
                .unwrap();
            assert_eq!(saved.len(), 4);
            assert_eq!(saved[0].kind, "written");
            assert_eq!(saved[3].kind, "coding");
            let listed = store.list_assignment_questions(chapter.id, 1).unwrap();
            assert_eq!(listed.len(), 4);
            assert!(listed.windows(2).all(|w| w[0].position <= w[1].position));
            // Attempts are isolated.
            assert_eq!(store.list_assignment_questions(chapter.id, 2).unwrap().len(), 0);
            let response = store
                .record_assignment_response(saved[0].id, "my answer", "2026-01-07", 1)
                .unwrap();
            assert_eq!(response.answer_text, "my answer");
            assert_eq!(store.list_assignment_responses(saved[0].id).unwrap().len(), 1);
            assert_eq!(store.list_assignment_responses(saved[1].id).unwrap().len(), 0);
            // Replace deletes questions and their responses together.
            assert_eq!(store.delete_assignment_questions_for(chapter.id, 1).unwrap(), 4);
            assert_eq!(store.list_assignment_questions(chapter.id, 1).unwrap().len(), 0);
            assert_eq!(store.list_assignment_responses(saved[0].id).unwrap().len(), 0);
            assert_eq!(store.delete_assignment_questions_for(chapter.id, 1).unwrap(), 0);
            // Bad attempts and kinds are rejected loudly.
            let mut bad = sample_assignment_questions(chapter.id, 1);
            bad[0].attempt_no = 0;
            assert!(store.save_assignment_questions(&bad).is_err());
            let mut bad_kind = sample_assignment_questions(chapter.id, 1);
            bad_kind[0].kind = "quiz".to_string();
            assert!(store.save_assignment_questions(&bad_kind).is_err());
        }
    }

    #[test]
    fn misconception_lifecycle_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let row = store
                .create_misconception(
                    chapter.id,
                    "addresses",
                    "took &x for the value",
                    "selected 'value' instead of 'address'",
                    "RETEST",
                    "2026-01-06",
                )
                .unwrap();
            assert_eq!(row.status, "ACTIVE");
            let listed = store.list_misconceptions(chapter.id).unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].concept_description, "addresses");
            assert_eq!(store.list_misconceptions(chapter.id + 999).unwrap().len(), 0);
            assert!(store
                .create_misconception(chapter.id, " ", "desc", "ev", "RETEST", "2026-01-06")
                .is_err());
        }
    }

    #[test]
    fn misconception_update_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let row = store
                .create_misconception(
                    chapter.id,
                    "addresses",
                    "took &x for the value",
                    "selected 'value'",
                    "RETEST",
                    "2026-01-06",
                )
                .unwrap();
            assert_eq!(row.confidence, 0.5);
            assert_eq!(row.updated_at, "2026-01-06");
            assert_eq!(row.resolved_at, None);
            // A nudge preserves the row identity and stamps updated_at.
            let nudged = store
                .update_misconception(row.id, 0.6, "IMPROVING", "2026-01-07", None)
                .unwrap();
            assert_eq!(nudged.id, row.id);
            assert_eq!(nudged.confidence, 0.6);
            assert_eq!(nudged.status, "IMPROVING");
            assert_eq!(nudged.updated_at, "2026-01-07");
            assert_eq!(nudged.resolved_at, None);
            assert_eq!(nudged.concept_description, "addresses");
            // Resolving stamps resolved_at; a later nudge preserves it when
            // the caller passes it back through.
            let resolved = store
                .update_misconception(row.id, 0.9, "RESOLVED", "2026-01-08", Some("2026-01-08"))
                .unwrap();
            assert_eq!(resolved.status, "RESOLVED");
            assert_eq!(resolved.resolved_at, Some("2026-01-08".to_string()));
            let kept = store
                .update_misconception(row.id, 0.95, "RESOLVED", "2026-01-09", resolved.resolved_at.as_deref())
                .unwrap();
            assert_eq!(kept.resolved_at, Some("2026-01-08".to_string()));
            // Bad inputs fail loudly on both backends.
            assert!(store.update_misconception(row.id, -0.1, "ACTIVE", "2026-01-09", None).is_err());
            assert!(store.update_misconception(row.id, 1.1, "ACTIVE", "2026-01-09", None).is_err());
            assert!(store.update_misconception(row.id, f64::NAN, "ACTIVE", "2026-01-09", None).is_err());
            assert!(store.update_misconception(row.id, 0.5, "STALE", "2026-01-09", None).is_err());
            assert!(store.update_misconception(row.id + 999, 0.5, "ACTIVE", "2026-01-09", None).is_err());
        }
    }

    #[test]
    fn reschedule_task_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let future = NaiveDate::from_ymd_opt(2026, 1, 12).unwrap();
            let task = store
                .create_task(&NewTask {
                    book_id: chapter.book_id,
                    chapter_id: chapter.id,
                    task_type: TaskType::Retest,
                    scheduled_for: future,
                    sequence: 1,
                    attempt_no: 1,
                })
                .unwrap();
            let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
            let moved_task = store.reschedule_task(task.id, today).unwrap();
            assert_eq!(moved_task.id, task.id);
            assert_eq!(moved_task.scheduled_for, today);
            assert_eq!(moved_task.status, TaskStatus::Pending);
            assert_eq!(store.get_task(task.id).unwrap().scheduled_for, today);
            // Rescheduling a DONE task fails loudly without mutating state.
            store.complete_task(task.id, "2026-01-10").unwrap();
            let done_err = store.reschedule_task(task.id, future).unwrap_err();
            assert!(matches!(done_err, Error::AlreadyCompleted(_)));
            assert_eq!(
                store.get_task(task.id).unwrap().scheduled_for,
                today
            );
            assert!(store.reschedule_task(task.id + 999, today).is_err());
        }
    }

    #[test]
    fn skip_attempt_flow_memory() {
        let mut store = MemoryStore::new();
        let chapter = chapter_for(&mut store);
        let date = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();
        let first = store
            .create_task(&NewTask {
                book_id: chapter.book_id,
                chapter_id: chapter.id,
                task_type: TaskType::Pretest,
                scheduled_for: date,
                sequence: 1,
                attempt_no: 1,
            })
            .unwrap();
        store
            .create_task(&NewTask {
                book_id: chapter.book_id,
                chapter_id: chapter.id,
                task_type: TaskType::Read,
                scheduled_for: date,
                sequence: 2,
                attempt_no: 1,
            })
            .unwrap();
        store.complete_task(first.id, "2026-01-05").unwrap();
        // Skip deletes only pending rows; completed work stays as audit trail.
        assert_eq!(store.delete_pending_tasks_for_chapter(chapter.id).unwrap(), 1);
        let remaining = store.list_tasks().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].status, TaskStatus::Done);
        // Unskip bumps the attempt; metrics filter on it.
        store.set_chapter_attempt(chapter.id, 2).unwrap();
        assert_eq!(store.get_chapter(chapter.id).unwrap().attempt_no, 2);
        assert!(store.set_chapter_attempt(chapter.id, 0).is_err());
        assert!(store.set_chapter_attempt(9999, 2).is_err());
        // Skipped chapters yield no next task.
        store.set_chapter_status(chapter.id, ChapterStatus::Skipped).unwrap();
        assert_eq!(store.get_chapter(chapter.id).unwrap().status.next_task(), None);
    }

    /// Build a v1 database file (no `attempt_no`, no MCQ/assignment tables),
    /// then open it through [`SqliteStore::open`] and assert the migration
    /// chain (v1 → v2 → v3) preserves rows and lands on [`SCHEMA_VERSION`].
    #[test]
    fn migrates_v1_to_v2_preserving_rows() {
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "migration-v1-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("cadence.db");
        let lock_path = dir.join("cadence.lock");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
                 INSERT INTO version (id, schema_version) VALUES (1, 1);
                 CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filepath TEXT NOT NULL, file_hash TEXT NOT NULL, start_page INTEGER NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), index_in_book INTEGER NOT NULL, level INTEGER NOT NULL, title TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL);
                 CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), chapter_id INTEGER NOT NULL REFERENCES chapters(id), type TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL, completed_at TEXT, sequence INTEGER NOT NULL);
                 CREATE TABLE event_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, chapter_id INTEGER, task_id INTEGER, evidence TEXT, created_at TEXT NOT NULL);
                 CREATE TABLE llm_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL, prompt_version TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL, raw_response TEXT, parsed_response TEXT, error TEXT);
                 CREATE TABLE llm_cache (cache_hash TEXT PRIMARY KEY, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                 INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES ('Legacy', '/b.pdf', 'h', 20, '2026-01-01');
                 INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status) VALUES (1, 0, 1, 'Ch 1', 20, 40, 'u.json', 'PRETEST_READY');
                 INSERT INTO tasks (book_id, chapter_id, type, scheduled_for, status, completed_at, sequence) VALUES (1, 1, 'PRETEST', '2026-01-05', 'PENDING', NULL, 1);",
            )
            .unwrap();
        }
        let mut store = SqliteStore::open(&db_path, &lock_path).unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        // Legacy rows survive with backfilled attempt 1.
        let chapter = store.get_chapter(1).unwrap();
        assert_eq!(chapter.attempt_no, 1);
        assert_eq!(store.list_tasks().unwrap().len(), 1);
        // New v2 writes work on the migrated DB.
        let saved = store.save_mcq_items(&sample_mcq_items(1)).unwrap();
        assert_eq!(saved.len(), 2);
        assert_eq!(store.list_mcq_items(1, "pretest", 1).unwrap().len(), 2);
        // New v3 writes work too (assignment tables arrived via the chain).
        let questions = store.save_assignment_questions(&sample_assignment_questions(1, 1)).unwrap();
        assert_eq!(questions.len(), 4);
        assert_eq!(store.list_assignment_questions(1, 1).unwrap().len(), 4);
        // Reopen is idempotent (migration reruns safely).
        drop(store);
        let store2 = SqliteStore::open(&db_path, &lock_path).unwrap();
        assert_eq!(store2.get_chapter(1).unwrap().attempt_no, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a v2 database file (with `attempt_no` + MCQ tables but no
    /// assignment tables), then assert the v2 → v3 step adds them while
    /// preserving existing rows.
    #[test]
    fn migrates_v2_to_v3_preserving_rows() {
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "migration-v2-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("cadence.db");
        let lock_path = dir.join("cadence.lock");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
                 INSERT INTO version (id, schema_version) VALUES (1, 2);
                 CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filepath TEXT NOT NULL, file_hash TEXT NOT NULL, start_page INTEGER NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), index_in_book INTEGER NOT NULL, level INTEGER NOT NULL, title TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), chapter_id INTEGER NOT NULL REFERENCES chapters(id), type TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL, completed_at TEXT, sequence INTEGER NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE event_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, chapter_id INTEGER, task_id INTEGER, evidence TEXT, created_at TEXT NOT NULL);
                 CREATE TABLE llm_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL, prompt_version TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL, raw_response TEXT, parsed_response TEXT, error TEXT);
                 CREATE TABLE llm_cache (cache_hash TEXT PRIMARY KEY, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE mcq_items (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), phase TEXT NOT NULL, question_text TEXT NOT NULL, options_json TEXT NOT NULL, correct_index INTEGER NOT NULL, trap_index INTEGER NOT NULL, explanation_text TEXT NOT NULL, source_refs TEXT NOT NULL, topic TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE mcq_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id), selected_index INTEGER NOT NULL, is_correct INTEGER NOT NULL, selected_trap INTEGER NOT NULL, answered_at TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE misconceptions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), concept_description TEXT NOT NULL, description TEXT NOT NULL, evidence TEXT NOT NULL, source_task TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, resolved_at TEXT);
                 INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES ('Legacy', '/b.pdf', 'h', 20, '2026-01-01');
                 INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no) VALUES (1, 0, 1, 'Ch 1', 20, 40, 'u.json', 'READ_COMPLETE', 1);",
            )
            .unwrap();
        }
        let mut store = SqliteStore::open(&db_path, &lock_path).unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        // v2 rows survive; v3 tables accept writes.
        assert_eq!(store.get_chapter(1).unwrap().status, ChapterStatus::ReadComplete);
        let questions = store.save_assignment_questions(&sample_assignment_questions(1, 1)).unwrap();
        assert_eq!(questions.len(), 4);
        drop(store);
        let store2 = SqliteStore::open(&db_path, &lock_path).unwrap();
        assert_eq!(store2.list_assignment_questions(1, 1).unwrap().len(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn sample_grade(question_id: i64) -> NewGrade {
        NewGrade {
            question_id,
            score: 2,
            max_score: 5,
            classification: "INCORRECT".to_string(),
            criteria_results_json: "[]".to_string(),
            feedback: "One lea cannot emit 5*x.".to_string(),
            grader_version: "v3".to_string(),
            created_at: "2026-01-08".to_string(),
        }
    }

    fn sample_dispute(grade_id: i64) -> NewDispute {
        NewDispute {
            grade_id,
            text: "Scales exclude 5, so one lea is impossible.".to_string(),
            decision: "REVISED".to_string(),
            final_score: 5,
            adjudication_json: "{\"action\":\"REVISED\"}".to_string(),
            adjudicator_model: "gemini-3.6-flash".to_string(),
            timestamp: "2026-01-09".to_string(),
        }
    }

    #[test]
    fn grades_disputes_round_trip_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            let questions = store
                .save_assignment_questions(&sample_assignment_questions(chapter.id, 1))
                .unwrap();
            let question_id = questions[0].id;
            let saved = store.save_grade(&sample_grade(question_id)).unwrap();
            assert!(!saved.disputed);
            assert_eq!(store.get_grade(saved.id).unwrap(), saved);
            assert_eq!(store.list_grades_for_question(question_id).unwrap().len(), 1);
            assert_eq!(store.list_grades_for_question(question_id + 999).unwrap().len(), 0);
            assert_eq!(store.list_disputes_for_grade(saved.id).unwrap().len(), 0);
            // Recording a dispute flips the grade without touching the award.
            let audit = store.record_dispute(&sample_dispute(saved.id)).unwrap();
            assert_eq!(audit.decision, "REVISED");
            let corrected = store.get_grade(saved.id).unwrap();
            assert!(corrected.disputed);
            assert_eq!(corrected.score, 2);
            assert_eq!(corrected.original_score, Some(2));
            assert_eq!(corrected.final_score, Some(5));
            assert_eq!(store.list_disputes_for_grade(saved.id).unwrap().len(), 1);
            // Re-dispute preserves the FIRST original, not the intermediate.
            let mut second = sample_dispute(saved.id);
            second.decision = "UPHELD".to_string();
            second.final_score = 2;
            store.record_dispute(&second).unwrap();
            let twice = store.get_grade(saved.id).unwrap();
            assert_eq!(twice.original_score, Some(2));
            assert_eq!(store.list_disputes_for_grade(saved.id).unwrap().len(), 2);
            // Bad inputs fail loudly on both backends.
            let mut bad_score = sample_grade(question_id);
            bad_score.score = 9;
            assert!(store.save_grade(&bad_score).is_err());
            let mut bad_class = sample_grade(question_id);
            bad_class.classification = " ".to_string();
            assert!(store.save_grade(&bad_class).is_err());
            let mut bad_decision = sample_dispute(saved.id);
            bad_decision.decision = "OVERTURNED".to_string();
            assert!(store.record_dispute(&bad_decision).is_err());
            let mut bad_final = sample_dispute(saved.id);
            bad_final.final_score = 9;
            assert!(store.record_dispute(&bad_final).is_err());
            let mut bad_text = sample_dispute(saved.id);
            bad_text.text = " ".to_string();
            assert!(store.record_dispute(&bad_text).is_err());
            assert!(store.record_dispute(&sample_dispute(saved.id + 999)).is_err());
            assert!(store.get_grade(saved.id + 999).is_err());
        }
    }

    fn sample_note(chapter_id: i64, attempt_no: i64) -> NewNote {
        NewNote {
            chapter_id,
            content_markdown: "## Addresses\nAddresses.\n## Null checks\n& takes addresses; a null pointer is false.\n## Aliasing\nTwo pointers can name the same cell.".to_string(),
            generated_at: "2026-09-25".to_string(),
            attempt_no,
        }
    }

    #[test]
    fn notes_round_trip_both_backends() {
        let mut stores: Vec<Box<dyn Store>> = vec![
            Box::new(MemoryStore::new()),
            Box::new(SqliteStore::open_in_memory().unwrap()),
        ];
        for store in &mut stores {
            let chapter = chapter_for(store.as_mut());
            assert_eq!(store.list_notes(chapter.id, 1).unwrap().len(), 0);
            let saved = store.save_note(&sample_note(chapter.id, 1)).unwrap();
            assert_eq!(saved.chapter_id, chapter.id);
            assert_eq!(saved.attempt_no, 1);
            assert!(saved.content_markdown.contains("Null checks"));
            let listed = store.list_notes(chapter.id, 1).unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0], saved);
            // Attempts stay isolated (§4.1): a fresh attempt starts empty.
            assert_eq!(store.list_notes(chapter.id, 2).unwrap().len(), 0);
            let second = store.save_note(&sample_note(chapter.id, 2)).unwrap();
            assert_eq!(store.list_notes(chapter.id, 2).unwrap().len(), 1);
            assert_eq!(store.list_notes(chapter.id, 1).unwrap().len(), 1);
            assert_ne!(saved.id, second.id);
            // Other chapters stay isolated too.
            assert_eq!(store.list_notes(chapter.id + 999, 1).unwrap().len(), 0);
            // Blank markdown fails loudly on both backends.
            let mut blank = sample_note(chapter.id, 1);
            blank.content_markdown = "   ".to_string();
            assert!(store.save_note(&blank).is_err());
        }
    }

    /// Build a v3 database file (through `assignment_responses`, no
    /// grades/disputes), then assert the v3 → v4 step adds them while
    /// preserving existing rows.
    #[test]
    fn migrates_v3_to_v4_preserving_rows() {
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "migration-v3-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("cadence.db");
        let lock_path = dir.join("cadence.lock");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
                 INSERT INTO version (id, schema_version) VALUES (1, 3);
                 CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filepath TEXT NOT NULL, file_hash TEXT NOT NULL, start_page INTEGER NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), index_in_book INTEGER NOT NULL, level INTEGER NOT NULL, title TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), chapter_id INTEGER NOT NULL REFERENCES chapters(id), type TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL, completed_at TEXT, sequence INTEGER NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE event_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, chapter_id INTEGER, task_id INTEGER, evidence TEXT, created_at TEXT NOT NULL);
                 CREATE TABLE llm_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL, prompt_version TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL, raw_response TEXT, parsed_response TEXT, error TEXT);
                 CREATE TABLE llm_cache (cache_hash TEXT PRIMARY KEY, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE mcq_items (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), phase TEXT NOT NULL, question_text TEXT NOT NULL, options_json TEXT NOT NULL, correct_index INTEGER NOT NULL, trap_index INTEGER NOT NULL, explanation_text TEXT NOT NULL, source_refs TEXT NOT NULL, topic TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE mcq_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id), selected_index INTEGER NOT NULL, is_correct INTEGER NOT NULL, selected_trap INTEGER NOT NULL, answered_at TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE misconceptions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), concept_description TEXT NOT NULL, description TEXT NOT NULL, evidence TEXT NOT NULL, source_task TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, resolved_at TEXT);
                 CREATE TABLE assignment_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), position INTEGER NOT NULL, kind TEXT NOT NULL, parts_json TEXT NOT NULL, rubric_json TEXT NOT NULL, target_misconception_ids TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE assignment_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, question_id INTEGER NOT NULL REFERENCES assignment_questions(id), answer_text TEXT NOT NULL, answered_at TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES ('Legacy', '/b.pdf', 'h', 20, '2026-01-01');
                 INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no) VALUES (1, 0, 1, 'Ch 1', 20, 40, 'u.json', 'READ_COMPLETE', 1);
                 INSERT INTO assignment_questions (chapter_id, position, kind, parts_json, rubric_json, target_misconception_ids, attempt_no) VALUES (1, 0, 'written', '[\"a) ...\"]', '{\"criteria\":[],\"max_score\":1,\"model_solution\":\"s\"}', '[]', 1);",
            )
            .unwrap();
        }
        let mut store = SqliteStore::open(&db_path, &lock_path).unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        // v3 rows survive; v4 tables accept writes with the dispute trail.
        assert_eq!(store.list_assignment_questions(1, 1).unwrap().len(), 1);
        let grade = store.save_grade(&sample_grade(1)).unwrap();
        store.record_dispute(&sample_dispute(grade.id)).unwrap();
        let corrected = store.get_grade(grade.id).unwrap();
        assert!(corrected.disputed);
        assert_eq!(corrected.original_score, Some(2));
        drop(store);
        let store2 = SqliteStore::open(&db_path, &lock_path).unwrap();
        assert_eq!(store2.list_disputes_for_grade(grade.id).unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a v4 database file (through grades/disputes, no notes), then
    /// assert the v4 → v5 step adds notes while preserving existing rows.
    #[test]
    fn migrates_v4_to_v5_preserving_rows() {
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "migration-v4-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("cadence.db");
        let lock_path = dir.join("cadence.lock");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
                 INSERT INTO version (id, schema_version) VALUES (1, 4);
                 CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filepath TEXT NOT NULL, file_hash TEXT NOT NULL, start_page INTEGER NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), index_in_book INTEGER NOT NULL, level INTEGER NOT NULL, title TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), chapter_id INTEGER NOT NULL REFERENCES chapters(id), type TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL, completed_at TEXT, sequence INTEGER NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE event_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, chapter_id INTEGER, task_id INTEGER, evidence TEXT, created_at TEXT NOT NULL);
                 CREATE TABLE llm_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL, prompt_version TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL, raw_response TEXT, parsed_response TEXT, error TEXT);
                 CREATE TABLE llm_cache (cache_hash TEXT PRIMARY KEY, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE mcq_items (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), phase TEXT NOT NULL, question_text TEXT NOT NULL, options_json TEXT NOT NULL, correct_index INTEGER NOT NULL, trap_index INTEGER NOT NULL, explanation_text TEXT NOT NULL, source_refs TEXT NOT NULL, topic TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE mcq_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id), selected_index INTEGER NOT NULL, is_correct INTEGER NOT NULL, selected_trap INTEGER NOT NULL, answered_at TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE misconceptions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), concept_description TEXT NOT NULL, description TEXT NOT NULL, evidence TEXT NOT NULL, source_task TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, resolved_at TEXT);
                 CREATE TABLE assignment_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), position INTEGER NOT NULL, kind TEXT NOT NULL, parts_json TEXT NOT NULL, rubric_json TEXT NOT NULL, target_misconception_ids TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE assignment_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, question_id INTEGER NOT NULL REFERENCES assignment_questions(id), answer_text TEXT NOT NULL, answered_at TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE grades (id INTEGER PRIMARY KEY AUTOINCREMENT, question_id INTEGER NOT NULL REFERENCES assignment_questions(id), score INTEGER NOT NULL, max_score INTEGER NOT NULL, classification TEXT NOT NULL, criteria_results_json TEXT NOT NULL, feedback TEXT NOT NULL, disputed INTEGER NOT NULL DEFAULT 0, original_score INTEGER, dispute_text TEXT, final_score INTEGER, adjudication_json TEXT, grader_version TEXT NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE disputes (id INTEGER PRIMARY KEY AUTOINCREMENT, grade_id INTEGER NOT NULL REFERENCES grades(id), dispute_text TEXT NOT NULL, decision TEXT NOT NULL, final_score INTEGER NOT NULL, adjudicator_model TEXT NOT NULL, timestamp TEXT NOT NULL);
                 INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES ('Legacy', '/b.pdf', 'h', 20, '2026-01-01');
                 INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no) VALUES (1, 0, 1, 'Ch 1', 20, 40, 'u.json', 'ASSIGNMENT_COMPLETE', 1);
                 INSERT INTO assignment_questions (chapter_id, position, kind, parts_json, rubric_json, target_misconception_ids, attempt_no) VALUES (1, 0, 'written', '[\"a) ...\"]', '{\"criteria\":[],\"max_score\":1,\"model_solution\":\"s\"}', '[]', 1);
                 INSERT INTO grades (question_id, score, max_score, classification, criteria_results_json, feedback, grader_version, created_at) VALUES (1, 4, 5, 'CORRECT_BUT_BRIEF', '[]', 'Brief but right.', 'v3', '2026-01-08');",
            )
            .unwrap();
        }
        let mut store = SqliteStore::open(&db_path, &lock_path).unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        // v4 rows survive; v5 notes accept writes and persist across reopen.
        assert_eq!(store.list_grades_for_question(1).unwrap().len(), 1);
        let saved = store.save_note(&sample_note(1, 1)).unwrap();
        assert!(saved.content_markdown.contains("Null checks"));
        drop(store);
        let store2 = SqliteStore::open(&db_path, &lock_path).unwrap();
        assert_eq!(store2.list_notes(1, 1).unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a drifted v5 database file (tables missing columns that later
    /// builds added to fresh-table DDL without a repair path — the live
    /// failure was `no such column: concept_description`), then assert the
    /// v5 → v6 step backfills every column losslessly and rows stay
    /// queryable through the v6 accessors.
    #[test]
    fn repairs_drifted_v5_tables_preserving_rows() {
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "migration-v5-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("cadence.db");
        let lock_path = dir.join("cadence.lock");
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE version (id INTEGER PRIMARY KEY CHECK (id = 1), schema_version INTEGER NOT NULL);
                 INSERT INTO version (id, schema_version) VALUES (1, 5);
                 CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filepath TEXT NOT NULL, file_hash TEXT NOT NULL, start_page INTEGER NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), index_in_book INTEGER NOT NULL, level INTEGER NOT NULL, title TEXT NOT NULL, start_page INTEGER NOT NULL, end_page INTEGER NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL REFERENCES books(id), chapter_id INTEGER NOT NULL REFERENCES chapters(id), type TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL, completed_at TEXT, sequence INTEGER NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE event_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, chapter_id INTEGER, task_id INTEGER, evidence TEXT, created_at TEXT NOT NULL);
                 CREATE TABLE llm_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, input_hash TEXT NOT NULL, prompt_version TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL, raw_response TEXT, parsed_response TEXT, error TEXT);
                 CREATE TABLE llm_cache (cache_hash TEXT PRIMARY KEY, operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                 CREATE TABLE mcq_items (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), phase TEXT NOT NULL, question_text TEXT NOT NULL, options_json TEXT NOT NULL, correct_index INTEGER NOT NULL, trap_index INTEGER NOT NULL, explanation_text TEXT NOT NULL, source_refs TEXT NOT NULL, topic TEXT NOT NULL);
                 CREATE TABLE mcq_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, mcq_item_id INTEGER NOT NULL REFERENCES mcq_items(id), selected_index INTEGER NOT NULL, is_correct INTEGER NOT NULL, selected_trap INTEGER NOT NULL, answered_at TEXT NOT NULL);
                 CREATE TABLE misconceptions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), description TEXT NOT NULL, evidence TEXT NOT NULL, source_task TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL, resolved_at TEXT);
                 CREATE TABLE assignment_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), position INTEGER NOT NULL, parts_json TEXT NOT NULL, rubric_json TEXT NOT NULL, target_misconception_ids TEXT NOT NULL);
                 CREATE TABLE assignment_responses (id INTEGER PRIMARY KEY AUTOINCREMENT, question_id INTEGER NOT NULL REFERENCES assignment_questions(id), answer_text TEXT NOT NULL, answered_at TEXT NOT NULL);
                 CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL REFERENCES chapters(id), content_markdown TEXT NOT NULL, generated_at TEXT NOT NULL);
                 INSERT INTO books (title, filepath, file_hash, start_page, created_at) VALUES ('Legacy', '/b.pdf', 'h', 20, '2026-01-01');
                 INSERT INTO chapters (book_id, index_in_book, level, title, start_page, end_page, file_path, status, attempt_no) VALUES (1, 0, 1, 'Ch 1', 20, 40, 'u.json', 'COMPLETED', 1);
                 INSERT INTO misconceptions (chapter_id, description, evidence, source_task, status, confidence, created_at, resolved_at) VALUES (1, 'took &x for the value', 'picked value', 'RETEST', 'ACTIVE', 0.5, '2026-01-06', NULL);
                 INSERT INTO mcq_items (chapter_id, phase, question_text, options_json, correct_index, trap_index, explanation_text, source_refs, topic) VALUES (1, 'retest', 'What does &x yield?', '[\"a\",\"b\",\"c\",\"d\"]', 0, 1, 'Because reasons plainly stated here.', '{\"pages\":[22],\"sections\":[\"S\"]}', 'addresses');
                 INSERT INTO assignment_questions (chapter_id, position, parts_json, rubric_json, target_misconception_ids) VALUES (1, 0, '[\"a) ...\"]', '{\"criteria\":[],\"max_score\":1,\"model_solution\":\"s\"}', '[]');",
            )
            .unwrap();
        }
        let mut store = SqliteStore::open(&db_path, &lock_path).unwrap();
        let version: i64 = store
            .conn
            .query_row("SELECT schema_version FROM version WHERE id = 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        // The exact live failure now reads through the v6 accessors.
        let rows = store.list_misconceptions(1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].description, "took &x for the value");
        assert_eq!(rows[0].concept_description, String::new());
        assert_eq!(rows[0].updated_at, "2026-01-06");
        // Backfilled attempt 1 keeps old rows inside the live filters.
        assert_eq!(store.list_mcq_items(1, "retest", 1).unwrap().len(), 1);
        let questions = store.list_assignment_questions(1, 1).unwrap();
        assert_eq!(questions.len(), 1);
        // Pre-`kind` rows predate the coding question: all written.
        assert_eq!(questions[0].kind, "written");
        // Writes use the repaired columns immediately.
        let fresh = store
            .create_misconception(1, "deref", "star on non-pointer", "picked compiles", "RETEST", "2026-01-07")
            .unwrap();
        assert_eq!(fresh.concept_description, "deref");
        assert_eq!(fresh.updated_at, "2026-01-07");
        drop(store);
        // Repair is idempotent: reopening a v6 database migrates nothing.
        let store2 = SqliteStore::open(&db_path, &lock_path).unwrap();
        assert_eq!(store2.list_misconceptions(1).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
