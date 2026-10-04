//! CLI definition (`clap` derive). Mirrors §14 plus the §2.1 `dev` harness.
//!
//! `dev` commands never mutate `~/.cadence/`; they use a temporary or
//! in-memory store.

use clap::{Args, Parser, Subcommand};

/// Deterministic, cadence-driven CLI for deep self-study.
#[derive(Debug, Parser)]
#[command(name = "cadence", version, about = "Cadence study scheduler")]
pub struct Cli {
    /// Subcommand. None = today's scheduled loop.
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// All top-level commands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Ingest a new book PDF (requires --start-page).
    Ingest(IngestArgs),
    /// Show today's queue and progress.
    Today(TodayArgs),
    /// Calendar view (today's queue plus upcoming scheduled work).
    Schedule(ScheduleArgs),
    /// Pull eligible future work (only if mandatory work complete).
    Pull,
    /// Progress dashboard.
    Metrics,
    /// Active / resolved misconception list.
    Misconceptions,
    /// Book-level progress with retention metrics.
    Progress,
    /// Manual cumulative misconception retest session.
    Review,
    /// Skip a chapter (lists skippable chapters without args).
    Skip(SkipArgs),
    /// Return a skipped chapter to the pipeline (restarts fresh).
    Unskip(UnskipArgs),
    /// Show stored chapter notes (lists chapters without args).
    Notes(NotesArgs),
    /// Grading dispute workflow.
    Dispute(DisputeArgs),
    /// System check: DB, nvim, PDF parser, LLM API, config, cache integrity.
    Doctor,
    /// Integration harness: isolated stage runs against fixture PDFs.
    Dev(DevArgs),
}

/// Args for `cadence ingest`.
#[derive(Debug, Args)]
pub struct IngestArgs {
    /// Path to the book PDF.
    pub pdf: String,
    /// One-based physical page where study content starts (mandatory).
    #[arg(long)]
    pub start_page: Option<i64>,
    /// Maximum pages per study unit.
    #[arg(long, default_value_t = 50)]
    pub max_unit_pages: i64,
    /// Optional display title override.
    #[arg(long)]
    pub title: Option<String>,
    /// Manual boundaries as `start-end,start-end,...` (one-based, optional).
    #[arg(long)]
    pub manual_boundaries: Option<String>,
    /// Outline depth to treat as chapters (default: auto-detect the deepest
    /// chapter level). Explicit levels tile strictly: the page cap is
    /// ignored and `--manual-boundaries` is rejected.
    #[arg(long)]
    pub chapter_level: Option<i64>,
}

/// Args for `cadence today`.
#[derive(Debug, Args)]
pub struct TodayArgs {
    /// Reserved for future date override (`YYYY-MM-DD`).
    #[arg(long)]
    pub date: Option<String>,
}

/// Args for `cadence schedule`.
#[derive(Debug, Args)]
pub struct ScheduleArgs {
    /// Days of upcoming scheduled work to show.
    #[arg(long, default_value_t = 7)]
    pub days: i64,
}

/// Args for `cadence skip [id]`.
#[derive(Debug, Args)]
pub struct SkipArgs {
    /// Chapter id to skip (lists chapters with status when omitted).
    pub id: Option<i64>,
}

/// Args for `cadence unskip <id>`.
#[derive(Debug, Args)]
pub struct UnskipArgs {
    /// Skipped chapter id to return to the pipeline.
    pub id: i64,
}

/// Args for `cadence notes [id]`.
#[derive(Debug, Args)]
pub struct NotesArgs {
    /// Chapter id to print stored notes for (lists chapters when omitted).
    pub id: Option<i64>,
}

/// Args for `cadence dispute`.
#[derive(Debug, Args)]
pub struct DisputeArgs {
    /// Chapter id whose assignment set holds the disputed question
    /// (omit to pick from a list of graded assignments).
    pub assignment_id: Option<i64>,
    /// Question number within the assignment (1-based position;
    /// omit to pick from the chapter's graded questions).
    #[arg(long)]
    pub question: Option<usize>,
    /// Dispute text (otherwise read from stdin).
    #[arg(long)]
    pub text: Option<String>,
}

/// Args for `cadence dev`.
#[derive(Debug, Args)]
pub struct DevArgs {
    /// Stage subcommand.
    #[command(subcommand)]
    pub stage: DevStage,
}

/// Isolated integration stages (§2.1).
#[derive(Debug, Subcommand)]
pub enum DevStage {
    /// Ingestion + split only, prints unit JSON.
    Ingest(DevIngestArgs),
    /// Present MCQs interactively (pretest|retest).
    Mcq(DevMcqArgs),
    /// Generate an assignment.
    Assignment(DevAssignmentArgs),
    /// Grade answers against a rubric.
    Grade(DevGradeArgs),
    /// Generate chapter notes.
    Notes(DevNotesArgs),
    /// Deterministic scheduler scenario, no LLM.
    Schedule(DevScheduleArgs),
    /// LLM transport smoke test.
    Llm(DevLlmArgs),
}

/// Args for `cadence dev ingest`.
#[derive(Debug, Args)]
pub struct DevIngestArgs {
    /// Fixture single-chapter PDF.
    #[arg(long)]
    pub pdf: String,
    /// One-based physical page where study content starts.
    #[arg(long, default_value_t = 1)]
    pub start_page: i64,
    /// Maximum pages per study unit.
    #[arg(long, default_value_t = 50)]
    pub max_unit_pages: i64,
    /// Manual boundaries as `start-end,...` (only used when a leaf exceeds
    /// the cap with no semantic split).
    #[arg(long)]
    pub manual_boundaries: Option<String>,
    /// Outline depth to treat as chapters (default: auto-detect the deepest
    /// chapter level). Explicit levels tile strictly: the page cap is
    /// ignored and `--manual-boundaries` is rejected.
    #[arg(long)]
    pub chapter_level: Option<i64>,
    /// Limit text extraction to the first N units (keeps fixture output small).
    #[arg(long)]
    pub max_units: Option<usize>,
}

/// Args for `cadence dev mcq`.
#[derive(Debug, Args)]
pub struct DevMcqArgs {
    /// Fixture single-chapter PDF.
    #[arg(long)]
    pub pdf: String,
    /// Assessment phase.
    #[arg(long, default_value = "pretest")]
    pub phase: String,
    /// One-based physical page where the fixture unit starts (front matter
    /// before this is excluded, mirroring `ingest --start-page`).
    #[arg(long, default_value_t = 1)]
    pub start_page: i64,
    /// Persist to `.scratch/dev-mcq/` with this seed (default: in-memory).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Write through to `.scratch/dev-mcq/`.
    #[arg(long, default_value_t = false)]
    pub persist: bool,
    /// Generate/cache without starting an interactive session.
    #[arg(long, default_value_t = false)]
    pub print_only: bool,
    /// Fresh experiment identity (cache isolation).
    #[arg(long)]
    pub generation: Option<String>,
}

/// Args for `cadence dev assignment`.
#[derive(Debug, Args)]
pub struct DevAssignmentArgs {
    /// Fixture single-chapter PDF.
    #[arg(long)]
    pub pdf: String,
    /// One-based physical page where the fixture unit starts (front matter
    /// before this is excluded, mirroring `ingest --start-page`).
    #[arg(long, default_value_t = 1)]
    pub start_page: i64,
    /// Persist to `.scratch/dev-assignment/` with this seed (in-memory default).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Write through to `.scratch/dev-assignment/`.
    #[arg(long, default_value_t = false)]
    pub persist: bool,
    /// Generate/cache without opening the editor.
    #[arg(long, default_value_t = false)]
    pub print_only: bool,
}

/// Args for `cadence dev grade`.
#[derive(Debug, Args)]
pub struct DevGradeArgs {
    /// Markdown file with answers.
    #[arg(long)]
    pub answers: String,
    /// Rubric JSON file.
    #[arg(long)]
    pub rubric: String,
}

/// Args for `cadence dev notes`.
#[derive(Debug, Args)]
pub struct DevNotesArgs {
    /// Fixture single-chapter PDF.
    #[arg(long)]
    pub pdf: String,
    /// Optional misconceptions JSON.
    #[arg(long)]
    pub misconceptions: Option<String>,
}

/// Args for `cadence dev schedule`.
#[derive(Debug, Args)]
pub struct DevScheduleArgs {
    /// Deterministic scenario fixture JSON.
    #[arg(long)]
    pub fixture: String,
}

/// Args for `cadence dev llm`.
#[derive(Debug, Args)]
pub struct DevLlmArgs {
    /// Operation label for the durable job log.
    #[arg(long, default_value = "smoke")]
    pub operation: String,
    /// Prompt text (otherwise read from stdin).
    #[arg(long)]
    pub prompt: Option<String>,
    /// Model override (default: free-tier Laguna; paid: zai/glm-5.3-flash).
    #[arg(long)]
    pub model: Option<String>,
}
