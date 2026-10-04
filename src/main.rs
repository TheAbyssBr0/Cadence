//! `cadence` entrypoint: CLI → Core Engine → Persistence.

mod assignment;
mod cli;
mod config;
mod dev_mcq;
mod dispute;
mod domain;
mod engines;
mod error;
mod grading;
mod ingest;
mod llm;
mod mcq;
mod metrics;
mod misconceptions;
mod notes;
mod pdf;
mod review;
mod scheduler;
mod split;
mod store;

use std::io::{IsTerminal, Write as _};
use std::process::Command;

use chrono::NaiveDate;
use clap::Parser;

use cli::{Cli, Commands, DevStage};
use config::Config;
use domain::{Task, TaskStatus, TaskType};
use engines::PdfSource;
use error::{Error, Result};
use scheduler::{
    classify_task, complete_and_advance, count_skipped, ensure_tasks, execution_rank,
    format_chapter_start_prompt, overdue_tasks, project_window, pull_available, pull_next,
    skip_chapter, today_queue, unskip_chapter,
};
use store::{MemoryStore, SqliteStore, Store};

/// Verify `nvim` is installed; it is a hard dependency (§7.2).
///
/// # Errors
///
/// Returns [`Error::MissingDependency`] when `nvim` cannot be executed.
fn require_nvim() -> Result<()> {
    Command::new("nvim")
        .arg("--version")
        .output()
        .map_err(|_| {
            Error::MissingDependency(
                "nvim is required but was not found on PATH; install neovim to continue"
                    .to_string(),
            )
        })
        .map(|_| ())
}

/// Parse a `YYYY-MM-DD` date.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on malformed dates.
fn parse_date(text: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .map_err(|e| Error::InvalidInput(format!("bad date '{text}': {e}")))
}

/// Today's date (injected boundary: only `main` and `dev schedule` read the
/// clock; engines take dates as parameters).
fn today_date() -> NaiveDate {
    chrono::Local::now().date_naive()
}

/// Open the production store (creates `data_dir` on demand).
fn open_production_store(config: &Config) -> Result<SqliteStore> {
    SqliteStore::open(&config.db_path(), &config.lock_path())
}

/// Top up scheduler state for every registered book (§4–§6): unlocks, opens
/// stranded reading stages, and creates missing tasks due `today`.
/// Returns the total number of tasks created.
fn ensure_all_books(store: &mut SqliteStore, today: NaiveDate, today_str: &str) -> Result<usize> {
    let books = store.list_books()?;
    let mut total = 0_usize;
    for book in &books {
        total = total.saturating_add(ensure_tasks(store, book.id, today, today_str)?.len());
    }
    Ok(total)
}

/// Ask a yes/no question on an interactive terminal. Returns `None` when
/// stdin is not a terminal (piped/CI): callers proceed without prompting.
fn ask_terminal(prompt: &str) -> Result<Option<bool>> {
    if !std::io::stdin().is_terminal() {
        return Ok(None);
    }
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    let bytes = std::io::stdin().read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    let answer = line.trim().to_lowercase();
    Ok(Some(answer.is_empty() || answer == "y" || answer == "yes"))
}

/// Whether a continue-prompt answer chains forward: only an explicit
/// `y`/`yes` — Enter, `n`, and anything else exits.
#[must_use]
fn parse_continue_answer(line: &str) -> bool {
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Ask the daily-loop continue prompt: only an explicit `y`/`yes` chains
/// into the next task — Enter, `n`, or EOF exits cleanly with state saved,
/// so the loop never cascades through the pipeline on lazy Enters. Piped
/// (non-terminal) stdin continues, preserving scripted runs.
///
/// # Errors
///
/// Propagates [`Error::Io`] on terminal read failures.
fn ask_continue(prompt: &str) -> Result<bool> {
    use std::io::Write as _;
    if !std::io::stdin().is_terminal() {
        return Ok(true);
    }
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    let bytes = std::io::stdin().read_line(&mut line)?;
    if bytes == 0 {
        return Ok(false);
    }
    Ok(parse_continue_answer(&line))
}

/// Outcome of the pre-chapter gate (§4.1): proceed into the chapter, skip it
/// outright (explicit `s` only), or leave the loop with state saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChapterGate {
    Proceed,
    Skip,
    Exit,
}

/// Parse one gate answer: `None` means unrecognized (the caller re-prompts).
/// Only an explicit `s` skips — `n` never does (§15: anything but `Y`
/// exits cleanly).
fn parse_gate_answer(line: &str) -> Option<ChapterGate> {
    match line.trim().to_lowercase().as_str() {
        "" | "y" | "yes" => Some(ChapterGate::Proceed),
        "s" | "skip" => Some(ChapterGate::Skip),
        "n" | "no" | "q" | "quit" | "exit" => Some(ChapterGate::Exit),
        _ => None,
    }
}

/// Ask the pre-chapter gate on an interactive terminal. Non-terminal stdin
/// (piped/CI) proceeds without prompting; EOF exits with state saved.
///
/// # Errors
///
/// Propagates [`Error::Io`] on terminal read failures.
fn ask_chapter_gate(prompt: &str) -> Result<ChapterGate> {
    use std::io::Write as _;
    if !std::io::stdin().is_terminal() {
        return Ok(ChapterGate::Proceed);
    }
    loop {
        print!("{prompt}");
        std::io::stdout().flush()?;
        let mut line = String::new();
        let bytes = std::io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            return Ok(ChapterGate::Exit);
        }
        if let Some(answer) = parse_gate_answer(&line) {
            return Ok(answer);
        }
        println!("Please answer Y (proceed), s (skip chapter), or n (exit).");
    }
}

/// Pre-chapter gate (§4.1): when the daily loop reaches a new chapter, offer
/// proceed / skip / exit. Skipping deletes pending tasks (completed work
/// stays as an audit trail); exiting saves state for resume. Returns the
/// gate outcome for the loop to act on.
fn gate_chapter_start(
    store: &mut SqliteStore,
    chapter: &domain::Chapter,
    today_str: &str,
) -> Result<ChapterGate> {
    let prompt = format_chapter_start_prompt(&chapter.title, chapter.start_page, chapter.end_page);
    match ask_chapter_gate(&prompt)? {
        ChapterGate::Skip => {
            skip_chapter(store, chapter.id, today_str)?;
            println!("Skipped chapter {} ('{}').", chapter.id, chapter.title);
            Ok(ChapterGate::Skip)
        }
        outcome => Ok(outcome),
    }
}

/// List chapters across all books with status, pages, and `~ likely meta`
/// presentation flags (§4.1). Never auto-skips.
/// Render the skip-listing: every book with its chapters, pages, attempts,
/// and likely-meta flags, plus usage. Pure text (the `print_*` twin writes
/// it).
///
/// # Errors
///
/// Propagates [`Error::Store`] from listing reads.
fn format_skip_listing(store: &SqliteStore) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let books = store.list_books()?;
    if books.is_empty() {
        let _ = writeln!(out, "No books ingested yet.");
        return Ok(out);
    }
    for book in &books {
        let _ = writeln!(out, "Book {} — {}:", book.id, book.title);
        for chapter in store.list_chapters(book.id)? {
            let pages = domain::unit_page_count(chapter.start_page, chapter.end_page).unwrap_or(0);
            let flag = if domain::likely_meta(&chapter.title, chapter.level, pages) {
                "~ likely meta"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  [{}] {} '{}' (pages {}–{}, attempt {}) {flag}",
                chapter.status.as_str(),
                chapter.id,
                chapter.title,
                chapter.start_page,
                chapter.end_page,
                chapter.attempt_no
            );
        }
    }
    let _ = writeln!(out, "Usage: cadence skip <id> | cadence unskip <id>");
    Ok(out)
}

fn print_skip_listing(store: &SqliteStore) -> Result<()> {
    print!("{}", format_skip_listing(store)?);
    Ok(())
}

/// `cadence skip [id]`: list without args, skip one chapter with an id.
fn run_skip(store: &mut SqliteStore, id: Option<i64>, today_str: &str) -> Result<()> {
    let Some(chapter_id) = id else {
        return print_skip_listing(store);
    };
    let removed = skip_chapter(store, chapter_id, today_str)?;
    println!("Skipped chapter {chapter_id} ({removed} pending task(s) removed).");
    Ok(())
}

/// `cadence unskip <id>`: restart a skipped chapter fresh on a new attempt.
fn run_unskip(
    store: &mut SqliteStore,
    chapter_id: i64,
    today: NaiveDate,
    today_str: &str,
) -> Result<()> {
    let task = unskip_chapter(store, chapter_id, today, today_str)?;
    println!(
        "Unskipped chapter {chapter_id} (fresh pretest scheduled for {}).",
        task.scheduled_for
    );
    Ok(())
}

/// Reading stage (§4): present the chapter's pages and confirm completion.
/// A `Y` answer completes the `READ` task via the scheduler (retest lands
/// tomorrow); anything else leaves state untouched for resume. Returns
/// whether the task completed.
fn run_reading_task(store: &mut SqliteStore, task: &Task, today_str: &str) -> Result<bool> {
    let chapter = store.get_chapter(task.chapter_id)?;
    println!(
        "Read '{}' (pages {}–{}): {}",
        chapter.title, chapter.start_page, chapter.end_page, chapter.file_path
    );
    match ask_terminal("Finished reading? [Y/n] > ")? {
        None => {
            println!("(non-interactive: reading left pending — rerun to complete it.)");
            return Ok(false);
        }
        Some(false) => {
            println!("Reading left pending — next launch resumes here.");
            return Ok(false);
        }
        Some(true) => {}
    }
    let today = parse_date(today_str)?;
    match complete_and_advance(store, task.id, today, today_str)? {
        Some(next) => println!(
            "Task complete: Read '{}'. Scheduled follow-up: {} (due {}).",
            chapter.title,
            next.task_type.as_str(),
            next.scheduled_for
        ),
        None => println!("Task complete: Read '{}'.", chapter.title),
    }
    Ok(true)
}

/// Generate-or-resume the stored MCQ set for a production chapter/attempt
/// (§7.1): reuse validated rows when present, otherwise run one
/// `complete_cached` generation (validated before caching) and persist it.
/// Without connectivity and without stored rows there is nothing runnable —
/// the LLM error propagates with state intact (§16: cached tasks still
/// completable, rerun resumes).
fn ensure_production_items(
    store: &mut SqliteStore,
    chapter: &domain::Chapter,
    unit: &engines::UnitText,
    phase: mcq::McqPhase,
    today_str: &str,
) -> Result<Vec<store::McqItem>> {
    let items = store.list_mcq_items(chapter.id, phase.as_str(), chapter.attempt_no)?;
    if !items.is_empty() {
        println!(
            "Resumed {} stored question(s) for {} ('{}').",
            items.len(),
            phase.as_str(),
            chapter.title
        );
        return Ok(items);
    }
    println!(
        "Generating {} questions for '{}' (pages {}–{}) …",
        phase.as_str(),
        chapter.title,
        unit.page_start,
        unit.page_end
    );
    let count = dev_mcq::DEV_MCQ_COUNT;
    let prompt = match phase {
        mcq::McqPhase::Pretest => mcq::build_pretest_prompt(unit, count),
        mcq::McqPhase::Retest => mcq::build_retest_prompt(unit, count),
        mcq::McqPhase::Review => {
            return Err(Error::InvalidInput(
                "review sets generate through ensure_review_items (they need misconception targets)".to_string(),
            ));
        }
    };
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = dev_mcq::DEV_MCQ_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &mcq::mcq_response_schema(),
        "mcq_set",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = mcq::mcq_params_json(count, phase);
    let source_hash = mcq::source_hash_for(&unit.text);
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today_str,
    };
    let operation = phase.as_str().to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let unit_ref = unit;
    let validate =
        |text: &str| mcq::validate_mcq_set(text, unit_ref).map(|_| text.trim().to_string());
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    println!(
        "Generated via LLM (cache: {}, transport sends: {}).",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let validated = mcq::validate_mcq_set(&result.text, unit)?;
    let new_rows = dev_mcq::to_new_items_for(chapter.id, phase, &validated, chapter.attempt_no)?;
    store.save_mcq_items(&new_rows)
}

/// Execute one due scheduler `PRETEST`/`RETEST` task against the production
/// store: load the chapter corpus, generate-or-resume its question set, run
/// the interactive session, and — only when every item is answered — advance
/// the pipeline with the §15 checkpoint footer. Returns whether the task
/// completed (early exits and offline waits return `false` with state saved).
fn run_production_mcq(
    store: &mut SqliteStore,
    task_id: i64,
    today: NaiveDate,
    today_str: &str,
) -> Result<bool> {
    let task = store.get_task(task_id)?;
    if task.status == TaskStatus::Done {
        println!("Task already completed: {} — no state changed.", task.id);
        return Ok(false);
    }
    let Some(phase) = mcq::McqPhase::for_task(task.task_type) else {
        return Err(Error::InvalidInput(format!(
            "task {} is {} — MCQ sessions only run pretest/retest",
            task.id,
            task.task_type.as_str()
        )));
    };
    let chapter = store.get_chapter(task.chapter_id)?;
    let unit = ingest::load_unit_text(std::path::Path::new(&chapter.file_path))?;
    let items = ensure_production_items(store, &chapter, &unit, phase, today_str)?;
    println!(
        "\n{}: '{}' — {} question(s). Closed book, no notes.",
        phase.as_str(),
        chapter.title,
        items.len()
    );
    let summary = run_mcq_session(
        store,
        &items,
        &unit,
        phase,
        chapter.id,
        chapter.attempt_no,
        None,
    )?;
    if !summary.completed {
        return Ok(false);
    }
    let next = complete_and_advance(store, task.id, today, today_str)?;
    println!(
        "\nTask complete: {} — {} MCQ (Score: {}/{})",
        chapter.title,
        phase.as_str(),
        summary.correct,
        items.len()
    );
    println!(
        "Misconceptions logged: {} item(s) added.",
        summary.misconceptions
    );
    match next {
        Some(followup) => println!(
            "Scheduled follow-up: {} (due {}).",
            followup.task_type.as_str(),
            followup.scheduled_for
        ),
        None => println!("Chapter '{}' complete.", chapter.title),
    }
    Ok(true)
}

/// Join stored assignment parts into grading/display text (§7.2): parts were
/// validated non-empty at generation time, so an empty result only means a
/// corrupt row — the grader treats it as rubric-only (see
/// `build_grading_prompt`), never as a silent pass.
#[must_use]
fn assignment_question_text(parts_json: &str) -> String {
    let parts: Vec<String> = serde_json::from_str(parts_json).unwrap_or_default();
    parts.join("\n")
}

/// Serialize per-criterion judgments into the `grades` row payload (§17).
/// The shape mirrors the validator output (`name`, `score`, `max_score`,
/// `comment`); serialization of an in-memory struct cannot fail in practice,
/// and `"[]"` keeps a corrupt edge visible instead of panicking.
#[must_use]
fn criteria_results_json(results: &[grading::CriterionResult]) -> String {
    let values: Vec<serde_json::Value> = results
        .iter()
        .map(|criterion| {
            serde_json::json!({
                "name": criterion.name,
                "score": criterion.score,
                "max_score": criterion.max_score,
                "comment": criterion.comment,
            })
        })
        .collect();
    serde_json::to_string(&values).unwrap_or_else(|_| "[]".to_string())
}

/// Convert a validated grade into its store row (§7.3/§10): the awarded
/// `score` is the original forever — disputes write `final_score` and never
/// overwrite it (§9 audit trail).
#[must_use]
fn grade_to_new(
    question_id: i64,
    grade: &grading::Grade,
    max_score: i64,
    today_str: &str,
) -> store::NewGrade {
    store::NewGrade {
        question_id,
        score: grade.score,
        max_score,
        classification: grade.classification.as_str().to_string(),
        criteria_results_json: criteria_results_json(&grade.criteria_results),
        feedback: grade.feedback.clone(),
        grader_version: llm::PROMPT_VERSION.to_string(),
        created_at: today_str.to_string(),
    }
}

/// Generate-or-resume the stored assignment set for a production
/// chapter/attempt (§7.2): reuse validated rows when present, otherwise run
/// one `complete_cached` generation (validated before caching, re-probing
/// open misconceptions) and persist it. Without connectivity and without
/// stored rows the LLM error propagates with state intact (§16).
fn ensure_production_assignment_questions(
    store: &mut SqliteStore,
    chapter: &domain::Chapter,
    unit: &engines::UnitText,
    today_str: &str,
) -> Result<Vec<store::AssignmentQuestion>> {
    let items = store.list_assignment_questions(chapter.id, chapter.attempt_no)?;
    if !items.is_empty() {
        println!(
            "Resumed {} stored assignment question(s) for '{}'.",
            items.len(),
            chapter.title
        );
        return Ok(items);
    }
    println!(
        "Generating assignment for '{}' (pages {}–{}) …",
        chapter.title, unit.page_start, unit.page_end
    );
    let open = open_misconceptions(store, chapter.id)?;
    let prompt = assignment::build_assignment_prompt(unit, &open);
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = assignment::ASSIGNMENT_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &assignment::assignment_response_schema(),
        "assignment_set",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let hash = assignment::misconceptions_hash(&open);
    let params = assignment::assignment_params_json(&hash);
    let source_hash = mcq::source_hash_for(&unit.text);
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today_str,
    };
    let operation = "assignment".to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let unit_ref = unit;
    let open_ref = &open;
    let validate = |text: &str| {
        assignment::validate_assignment_set(text, unit_ref, open_ref)
            .map(|_| text.trim().to_string())
    };
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    println!(
        "Generated via LLM (cache: {}, transport sends: {}).",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let validated = assignment::validate_assignment_set(&result.text, unit, &open)?;
    let new_rows = assignment::to_new_questions(chapter.id, chapter.attempt_no, &validated)?;
    store.save_assignment_questions(&new_rows)
}

/// Build misconception concept/description/evidence for a wrong assignment
/// answer (§12): the question (truncated) as concept, the verdict plus
/// feedback as description, and an answer excerpt as evidence. Pure and
/// deterministic — the caller persists the triple.
#[must_use]
fn assignment_misconception_texts(
    position: i64,
    question_text: &str,
    classification: &str,
    score: i64,
    max_score: i64,
    feedback: &str,
    answer: &str,
) -> (String, String, String) {
    fn truncate(text: &str, chars: usize) -> String {
        let count = text.chars().count();
        if count <= chars {
            text.trim().to_string()
        } else {
            format!("{}...", text.chars().take(chars).collect::<String>())
        }
    }
    let number = position.saturating_add(1);
    let concept = format!("Assignment Q{number}: {}", truncate(question_text, 100));
    let description = format!("Graded {classification} {score}/{max_score}: {feedback}");
    let evidence = format!("Answer excerpt: {}", truncate(answer, 200));
    (concept, description, evidence)
}

/// Apply the §12 lifecycle for one freshly saved grade: wrong verdicts log a
/// new `ASSIGNMENT` row and push re-probed targets down substantially;
/// full-credit verdicts push re-probed targets up substantially; blanks and
/// defective questions carry no signal. Only `ACTIVE`/`IMPROVING` targets
/// move — resolved rows never reopen here. Grade first, lifecycle second:
/// the grade is the source of truth, and a crash between the two only skips
/// a nudge that later outcomes correct.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn apply_grade_lifecycle(
    store: &mut SqliteStore,
    question: &store::AssignmentQuestion,
    grade: &grading::Grade,
    max_score: i64,
    answer_text: &str,
    today_str: &str,
) -> Result<()> {
    let number = question.position.saturating_add(1);
    let targets: Vec<i64> = serde_json::from_str(&question.target_misconception_ids)
        .unwrap_or_else(|_| {
            println!(
                "Question {number} has corrupt re-probe targets — skipping confidence updates."
            );
            Vec::new()
        });
    let chapter_id = question.chapter_id;
    if misconceptions::should_log_assignment_misconception(grade.classification) {
        let question_text = assignment_question_text(&question.parts_json);
        let (concept, description, evidence) = assignment_misconception_texts(
            question.position,
            &question_text,
            grade.classification.as_str(),
            grade.score,
            max_score,
            &grade.feedback,
            answer_text,
        );
        store.create_misconception(
            chapter_id,
            &concept,
            &description,
            &evidence,
            "ASSIGNMENT",
            today_str,
        )?;
        println!("Misconception logged: {concept}");
        nudge_targeted_misconceptions(store, chapter_id, &targets, false, today_str)?;
    } else if misconceptions::is_assignment_correct(grade.classification) {
        nudge_targeted_misconceptions(store, chapter_id, &targets, true, today_str)?;
    }
    Ok(())
}

/// Push one lifecycle step onto re-probed target rows (§12: assignment-sized
/// deltas). `correct` selects the direction. Unknown ids are reported and
/// skipped — a stale target never fails grading.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn nudge_targeted_misconceptions(
    store: &mut SqliteStore,
    chapter_id: i64,
    target_ids: &[i64],
    correct: bool,
    today_str: &str,
) -> Result<()> {
    if target_ids.is_empty() {
        return Ok(());
    }
    let rows = store.list_misconceptions(chapter_id)?;
    for target in target_ids {
        let Some(row) = rows.iter().find(|row| row.id == *target) else {
            println!("Re-probed misconception {target} is gone — skipping its confidence update.");
            continue;
        };
        if row.status != "ACTIVE" && row.status != "IMPROVING" {
            continue;
        }
        let step = misconceptions::apply_outcome(row.confidence, &row.status, correct, true);
        let resolved_at = if step.just_resolved {
            Some(today_str)
        } else {
            row.resolved_at.as_deref()
        };
        store.update_misconception(row.id, step.confidence, step.status, today_str, resolved_at)?;
        if step.just_resolved {
            println!("Misconception resolved: {}", row.concept_description);
        } else {
            let direction = if correct { "improving" } else { "worsening" };
            println!(
                "Misconception {direction}: {} (confidence {:.2} → {:.2})",
                row.concept_description, row.confidence, step.confidence
            );
        }
    }
    Ok(())
}

/// Grade every answered production assignment question against its frozen
/// rubric (§7.3/§10) via `complete_cached` (validated before caching) and
/// persist each verdict with `save_grade`. Questions with a stored grade are
/// skipped so reruns resume; `QUESTION_DEFECTIVE` awards full credit and is
/// flagged, never penalizing the user. Returns graded + resumed totals.
/// Grade one unanswered question end to end: rubric → prompt → cached LLM
/// call → persist → lifecycle. Returns `(earned, possible)` score deltas.
///
/// # Errors
///
/// Propagates LLM, validation, and [`Error::Store`] failures.
fn grade_one_question(
    store: &mut SqliteStore,
    question: &store::AssignmentQuestion,
    answer_text: &str,
    questions_len: usize,
    today_str: &str,
) -> Result<(i64, i64)> {
    let rubric = grading::parse_rubric_json(&question.rubric_json)?;
    let question_text = assignment_question_text(&question.parts_json);
    let prompt = grading::build_grading_prompt(&question_text, &rubric, answer_text);
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = grading::GRADING_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &grading::grade_response_schema(&rubric),
        "grade",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = grading::grade_params_json_for(&rubric);
    let source_hash =
        grading::grade_source_hash(&question_text, &question.rubric_json, answer_text);
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today_str,
    };
    let operation = "grade".to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let rubric_ref = &rubric;
    let validate =
        |text: &str| grading::validate_grade(text, rubric_ref).map(|_| text.trim().to_string());
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    let grade = grading::validate_grade(&result.text, &rubric)?;
    let saved = store.save_grade(&grade_to_new(
        question.id,
        &grade,
        rubric.max_score,
        today_str,
    ))?;
    apply_grade_lifecycle(
        store,
        question,
        &grade,
        rubric.max_score,
        answer_text,
        today_str,
    )?;
    println!(
        "Graded {}/{}: {} — {}/{}",
        question.position.saturating_add(1),
        questions_len,
        grade.classification.as_str(),
        grade.score,
        rubric.max_score
    );
    println!("Feedback: {}", grade.feedback);
    if grade.classification == grading::GradeClass::QuestionDefective {
        println!(
            "(QUESTION_DEFECTIVE never penalizes the user: full credit, item flagged for replacement.)"
        );
    }
    Ok((saved.score, saved.max_score))
}

fn run_production_grading(
    store: &mut SqliteStore,
    questions: &[store::AssignmentQuestion],
    today_str: &str,
) -> Result<(usize, i64, i64)> {
    let mut graded = 0_usize;
    let mut earned = 0_i64;
    let mut possible = 0_i64;
    for question in questions {
        let responses = store.list_assignment_responses(question.id)?;
        let Some(latest) = responses.last() else {
            return Err(Error::InvalidInput(format!(
                "question {} has no saved answer — complete every answer before grading",
                question.id
            )));
        };
        if !store.list_grades_for_question(question.id)?.is_empty() {
            let existing = store.list_grades_for_question(question.id)?;
            if let Some(first) = existing.first() {
                earned = earned.saturating_add(first.final_score.unwrap_or(first.score));
                possible = possible.saturating_add(first.max_score);
                graded = graded.saturating_add(1);
            }
            continue;
        }
        let (delta_earned, delta_possible) = grade_one_question(
            store,
            question,
            &latest.answer_text,
            questions.len(),
            today_str,
        )?;
        earned = earned.saturating_add(delta_earned);
        possible = possible.saturating_add(delta_possible);
        graded = graded.saturating_add(1);
    }
    Ok((graded, earned, possible))
}

/// Execute one due scheduler `ASSIGNMENT_WRITE` task against the production
/// store: generate-or-resume the set (re-probing open misconceptions),
/// collect closed-book answers in `nvim`, grade each answer against its
/// frozen rubric, and — only when every question is answered and graded —
/// advance the pipeline with the §15 checkpoint footer. Returns whether the
/// task completed (early exits return `false` with state saved; rerun
/// resumes answers, then grades).
fn run_production_assignment(
    store: &mut SqliteStore,
    task_id: i64,
    today: NaiveDate,
    today_str: &str,
) -> Result<bool> {
    let task = store.get_task(task_id)?;
    if task.status == TaskStatus::Done {
        println!("Task already completed: {} — no state changed.", task.id);
        return Ok(false);
    }
    if task.task_type != TaskType::AssignmentWrite {
        return Err(Error::InvalidInput(format!(
            "task {} is {} — assignment sessions only run assignment_write",
            task.id,
            task.task_type.as_str()
        )));
    }
    let chapter = store.get_chapter(task.chapter_id)?;
    let unit = ingest::load_unit_text(std::path::Path::new(&chapter.file_path))?;
    let items = ensure_production_assignment_questions(store, &chapter, &unit, today_str)?;
    println!(
        "\nAssignment: '{}' — {} question(s). Closed book, no notes.",
        chapter.title,
        items.len()
    );
    print_assignment_set(&items);
    if !std::io::stdin().is_terminal() {
        println!("(non-interactive: assignment left pending — rerun to answer it.)");
        return Ok(false);
    }
    require_nvim()?;
    if !run_assignment_answers(store, &items, chapter.attempt_no)? {
        return Ok(false);
    }
    let (graded, earned, possible) = run_production_grading(store, &items, today_str)?;
    let next = complete_and_advance(store, task.id, today, today_str)?;
    println!(
        "\nTask complete: {} — Assignment graded ({graded}/{}) (Score: {earned}/{possible})",
        chapter.title,
        items.len()
    );
    match next {
        Some(followup) => println!(
            "Scheduled follow-up: {} (due {}).",
            followup.task_type.as_str(),
            followup.scheduled_for
        ),
        None => println!("Chapter '{}' complete.", chapter.title),
    }
    Ok(true)
}

/// Collect every graded answer for a chapter/attempt as notes input (§11
/// items 4–5): the frozen verdict plus feedback per question. Questions
/// without a stored grade are reported and skipped — notes stay generatable
/// even when one grade is missing, and the gap is visible, never silent.
fn grade_summaries_for(
    store: &SqliteStore,
    chapter_id: i64,
    attempt_no: i64,
) -> Result<Vec<notes::GradeSummary>> {
    let mut out = Vec::new();
    for question in store.list_assignment_questions(chapter_id, attempt_no)? {
        let grades = store.list_grades_for_question(question.id)?;
        let Some(latest) = grades.last() else {
            println!(
                "Question {} has no grade yet — leaving it out of the notes.",
                question.position.saturating_add(1)
            );
            continue;
        };
        out.push(notes::GradeSummary {
            question: assignment_question_text(&question.parts_json),
            score: latest.final_score.unwrap_or(latest.score),
            max_score: latest.max_score,
            classification: latest.classification.clone(),
            feedback: latest.feedback.clone(),
        });
    }
    Ok(out)
}

/// Map every tracked misconception onto notes input (§11 item 3, §12): open
/// rows are highlighted with corrections, resolved rows are noted as cleared.
/// All rows pass through so future lifecycle transitions flow without an
/// engine change.
fn misconception_items_for(rows: &[store::Misconception]) -> Vec<notes::MisconceptionItem> {
    rows.iter()
        .map(|row| notes::MisconceptionItem {
            concept: row.concept_description.clone(),
            evidence: row.evidence.clone(),
            status: row.status.clone(),
        })
        .collect()
}

/// MCQ fraction for one chapter/phase/attempt (§13 performance): correct over
/// answered items (latest response per item; unanswered items are excluded,
/// never counted as wrong).
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn mcq_fraction(
    store: &dyn Store,
    chapter_id: i64,
    phase: &str,
    attempt_no: i64,
) -> Result<(u64, u64)> {
    let mut correct = 0_u64;
    let mut answered = 0_u64;
    for item in store.list_mcq_items(chapter_id, phase, attempt_no)? {
        let responses = store.list_mcq_responses(item.id)?;
        if let Some(latest) = responses.last() {
            answered = answered.saturating_add(1);
            if latest.is_correct {
                correct = correct.saturating_add(1);
            }
        }
    }
    Ok((correct, answered))
}

/// Assignment fraction for one chapter/attempt (§13 performance): effective
/// points (post-dispute `final_score`, else `score`) over rubric points,
/// latest grade per question. Ungraded questions contribute nothing.
///
/// # Errors
///
/// Returns [`Error::Store`] on corrupt negative stored scores, and
/// propagates backend failures.
fn assignment_fraction(store: &dyn Store, chapter_id: i64, attempt_no: i64) -> Result<(u64, u64)> {
    let mut earned = 0_u64;
    let mut possible = 0_u64;
    for question in store.list_assignment_questions(chapter_id, attempt_no)? {
        let grades = store.list_grades_for_question(question.id)?;
        if let Some(latest) = grades.last() {
            let effective = latest.final_score.unwrap_or(latest.score);
            let points = u64::try_from(effective)
                .map_err(|e| Error::Store(format!("grade {} has corrupt score: {e}", latest.id)))?;
            let total = u64::try_from(latest.max_score).map_err(|e| {
                Error::Store(format!("grade {} has corrupt max_score: {e}", latest.id))
            })?;
            earned = earned.saturating_add(points);
            possible = possible.saturating_add(total);
        }
    }
    Ok((earned, possible))
}

/// Open vs. resolved tally for one chapter (§13 misconceptions): `ACTIVE` +
/// `IMPROVING` vs. `RESOLVED`. `DISPUTED` rows are purged evidence and count
/// toward neither side.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn misconception_tally(store: &dyn Store, chapter_id: i64) -> Result<(u64, u64)> {
    let mut active = 0_u64;
    let mut resolved = 0_u64;
    for row in store.list_misconceptions(chapter_id)? {
        if row.status == "ACTIVE" || row.status == "IMPROVING" {
            active = active.saturating_add(1);
        } else if row.status == "RESOLVED" {
            resolved = resolved.saturating_add(1);
        }
    }
    Ok((active, resolved))
}

/// Evidence snapshot for one live (non-skipped) chapter on its current
/// attempt (§13): MCQ fractions, effective assignment scores, and the
/// misconception tally, plus the completion date for pace (latest live `DONE`
/// task, only when the chapter itself is `Completed`).
///
/// # Errors
///
/// Propagates corrupt-score and backend failures from the helpers above.
fn chapter_evidence(
    store: &dyn Store,
    chapter: &domain::Chapter,
) -> Result<metrics::ChapterEvidence> {
    let raw_pages = domain::unit_page_count(chapter.start_page, chapter.end_page).unwrap_or(0);
    let pages = u64::try_from(raw_pages)
        .map_err(|e| Error::Store(format!("chapter {} has corrupt pages: {e}", chapter.id)))?;
    let chapter_id = chapter.id;
    let attempt_no = chapter.attempt_no;
    let live_tasks: Vec<Task> = store
        .list_tasks()?
        .into_iter()
        .filter(|t| t.chapter_id == chapter_id && t.attempt_no == attempt_no)
        .collect();
    let completed_on = if chapter.status == domain::ChapterStatus::Completed {
        live_tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Done)
            .filter_map(|t| {
                t.completed_at
                    .as_deref()
                    .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            })
            .max()
    } else {
        None
    };
    let (pretest_correct, pretest_answered) =
        mcq_fraction(store, chapter.id, "pretest", chapter.attempt_no)?;
    let (retest_correct, retest_answered) =
        mcq_fraction(store, chapter.id, "retest", chapter.attempt_no)?;
    let (assignment_earned, assignment_possible) =
        assignment_fraction(store, chapter.id, chapter.attempt_no)?;
    let (active_misconceptions, resolved_misconceptions) = misconception_tally(store, chapter.id)?;
    Ok(metrics::ChapterEvidence {
        title: chapter.title.clone(),
        status: chapter.status,
        pages,
        completed_on,
        pretest_correct,
        pretest_answered,
        retest_correct,
        retest_answered,
        assignment_earned,
        assignment_possible,
        active_misconceptions,
        resolved_misconceptions,
    })
}

/// Generate-or-resume the stored notes for a production chapter/attempt
/// (§11): reuse the validated document when present, otherwise run one
/// `complete_cached` synthesis (validated before caching) over the chapter
/// text plus misconception and grading evidence, and persist it. Without
/// connectivity and without stored notes the LLM error propagates with state
/// intact (§16).
fn ensure_production_notes(
    store: &mut SqliteStore,
    chapter: &domain::Chapter,
    unit: &engines::UnitText,
    today_str: &str,
) -> Result<store::Note> {
    let existing = store.list_notes(chapter.id, chapter.attempt_no)?;
    if let Some(first) = existing.into_iter().next() {
        println!(
            "Resumed stored notes for '{}' ({} chars).",
            chapter.title,
            first.content_markdown.chars().count()
        );
        return Ok(first);
    }
    println!(
        "Generating notes for '{}' (pages {}–{}) …",
        chapter.title, unit.page_start, unit.page_end
    );
    let misconceptions = misconception_items_for(&store.list_misconceptions(chapter.id)?);
    let grades = grade_summaries_for(store, chapter.id, chapter.attempt_no)?;
    if grades.is_empty() {
        return Err(Error::InvalidInput(format!(
            "chapter {} has no graded answers — grade the assignment before synthesizing notes",
            chapter.id
        )));
    }
    let prompt = notes::build_notes_prompt(unit, &misconceptions, &grades);
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = notes::NOTES_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &notes::notes_response_schema(),
        "notes",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = notes::notes_params_json();
    let source_hash = notes::notes_source_hash_for(&unit.text, &misconceptions, &grades);
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today_str,
    };
    let operation = "notes".to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let validate = |text: &str| notes::validate_notes(text).map(|_| text.trim().to_string());
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    println!(
        "Generated via LLM (cache: {}, transport sends: {}).",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let validated = notes::validate_notes(&result.text)?;
    let row = notes::to_new_note(chapter.id, chapter.attempt_no, &validated, today_str);
    store.save_note(&row)
}

/// Execute one due scheduler `NOTES` task against the production store:
/// synthesize the six-section personalized notes from chapter text plus
/// misconception and grading evidence, print them, and advance the pipeline
/// (chapter `COMPLETED`, next chapter unlocked). Returns whether the task
/// completed.
fn run_production_notes(
    store: &mut SqliteStore,
    task_id: i64,
    today: NaiveDate,
    today_str: &str,
) -> Result<bool> {
    let task = store.get_task(task_id)?;
    if task.status == TaskStatus::Done {
        println!("Task already completed: {} — no state changed.", task.id);
        return Ok(false);
    }
    if task.task_type != TaskType::Notes {
        return Err(Error::InvalidInput(format!(
            "task {} is {} — notes sessions only run notes",
            task.id,
            task.task_type.as_str()
        )));
    }
    let chapter = store.get_chapter(task.chapter_id)?;
    let unit = ingest::load_unit_text(std::path::Path::new(&chapter.file_path))?;
    let note = ensure_production_notes(store, &chapter, &unit, today_str)?;
    println!("\nChapter notes: '{}'\n", chapter.title);
    println!("{}", note.content_markdown);
    let next = complete_and_advance(store, task.id, today, today_str)?;
    println!("\nTask complete: {} — Notes synthesized.", chapter.title);
    match next {
        Some(followup) => println!(
            "Scheduled follow-up: {} (due {}).",
            followup.task_type.as_str(),
            followup.scheduled_for
        ),
        None => println!("Chapter '{}' complete.", chapter.title),
    }
    Ok(true)
}

/// Default entrypoint: today's scheduled loop (§15 checkpoint & resume).
/// Each pass tops up scheduler state, runs the head due task through its
/// stage, then offers the actual next due task (or names the nearest future
/// one when the queue clears) — only an explicit `y` chains forward; Enter
/// exits. The footer on a completed task only describes its scheduled
/// follow-up, never what the loop does next. A clear queue prints the same
/// day-complete footer whether the loop just drained it or the run started
/// with it clear, so reruns hold steady until an explicit `cadence pull`.
/// `<Enter>` (or `Ctrl-C`/`Ctrl-D`) exits cleanly with full persistence;
/// relaunch resumes exactly here.
fn run_daily_loop(store: &mut SqliteStore, today: NaiveDate, today_str: &str) -> Result<()> {
    loop {
        let _ = ensure_all_books(store, today, today_str)?;
        let queue = today_queue(&store.list_tasks()?, today);
        let Some(head) = queue.first() else {
            print_day_complete(store, today)?;
            return Ok(());
        };
        let task = head.clone();
        // Pre-chapter gate (§4.1): fresh pretests offer proceed/skip/exit.
        if task.task_type == TaskType::Pretest {
            let chapter = store.get_chapter(task.chapter_id)?;
            if chapter.status == domain::ChapterStatus::PretestReady {
                match gate_chapter_start(store, &chapter, today_str)? {
                    ChapterGate::Proceed => {}
                    ChapterGate::Skip => continue,
                    ChapterGate::Exit => {
                        println!("Exiting — state saved; rerun to resume.");
                        return Ok(());
                    }
                }
            }
        }
        let progressed = match task.task_type {
            TaskType::Pretest | TaskType::Retest => {
                run_production_mcq(store, task.id, today, today_str)?
            }
            TaskType::Read => run_reading_task(store, &task, today_str)?,
            TaskType::AssignmentWrite => {
                run_production_assignment(store, task.id, today, today_str)?
            }
            TaskType::Notes => run_production_notes(store, task.id, today, today_str)?,
        };
        if !progressed {
            return Ok(());
        }
        // Top up before prompting so the offer names real work: same-chapter
        // follow-ups (e.g. pretest → read) are executable now, while the next
        // chapter's first pretest lands tomorrow (pullable via `cadence pull`).
        let _ = ensure_all_books(store, today, today_str)?;
        let (due, _) = due_and_future(&store.list_tasks()?, today);
        if let Some(next) = due.first() {
            let what = describe_task(store, next)?;
            if !ask_continue(&format!("Next up: {what}. Continue? [y/N] > "))? {
                println!("Exiting — state saved; rerun to resume.");
                return Ok(());
            }
        } else {
            print_day_complete(store, today)?;
            return Ok(());
        }
    }
}

/// Day-complete footer, shared by the drained-queue and started-clear paths:
/// names the nearest future task when one exists, always points at
/// `cadence pull` for more work today, then exits with state saved.
///
/// # Errors
///
/// Returns [`Error::NotFound`] when the nearest future task's chapter is
/// missing (corrupt store).
/// Render the day-complete summary: next scheduled task (if any) plus the
/// pull hint. Pure text (the `print_*` twin writes it).
///
/// # Errors
///
/// Returns [`Error::NotFound`] when the chapter is missing (corrupt store).
fn format_day_complete(store: &SqliteStore, today: NaiveDate) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let (_, future) = due_and_future(&store.list_tasks()?, today);
    match future.first() {
        Some(next) => {
            let what = describe_task(store, next)?;
            let _ = writeln!(out, "All tasks complete today. Next scheduled: {what}.");
        }
        None => {
            let _ = writeln!(out, "All tasks complete today — nothing scheduled ahead.");
        }
    }
    let _ = writeln!(out, "To pull more work forward today, run `cadence pull`.");
    let _ = writeln!(out, "Exiting — state saved; rerun to resume.");
    Ok(out)
}

fn print_day_complete(store: &SqliteStore, today: NaiveDate) -> Result<()> {
    print!("{}", format_day_complete(store, today)?);
    Ok(())
}

/// Partition pending tasks into due-today (executable now, §6 order) and
/// future work (date, then creation order). Pure over a task slice so the
/// loop prompt never names a task it cannot run.
fn due_and_future(tasks: &[Task], today: NaiveDate) -> (Vec<Task>, Vec<Task>) {
    let mut future: Vec<Task> = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::Pending && task.scheduled_for > today)
        .cloned()
        .collect();
    future.sort_by(|a, b| {
        a.scheduled_for
            .cmp(&b.scheduled_for)
            .then_with(|| a.sequence.cmp(&b.sequence))
    });
    (today_queue(tasks, today), future)
}

/// Last date a `days`-day horizon covers: `today + days`, saturating on
/// overflow (unreachable for real schedules). Non-positive spans collapse to
/// `today`, which excludes everything future-dated.
fn horizon_end(today: NaiveDate, days: i64) -> NaiveDate {
    if days < 1 {
        return today;
    }
    today
        .checked_add_days(chrono::Days::new(u64::try_from(days).unwrap_or(0)))
        .unwrap_or(today)
}

/// Pending future tasks within the next `days` days.
///
/// Strictly after `today` (the queue section already covers due work), up to
/// and including the horizon — ordered by date, then §6 rank. Pure over a
/// task slice.
#[must_use]
pub fn upcoming_tasks(tasks: &[Task], today: NaiveDate, days: i64) -> Vec<Task> {
    if days < 1 {
        return Vec::new();
    }
    let end = horizon_end(today, days);
    let mut out: Vec<Task> = tasks
        .iter()
        .filter(|task| {
            task.status == TaskStatus::Pending
                && task.scheduled_for > today
                && task.scheduled_for <= end
        })
        .cloned()
        .collect();
    out.sort_by_key(|task| (task.scheduled_for, execution_rank(task, today)));
    out
}

/// Render the upcoming-tasks listing: grouped by date with chapter titles,
/// plus a count of anything beyond the horizon so the calendar view never
/// silently truncates. Pure text (the `print_*` twin writes it).
fn format_upcoming(store: &SqliteStore, tasks: &[Task], today: NaiveDate, days: i64) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let upcoming = upcoming_tasks(tasks, today, days);
    if upcoming.is_empty() {
        let _ = writeln!(out, "Nothing scheduled in the next {days} day(s).");
    } else {
        let _ = writeln!(out, "Coming up (next {days} day(s)):");
        let mut current: Option<NaiveDate> = None;
        for task in &upcoming {
            if current != Some(task.scheduled_for) {
                current = Some(task.scheduled_for);
                let _ = writeln!(out, "  {}:", task.scheduled_for);
            }
            let _ = writeln!(
                out,
                "    [{}] {}",
                task.task_type.as_str(),
                chapter_title_or_id(store, task.chapter_id)
            );
        }
    }
    let end = horizon_end(today, days);
    let beyond = tasks
        .iter()
        .filter(|task| task.status == TaskStatus::Pending && task.scheduled_for > end)
        .count();
    if beyond > 0 {
        let _ = writeln!(out, "…and {beyond} more task(s) beyond {end}.");
    }
    out
}

/// Print pending future tasks within the next `days` days, grouped by date
/// with chapter titles, plus a count of anything scheduled beyond the
/// horizon so the calendar view never silently truncates.
fn print_upcoming(store: &SqliteStore, tasks: &[Task], today: NaiveDate, days: i64) {
    print!("{}", format_upcoming(store, tasks, today, days));
}

/// One-line description of a scheduled task with its chapter title.
///
/// # Errors
///
/// Returns [`Error::NotFound`] when the chapter is missing (corrupt store).
fn describe_task(store: &SqliteStore, task: &Task) -> Result<String> {
    let chapter = store.get_chapter(task.chapter_id)?;
    Ok(format!(
        "{} '{}' (due {})",
        task.task_type.as_str(),
        chapter.title,
        task.scheduled_for
    ))
}

/// Chapter title for display; falls back to `chapter {id}` when the row is
/// missing so a read-only listing never fails on a corrupt store.
fn chapter_title_or_id(store: &SqliteStore, chapter_id: i64) -> String {
    store.get_chapter(chapter_id).map_or_else(
        |_| format!("chapter {chapter_id}"),
        |chapter| format!("'{}'", chapter.title),
    )
}

/// Render the executable queue for `today`, naming chapters by title, with
/// the overdue section first. Pure text (the `print_*` twin writes it).
fn format_queue(store: &SqliteStore, tasks: &[Task], today: NaiveDate) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let queue = today_queue(tasks, today);
    let overdue = overdue_tasks(tasks, today);
    if queue.is_empty() {
        let _ = writeln!(out, "No tasks due for {today}. Queue is clear.");
        return out;
    }
    if !overdue.is_empty() {
        let _ = writeln!(out, "Overdue ({}):", overdue.len());
        for task in &overdue {
            let _ = writeln!(
                out,
                "  [{}] {} — {} (scheduled {})",
                task.task_type.as_str(),
                chapter_title_or_id(store, task.chapter_id),
                classify_label(task, today),
                task.scheduled_for
            );
        }
    }
    let _ = writeln!(out, "Today's queue ({}) for {today}:", queue.len());
    for (position, task) in queue.iter().enumerate() {
        let Some(number) = position.checked_add(1) else {
            continue;
        };
        let _ = writeln!(
            out,
            "  {number}. [{}] {} (scheduled {})",
            task.task_type.as_str(),
            chapter_title_or_id(store, task.chapter_id),
            task.scheduled_for
        );
    }
    out
}

/// Print the executable queue for `today`, naming chapters by title.
fn print_queue(store: &SqliteStore, tasks: &[Task], today: NaiveDate) {
    print!("{}", format_queue(store, tasks, today));
}

/// Short bucket label for display.
fn classify_label(task: &Task, today: NaiveDate) -> &'static str {
    match classify_task(task, today) {
        scheduler::TaskBucket::Overdue => "OVERDUE",
        scheduler::TaskBucket::DueToday => "DUE TODAY",
        scheduler::TaskBucket::Future => "FUTURE",
        scheduler::TaskBucket::Done => "DONE",
    }
}

/// First line of `text`, truncated to `chars` characters with an ellipsis.
#[must_use]
fn snip(text: &str, chars: usize) -> String {
    let first = text.lines().next().unwrap_or("");
    if first.chars().count() <= chars {
        return first.to_string();
    }
    format!("{}...", first.chars().take(chars).collect::<String>())
}

/// Evidence dashboard (§13): completion, performance fractions, retention
/// proxy, misconception resolution, trailing pace + ETA, consistency, and
/// learning debt. Skipped chapters are excluded from every denominator.
/// Observed numbers only — no mastery scores.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn run_metrics(store: &SqliteStore, today: NaiveDate) -> Result<()> {
    let report = collect_metrics(store, today)?;
    print!("{}", format_metrics(&report));
    Ok(())
}

/// Metrics inputs gathered from the store: the summarized dashboard board,
/// overdue tasks for the debt section, and chapters for title lookup.
struct MetricsReport {
    board: metrics::Dashboard,
    overdue: Vec<Task>,
    chapters: Vec<domain::Chapter>,
}

/// Gather metrics inputs: live chapters/tasks, evidence rows, and the
/// summarized dashboard board.
///
/// # Errors
///
/// Propagates [`Error::Store`] from store reads.
fn collect_metrics(store: &SqliteStore, today: NaiveDate) -> Result<MetricsReport> {
    let mut chapters = Vec::new();
    for book in store.list_books()? {
        chapters.extend(store.list_chapters(book.id)?);
    }
    let skipped = count_skipped(&chapters);
    let live: Vec<&domain::Chapter> = chapters.iter().filter(|c| !c.status.is_skipped()).collect();
    // Metrics only consider live rows: tasks on skipped chapters or from
    // abandoned attempts are audit trail, not evidence (§4.1).
    let live_tasks: Vec<Task> = store
        .list_tasks()?
        .into_iter()
        .filter(|t| {
            chapters
                .iter()
                .find(|c| c.id == t.chapter_id)
                .is_some_and(|c| !c.status.is_skipped() && t.attempt_no == c.attempt_no)
        })
        .collect();
    let tasks_done = live_tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Done)
        .count();
    let mut evidence = Vec::with_capacity(live.len());
    for chapter in &live {
        evidence.push(chapter_evidence(store, chapter)?);
    }
    let mut active_days = Vec::new();
    let mut punctual = Vec::new();
    for task in &live_tasks {
        if task.status != TaskStatus::Done {
            continue;
        }
        let Some(done) = task
            .completed_at
            .as_deref()
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        else {
            continue;
        };
        active_days.push(done);
        punctual.push((task.scheduled_for, done));
    }
    let overdue = overdue_tasks(&live_tasks, today);
    let board = metrics::summarize(&metrics::DashboardInput {
        chapters: &evidence,
        tasks_done,
        tasks_total: live_tasks.len(),
        skipped,
        active_days: &active_days,
        punctual: &punctual,
        overdue: overdue.len(),
        today,
    });
    Ok(MetricsReport {
        board,
        overdue,
        chapters,
    })
}

/// Render the trailing-pace line: ETA date, all-complete, or unknown.
/// Pure text.
fn format_pace(board: &metrics::Dashboard) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    match board.pace.eta {
        Some(_) if board.units_remaining == 0 => {
            let _ = writeln!(
                out,
                "Pace (last 7 days): {:.1} units/week, {:.1} pages/week — all units complete.",
                board.pace.units_per_week, board.pace.pages_per_week
            );
        }
        Some(date) => {
            let _ = writeln!(
                out,
                "Pace (last 7 days): {:.1} units/week, {:.1} pages/week — ETA {date} ({} units left).",
                board.pace.units_per_week, board.pace.pages_per_week, board.units_remaining
            );
        }
        None => {
            let _ = writeln!(
                out,
                "Pace (last 7 days): {:.1} units/week, {:.1} pages/week — ETA unknown (no completions in the last 7 days).",
                board.pace.units_per_week, board.pace.pages_per_week
            );
        }
    }
    out
}

/// Render the learning-debt section: overdue tasks with chapter titles, or
/// the all-clear line. Pure text.
fn format_debt(report: &MetricsReport) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    if report.overdue.is_empty() {
        let _ = writeln!(out, "Learning debt: none.");
        return out;
    }
    let _ = writeln!(
        out,
        "Learning debt: {} overdue task(s):",
        report.overdue.len()
    );
    for task in &report.overdue {
        let title = report
            .chapters
            .iter()
            .find(|c| c.id == task.chapter_id)
            .map_or_else(|| "unknown chapter".to_string(), |c| c.title.clone());
        let _ = writeln!(
            out,
            "  [{}] '{title}' (chapter {}, scheduled {})",
            task.task_type.as_str(),
            task.chapter_id,
            task.scheduled_for
        );
    }
    out
}
/// Render the evidence dashboard (§13): completion, performance fractions,
/// retention proxy, misconception resolution, trailing pace + ETA,
/// consistency, and learning debt. Pure text (the `print_*` twin writes it).
fn format_metrics(report: &MetricsReport) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let board = &report.board;
    let mut out = String::new();
    let units_total = board.units_completed.saturating_add(board.units_remaining);
    let _ = writeln!(
        out,
        "Completion: {}/{} units ({} pages done, {} left), {}/{} tasks done.",
        board.units_completed,
        units_total,
        board.pages_completed,
        board.pages_remaining,
        board.tasks_done,
        board.tasks_total
    );
    let _ = writeln!(
        out,
        "Skipped: {} chapter(s) (excluded from completion).",
        board.skipped
    );
    let (pretest_c, pretest_n) = board.pretest_fraction;
    let (retest_c, retest_n) = board.retest_fraction;
    let (assign_e, assign_p) = board.assignment_fraction;
    let _ = writeln!(
        out,
        "Performance: pretest {} · retest {} · assignment {}",
        metrics::format_fraction(pretest_c, pretest_n),
        metrics::format_fraction(retest_c, retest_n),
        metrics::format_fraction(assign_e, assign_p)
    );
    let _ = writeln!(
        out,
        "Retention proxy: {} → {} → {}",
        metrics::format_percent(board.pretest),
        metrics::format_percent(board.retest),
        metrics::format_percent(board.assignment)
    );
    let _ = writeln!(
        out,
        "Misconceptions: {} active, {} resolved (resolution rate {}).",
        board.active_misconceptions,
        board.resolved_misconceptions,
        metrics::format_percent(board.resolution)
    );
    out.push_str(&format_pace(board));
    let _ = writeln!(
        out,
        "Consistency: {} days active, current streak {} (longest {}), on-time {}.",
        board.consistency.days_active,
        board.consistency.current_streak,
        board.consistency.longest_streak,
        metrics::format_percent(board.on_time)
    );
    out.push_str(&format_debt(report));
    out
}

/// Render the misconception list across all chapters: per-book rows with
/// status counts. Skipped chapters stay listed (flagged). Pure text (the
/// `print_*` twin writes it).
///
/// # Errors
///
/// Propagates [`Error::Store`] from row listing.
fn format_misconceptions(store: &SqliteStore) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let mut active = 0_u64;
    let mut resolved = 0_u64;
    let mut disputed = 0_u64;
    for book in store.list_books()? {
        let mut printed_book = false;
        for chapter in store.list_chapters(book.id)? {
            let rows = store.list_misconceptions(chapter.id)?;
            if rows.is_empty() {
                continue;
            }
            if !printed_book {
                let _ = writeln!(out, "{}:", book.title);
                printed_book = true;
            }
            let skipped_mark = if chapter.status.is_skipped() {
                " [skipped — stays under review]"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  Chapter {} — '{}{skipped_mark}':",
                chapter.index_in_book.saturating_add(1),
                chapter.title
            );
            for row in &rows {
                match row.status.as_str() {
                    "ACTIVE" | "IMPROVING" => active = active.saturating_add(1),
                    "RESOLVED" => resolved = resolved.saturating_add(1),
                    "DISPUTED" => disputed = disputed.saturating_add(1),
                    _ => {}
                }
                let _ = writeln!(
                    out,
                    "    [{}] {} (confidence {:.2}, logged {})\n      {}",
                    row.status,
                    row.concept_description,
                    row.confidence,
                    row.created_at,
                    snip(&row.evidence, 120)
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "{active} active, {resolved} resolved, {disputed} disputed (purged)."
    );
    Ok(out)
}

fn run_misconceptions(store: &SqliteStore) -> Result<()> {
    print!("{}", format_misconceptions(store)?);
    Ok(())
}

/// Latest stored notes markdown for a chapter's current attempt (the notes
/// of record; generate-or-resume reuses it on rerun).
///
/// # Errors
///
/// Returns [`Error::NotFound`] for unknown chapters,
/// [`Error::InvalidInput`] when the chapter has no notes yet.
fn stored_notes_markdown(store: &dyn Store, chapter_id: i64) -> Result<String> {
    let chapter = store.get_chapter(chapter_id)?;
    let mut rows = store.list_notes(chapter.id, chapter.attempt_no)?;
    rows.pop().map(|note| note.content_markdown).ok_or_else(|| {
        Error::InvalidInput(format!(
            "chapter {chapter_id} ('{}') has no notes yet — notes are synthesized after grading",
            chapter.title
        ))
    })
}

/// `cadence notes [id]`: without args, list chapters with ids and whether
/// notes exist on the current attempt; with an id, print the chapter's
/// stored notes markdown (pipe into a pager for long notes).
///
/// # Errors
///
/// Propagates lookup failures from [`stored_notes_markdown`] and
/// [`Error::Store`] from the backend.
/// Render the notes listing: chapters that have notes on their current
/// attempt, plus usage. Pure text (the `print_*` twin writes it).
///
/// # Errors
///
/// Propagates [`Error::Store`] from listing reads.
fn format_notes_listing(store: &SqliteStore) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let _ = writeln!(out, "Chapters with notes:");
    for book in store.list_books()? {
        let mut printed_book = false;
        for chapter in store.list_chapters(book.id)? {
            if store.list_notes(chapter.id, chapter.attempt_no)?.is_empty() {
                continue;
            }
            if !printed_book {
                let _ = writeln!(out, "Book {} — {}:", book.id, book.title);
                printed_book = true;
            }
            let _ = writeln!(
                out,
                "  {} '{}' (pages {}–{}, attempt {})",
                chapter.id, chapter.title, chapter.start_page, chapter.end_page, chapter.attempt_no,
            );
        }
    }
    let _ = writeln!(
        out,
        "Usage: cadence notes <id> (pipe into a pager for long notes)"
    );
    Ok(out)
}

fn run_notes(store: &SqliteStore, id: Option<i64>) -> Result<()> {
    let Some(wanted) = id else {
        print!("{}", format_notes_listing(store)?);
        return Ok(());
    };
    print!("{}", stored_notes_markdown(store, wanted)?);
    Ok(())
}

/// Render per-book progress: live vs skipped chapter counts plus per-chapter
/// evidence lines. Pure text (the `print_*` twin writes it).
///
/// # Errors
///
/// Propagates [`Error::Store`] from evidence reads.
fn format_run_progress(store: &SqliteStore) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    for book in store.list_books()? {
        let chapters = store.list_chapters(book.id)?;
        let live_count = chapters.iter().filter(|c| !c.status.is_skipped()).count();
        let skipped_count = chapters.len().saturating_sub(live_count);
        let _ = writeln!(
            out,
            "{} — {live_count} unit(s), {skipped_count} skipped:",
            book.title
        );
        for chapter in chapters.iter().filter(|c| !c.status.is_skipped()) {
            let ev = chapter_evidence(store, chapter)?;
            let _ = writeln!(
                out,
                "  Ch{} '{}' [{}] pages {}–{}: pretest {} · retest {} · assignment {} · misconceptions {} active / {} resolved",
                chapter.index_in_book.saturating_add(1),
                chapter.title,
                chapter.status.as_str(),
                chapter.start_page,
                chapter.end_page,
                metrics::format_fraction(ev.pretest_correct, ev.pretest_answered),
                metrics::format_fraction(ev.retest_correct, ev.retest_answered),
                metrics::format_fraction(ev.assignment_earned, ev.assignment_possible),
                ev.active_misconceptions,
                ev.resolved_misconceptions
            );
        }
        for chapter in chapters.iter().filter(|c| c.status.is_skipped()) {
            let _ = writeln!(
                out,
                "  Skipped: Ch{} '{}' (excluded from evidence)",
                chapter.index_in_book.saturating_add(1),
                chapter.title
            );
        }
    }
    Ok(out)
}

/// Book-level progress with retention evidence (§13): per-chapter status,
/// pages, MCQ/assignment fractions, and misconception tallies. Skipped
/// chapters are listed separately, excluded from evidence.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn run_progress(store: &SqliteStore) -> Result<()> {
    print!("{}", format_run_progress(store)?);
    Ok(())
}

/// Read one manual-boundaries line from interactive stdin.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when stdin is not a terminal or yields no
/// usable line.
fn read_manual_from_stdin(leaf: &str) -> Result<String> {
    if !std::io::stdin().is_terminal() {
        return Err(Error::NeedsManual(format!(
            "{leaf} (re-run with --manual-boundaries start-end,...)"
        )));
    }
    println!("{leaf}");
    print!("Enter manual boundaries (start-end,start-end,...): ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let trimmed = line.trim().to_string();
    if trimmed.is_empty() {
        return Err(Error::InvalidInput(
            "no manual boundaries entered".to_string(),
        ));
    }
    Ok(trimmed)
}

/// Generate-or-resume the stored review set for a production chapter/attempt
/// (§12): reuse the validated set when present, otherwise run one
/// `complete_cached` targeted generation (shape- plus topic-validated before
/// caching, so drift fast-retries with direction) over the chapter text and
/// persist it. The target-set hash is part of the cache identity: a changed
/// open set regenerates instead of serving stale questions. Without
/// connectivity and without stored rows the LLM error propagates with state
/// intact (§16).
fn ensure_review_items(
    store: &mut SqliteStore,
    chapter: &domain::Chapter,
    unit: &engines::UnitText,
    targets: &[review::ReviewTarget],
    today_str: &str,
) -> Result<Vec<store::McqItem>> {
    let items = store.list_mcq_items(
        chapter.id,
        mcq::McqPhase::Review.as_str(),
        chapter.attempt_no,
    )?;
    if !items.is_empty() {
        println!(
            "Resumed {} stored review question(s) for '{}'.",
            items.len(),
            chapter.title
        );
        return Ok(items);
    }
    let count = review::review_question_count(targets.len());
    println!(
        "Generating {count} targeted review question(s) for '{}' (pages {}–{}) …",
        chapter.title, unit.page_start, unit.page_end
    );
    let concepts = review::concepts_of(targets);
    let prompt = mcq::build_review_prompt(unit, count, &concepts);
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = review::REVIEW_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &mcq::mcq_review_response_schema(),
        "review_set",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = review::review_params_json(count, &review::targets_hash(targets));
    let source_hash = mcq::source_hash_for(&unit.text);
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today_str,
    };
    let operation = mcq::McqPhase::Review.as_str().to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let unit_ref = unit;
    let concepts_ref = &concepts;
    let validate = |text: &str| {
        let items = mcq::validate_mcq_set_ranged(text, unit_ref, count, count)?;
        mcq::check_review_topics(&items, concepts_ref)?;
        Ok(text.trim().to_string())
    };
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    println!(
        "Generated via LLM (cache: {}, transport sends: {}).",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let validated = mcq::validate_mcq_set_ranged(&result.text, unit, count, count)?;
    mcq::check_review_topics(&validated, &concepts)?;
    let new_rows = dev_mcq::to_new_items_for(
        chapter.id,
        mcq::McqPhase::Review,
        &validated,
        chapter.attempt_no,
    )?;
    store.save_mcq_items(&new_rows)
}

/// Totals accumulated across review chapters.
#[derive(Default)]
struct ReviewTotals {
    /// Chapters fully reviewed.
    chapters_done: usize,
    /// Correct answers across chapters.
    correct: usize,
    /// Answered questions across chapters.
    answered: usize,
}

/// Collect one [`review::ChapterOpen`] view per chapter: every stored
/// misconception row mapped to its display shape.
///
/// # Errors
///
/// Propagates [`Error::Store`] from row listing.
fn collect_review_views(store: &SqliteStore) -> Result<Vec<review::ChapterOpen>> {
    let mut views = Vec::new();
    for book in store.list_books()? {
        for chapter in store.list_chapters(book.id)? {
            let mut rows = Vec::new();
            for row in store.list_misconceptions(chapter.id)? {
                rows.push(review::MisconceptionView {
                    id: row.id,
                    concept: row.concept_description,
                    description: row.description,
                    evidence: row.evidence,
                    status: row.status,
                    confidence: row.confidence,
                });
            }
            views.push(review::ChapterOpen {
                chapter_id: chapter.id,
                title: chapter.title.clone(),
                status: chapter.status,
                rows,
            });
        }
    }
    Ok(views)
}

/// Run one chapter's review group: load text, ensure items, run the session,
/// log the event. Returns `Ok(false)` on early user exit (counters left
/// untouched); otherwise folds the summary into `totals` and returns `Ok(true)`.
///
/// # Errors
///
/// Propagates [`Error::Store`] and LLM failures.
fn run_review_group(
    store: &mut SqliteStore,
    chapter_id: i64,
    group: &[review::ReviewTarget],
    today_str: &str,
    totals: &mut ReviewTotals,
) -> Result<bool> {
    let chapter = store.get_chapter(chapter_id)?;
    let unit = match ingest::load_unit_text(std::path::Path::new(&chapter.file_path)) {
        Ok(loaded) => loaded,
        Err(error) => {
            println!(
                "Skipping review for '{}': cannot load chapter text ({error}).",
                chapter.title
            );
            return Ok(true);
        }
    };
    let items = ensure_review_items(store, &chapter, &unit, group, today_str)?;
    println!(
        "\nReview: '{}' — {} targeted question(s) re-probing {} misconception(s).",
        chapter.title,
        items.len(),
        group.len()
    );
    let summary = run_mcq_session(
        store,
        &items,
        &unit,
        mcq::McqPhase::Review,
        chapter.id,
        chapter.attempt_no,
        None,
    )?;
    if !summary.completed {
        return Ok(false);
    }
    totals.chapters_done = totals.chapters_done.saturating_add(1);
    totals.correct = totals.correct.saturating_add(summary.correct);
    totals.answered = totals.answered.saturating_add(summary.answered);
    store.log_event(
        "REVIEW_SESSION",
        Some(chapter.id),
        None,
        Some(&format!("{}/{}", summary.correct, summary.answered)),
        today_str,
    )?;
    Ok(true)
}

/// Manual cumulative review (§12): pull every open misconception across
/// completed (and skipped) chapters, generate-or-resume one targeted
/// maximum-difficulty MCQ set per chapter, and run the interactive sessions
/// in book order. Correct answers boost topic-matched rows (+0.1, resolving
/// at threshold); wrong answers nudge them (−0.1); `"I don't know"`
/// abstains with no lifecycle effect. Review never touches scheduler state
/// — it is manual, outside the 3-day window. Chapters whose text cannot
/// load are reported and skipped, never silent. Early exits keep
/// per-question state for resume.
///
/// # Errors
///
/// Propagates [`Error::Store`] and LLM failures; a failed chapter aborts the
/// run with earlier chapters' state intact.
fn run_review(store: &mut SqliteStore, today_str: &str) -> Result<()> {
    let views = collect_review_views(store)?;
    let targets = review::select_targets(&views);
    if targets.is_empty() {
        println!("No open misconceptions — nothing to review.");
        return Ok(());
    }
    let mut index = 0_usize;
    let mut totals = ReviewTotals::default();
    while let Some(head) = targets.get(index) {
        let chapter_id = head.chapter_id;
        let group_len = targets.get(index..).map_or(0, |rest| {
            rest.iter()
                .take_while(|target| target.chapter_id == chapter_id)
                .count()
        });
        let Some(next) = index.checked_add(group_len) else {
            break;
        };
        let Some(group) = targets.get(index..next) else {
            break;
        };
        index = next;
        if group.is_empty() {
            continue;
        }
        if !run_review_group(store, chapter_id, group, today_str, &mut totals)? {
            return Ok(());
        }
    }
    println!(
        "\nReview complete: {}/{} correct across {} chapter(s).",
        totals.correct, totals.answered, totals.chapters_done
    );
    Ok(())
}

/// Destructive-action twin of [`ask_continue`]: same explicit-`y` rule, but
/// piped stdin and EOF abort instead of proceeding, so a library wipe can
/// never fire without an interactive yes.
///
/// # Errors
///
/// Propagates [`Error::Io`] on terminal read failures.
fn ask_destructive(prompt: &str) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Ok(false);
    }
    ask_continue(prompt)
}

/// Progress snapshot of one tracked book for the single-book ingest guard.
struct BookProgress {
    title: String,
    done: u64,
    total: u64,
    skipped: u64,
    pages_done: u64,
    pages_total: u64,
}

/// Summarize one book's completion for the ingest guard: chapters completed
/// vs tracked (skips called out, never counted as done) plus chapter pages.
///
/// # Errors
///
/// Propagates [`Error::Store`] from chapter listing.
fn book_progress(store: &dyn Store, book: &domain::Book) -> Result<BookProgress> {
    let mut progress = BookProgress {
        title: book.title.clone(),
        done: 0,
        total: 0,
        skipped: 0,
        pages_done: 0,
        pages_total: 0,
    };
    for chapter in store.list_chapters(book.id)? {
        progress.total = progress.total.saturating_add(1);
        let pages = domain::unit_page_count(chapter.start_page, chapter.end_page).unwrap_or(0);
        let pages = u64::try_from(pages).unwrap_or(0);
        progress.pages_total = progress.pages_total.saturating_add(pages);
        if chapter.status == domain::ChapterStatus::Completed {
            progress.done = progress.done.saturating_add(1);
            progress.pages_done = progress.pages_done.saturating_add(pages);
        } else if chapter.status.is_skipped() {
            progress.skipped = progress.skipped.saturating_add(1);
        }
    }
    Ok(progress)
}

/// One-line library summary, e.g.
/// `Modern C: 1/22 (5%) chapters, 2 skipped, 10/391 pages`.
#[must_use]
fn format_book_progress(progress: &BookProgress) -> String {
    let skipped = if progress.skipped == 0 {
        String::new()
    } else {
        format!(", {} skipped", progress.skipped)
    };
    format!(
        "  {}: {} chapters{skipped}, {}/{} pages",
        progress.title,
        metrics::format_fraction(progress.done, progress.total),
        progress.pages_done,
        progress.pages_total
    )
}

/// Single-book guard (§3): `cadence` tracks one book at a time. When the
/// store already holds books, show their progress and require an explicit
/// `y` to wipe the library before ingesting `new_title`: every record goes
/// (books, tasks, grades, notes, history) while the PDF file itself is kept
/// and the content-addressed LLM cache survives. Returns whether ingest
/// should proceed — anything but `y` aborts with the library untouched.
///
/// # Errors
///
/// Propagates [`Error::Store`] from listing/clearing and [`Error::Io`] from
/// prompt reads and unit-cache removal.
fn guard_single_book(store: &mut dyn Store, config: &Config, new_title: &str) -> Result<bool> {
    let books = store.list_books()?;
    if books.is_empty() {
        return Ok(true);
    }
    if books.len() == 1 {
        println!("Already tracking a book:");
    } else {
        println!("Already tracking {} books:", books.len());
    }
    for book in &books {
        println!("{}", format_book_progress(&book_progress(store, book)?));
    }
    println!(
        "Starting a new book deletes ALL existing records (books, tasks, grades, notes, history). The PDF file itself is kept."
    );
    if !ask_destructive(&format!("Replace the library with '{new_title}'? [y/N] > "))? {
        println!("Ingest aborted — existing library untouched.");
        return Ok(false);
    }
    store.clear_library()?;
    let mut books_dir = config.data_dir.clone();
    books_dir.push("books");
    if books_dir.exists() {
        std::fs::remove_dir_all(&books_dir)?;
    }
    println!("Library cleared.");
    Ok(true)
}

/// Real ingest: outline → split → extract → file store → book registration.
#[expect(
    clippy::too_many_lines,
    reason = "pipeline stages run in fixed order; per-stage helpers already extracted"
)]
fn run_ingest(
    pdf: &str,
    start_page: i64,
    max_unit_pages: i64,
    title: Option<&str>,
    manual: Option<&str>,
    chapter_level: Option<i64>,
    config: &Config,
) -> Result<()> {
    use std::path::Path;
    domain::validate_book_registration(start_page)?;
    if max_unit_pages < 1 {
        return Err(Error::InvalidInput(
            "--max-unit-pages must be >= 1".to_string(),
        ));
    }
    let pdf_path = Path::new(pdf);
    let resolved_title = title.map_or_else(|| ingest::default_title(pdf_path), ToString::to_string);
    let bytes = std::fs::read(pdf_path)?;
    let hash = ingest::file_hash(&bytes);
    let source = pdf::MuPdfSource::open(pdf_path)?;
    let page_count = source.page_count();
    if start_page > page_count {
        return Err(Error::InvalidInput(format!(
            "start page {start_page} exceeds document page count {page_count}"
        )));
    }
    let boundaries = source.outline(start_page)?;
    println!(
        "{}",
        split::describe_levels(&boundaries, start_page, page_count, chapter_level)
    );
    let planned = match ingest::plan_book(
        &boundaries,
        start_page,
        page_count,
        max_unit_pages,
        manual,
        chapter_level,
    ) {
        Ok(units) => units,
        Err(Error::NeedsManual(msg)) if manual.is_none() => {
            let line = read_manual_from_stdin(&msg)?;
            ingest::plan_book(
                &boundaries,
                start_page,
                page_count,
                max_unit_pages,
                Some(&line),
                chapter_level,
            )?
        }
        Err(other) => return Err(other),
    };
    let mut store = SqliteStore::open(&config.db_path(), &config.lock_path())?;
    if !guard_single_book(&mut store, config, &resolved_title)? {
        return Ok(());
    }
    let units = ingest::extract_units(&source, &planned, &hash)?;
    let stored = ingest::store_units(&config.data_dir, &hash, &units)?;
    let today = today_date().format("%Y-%m-%d").to_string();
    let book = ingest::register_book(
        &mut store,
        &store::NewBook {
            title: resolved_title.clone(),
            filepath: pdf_path.to_string_lossy().into_owned(),
            file_hash: hash.clone(),
            start_page,
        },
        &units,
        &stored,
        &today,
    )?;
    let ingest_day = today_date();
    let topped = ensure_tasks(&mut store, book.id, ingest_day, &today)?.len();
    println!(
        "ingested book {} ({resolved_title}): {} unit(s), hash {hash} ({topped} initial task(s) scheduled)",
        book.id,
        units.len()
    );
    for (index, unit) in units.iter().enumerate() {
        let Some(number) = index.checked_add(1) else {
            continue;
        };
        println!(
            "  {number}. {} (pages {}–{}, level {})",
            unit.heading, unit.start_page, unit.end_page, unit.level
        );
    }
    Ok(())
}

/// `cadence dev ingest`: isolated outline + split + text, printed as JSON.
/// Never mutates production storage.
fn run_dev_ingest(
    pdf_path: &str,
    start_page: i64,
    max_unit_pages: i64,
    manual: Option<&str>,
    chapter_level: Option<i64>,
    max_units: Option<usize>,
) -> Result<()> {
    use std::path::Path;
    let path = Path::new(pdf_path);
    let source = pdf::MuPdfSource::open(path)?;
    let page_count = source.page_count();
    let boundaries = source.outline(start_page)?;
    // Table goes to stderr (best-effort; stdout stays pure unit JSON).
    let _ = writeln!(
        std::io::stderr(),
        "{}",
        split::describe_levels(&boundaries, start_page, page_count, chapter_level)
    );
    let planned = ingest::plan_book(
        &boundaries,
        start_page,
        page_count,
        max_unit_pages,
        manual,
        chapter_level,
    )?;
    let take = max_units.unwrap_or(planned.len());
    let capped: Vec<engines::PlannedUnit> = planned.into_iter().take(take).collect();
    let bytes = std::fs::read(path).unwrap_or_default();
    let hash = ingest::file_hash(&bytes);
    let units = ingest::extract_units(&source, &capped, &hash)?;
    let payloads: Vec<ingest::UnitPayload> = units
        .iter()
        .map(|u| ingest::UnitPayload {
            heading: u.heading.clone(),
            level: u.level,
            page_start: u.start_page,
            page_end: u.end_page,
            text: u.text.text.clone(),
            source_pdf_hash: hash.clone(),
        })
        .collect();
    let json = serde_json::to_string_pretty(&payloads).map_err(|e| Error::Io(e.to_string()))?;
    println!("{json}");
    Ok(())
}

/// Read a smoke prompt from piped stdin (`--prompt` not given).
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when stdin is a terminal (no piped input)
/// or yields only blank text.
fn read_prompt_from_stdin() -> Result<String> {
    use std::io::Read as _;
    if std::io::stdin().is_terminal() {
        return Err(Error::InvalidInput(
            "pass --prompt \"...\" or pipe a prompt on stdin".to_string(),
        ));
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    if text.trim().is_empty() {
        return Err(Error::InvalidInput("empty prompt on stdin".to_string()));
    }
    Ok(text)
}

/// `cadence dev llm`: real transport smoke test with durable job + cache.
/// Uses an isolated scratch database (`.scratch/dev-llm/`); never touches
/// production storage.
fn run_dev_llm(operation: &str, prompt: Option<&str>, model: Option<&str>) -> Result<()> {
    let mut config = llm::LlmConfig::from_env()?;
    if let Some(name) = model
        && !name.trim().is_empty()
    {
        config = config.with_model(name);
    }
    let prompt_text = prompt.map_or_else(
        || read_prompt_from_stdin().map(|t| t.trim().to_string()),
        |p| {
            if p.trim().is_empty() {
                Err(Error::InvalidInput("empty --prompt".to_string()))
            } else {
                Ok(p.trim().to_string())
            }
        },
    )?;
    let db_path = std::path::PathBuf::from(".scratch/dev-llm/cadence.db");
    let lock_path = std::path::PathBuf::from(".scratch/dev-llm/cadence.lock");
    let mut store = SqliteStore::open(&db_path, &lock_path)?;
    let provider = llm::HttpLlmProvider::new(config.clone())?;
    let params = format!("{{\"max_tokens\":{}}}", config.max_tokens);
    let today = today_date().format("%Y-%m-%d").to_string();
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today.as_str(),
    };
    let result = llm::complete_cached(
        &provider,
        &mut store,
        &llm::CachedRequest {
            operation,
            prompt: prompt_text.as_str(),
            source_hash: "",
            params_json: params.as_str(),
        },
        &llm::validate_smoke,
        &hooks,
    )?;
    println!("model: {}", config.model);
    println!("cache: {}", if result.cache_hit { "hit" } else { "miss" });
    println!("transport sends: {}", result.transport_calls);
    println!("--- response ---\n{}", result.text);
    println!(
        "(paid upgrade target: {} — needs purchased credits)",
        llm::UPGRADE_MODEL
    );
    Ok(())
}
/// First-unit context shared by the dev stage harnesses (§2.1).
struct FirstUnit {
    /// Extracted chapter text.
    unit: engines::UnitText,
    /// Hex SHA-256 of the PDF bytes (dev-book identity).
    pdf_hash: String,
    /// File name for display and dev-book titles.
    pdf_name: String,
}

/// Resolve the first study unit of a fixture PDF: semantic outline + 50-page
/// plan when available, otherwise the leading ≤50 pages as one unit (fixture
/// PDFs are single chapters; never invent splits beyond that fallback).
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on bad page ranges and [`Error::Pdf`] on
/// extraction failures.
fn extract_first_unit(pdf_path: &str, start_page: i64) -> Result<FirstUnit> {
    use std::path::Path;
    if start_page < 1 {
        return Err(Error::InvalidInput(
            "--start-page must be >= 1 (one-based physical page)".to_string(),
        ));
    }
    let path = Path::new(pdf_path);
    let source = pdf::MuPdfSource::open(path)?;
    let page_count = source.page_count();
    if start_page > page_count {
        return Err(Error::InvalidInput(format!(
            "start page {start_page} exceeds document page count {page_count}"
        )));
    }
    let pdf_bytes = std::fs::read(path).unwrap_or_default();
    let pdf_hash = ingest::file_hash(&pdf_bytes);
    let unit: engines::UnitText = match source.outline(start_page) {
        Ok(boundaries) => {
            match ingest::plan_book(&boundaries, start_page, page_count, 50, None, None) {
                Ok(planned) => {
                    let Some(first) = planned.first() else {
                        return Err(Error::InvalidInput("no study units planned".to_string()));
                    };
                    let mut text = source.text_for_range(first.start_page, first.end_page)?;
                    text.heading.clone_from(&first.heading);
                    println!(
                        "dev: unit '{}' (pages {}–{} of {page_count})",
                        first.heading, first.start_page, first.end_page
                    );
                    text
                }
                Err(_) => fallback_unit(&source, path, start_page, page_count)?,
            }
        }
        Err(_) => fallback_unit(&source, path, start_page, page_count)?,
    };
    let pdf_name = path.file_name().map_or_else(
        || pdf_path.to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    Ok(FirstUnit {
        unit,
        pdf_hash,
        pdf_name,
    })
}

/// `cadence dev mcq`: isolated MCQ stage against a fixture PDF (§2.1, §7.1).
/// Extracts the first unit's text via `MuPdfSource`, generates 8 questions
/// through `complete_cached` (cache → durable job → retry with MCQ
/// validation), persists them with `save_mcq_items`, then runs the interactive
/// A–E loop with immediate feedback. Defaults to an in-memory store;
/// `--persist --seed S` writes to `.scratch/dev-mcq/` and resumes answered
/// items on rerun. `--print-only` generates/caches without interacting.
/// Never touches `~/.cadence/`.
#[expect(
    clippy::too_many_lines,
    reason = "dev-harness orchestration: extract, generate, persist, and interact in one readable flow"
)]
fn run_dev_mcq(
    pdf_path: &str,
    phase_label: &str,
    start_page: i64,
    seed: Option<u64>,
    persist: bool,
    print_only: bool,
    generation: Option<&str>,
) -> Result<()> {
    use std::path::Path;
    let phase = mcq::McqPhase::parse(phase_label)?;
    let first = extract_first_unit(pdf_path, start_page)?;
    let unit = first.unit;
    let pdf_hash = first.pdf_hash.clone();
    let pdf_name = first.pdf_name;
    let count = dev_mcq::DEV_MCQ_COUNT;
    let prompt = match phase {
        mcq::McqPhase::Pretest => mcq::build_pretest_prompt(&unit, count),
        mcq::McqPhase::Retest => mcq::build_retest_prompt(&unit, count),
        mcq::McqPhase::Review => {
            return Err(Error::InvalidInput(
                "dev mcq supports pretest|retest (review sets need stored misconceptions — use `cadence review`)".to_string(),
            ));
        }
    };
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = dev_mcq::DEV_MCQ_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &mcq::mcq_response_schema(),
        "mcq_set",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = mcq::mcq_params_json_with_generation(count, phase, generation);
    let source_hash = mcq::source_hash_for(&unit.text);
    // Store: in-memory default; `--persist` isolates to `.scratch/dev-mcq/`
    // keyed by seed so reruns resume (never production storage).
    let seed_tag = seed.unwrap_or(0);
    let mut mem = MemoryStore::new();
    let mut disk: Option<SqliteStore> = None;
    if persist {
        let dir = Path::new(".scratch/dev-mcq");
        let db_path = dir.join(format!("dev-mcq-{seed_tag}.db"));
        let lock_path = dir.join(format!("dev-mcq-{seed_tag}.lock"));
        disk = Some(SqliteStore::open(&db_path, &lock_path)?);
        println!("dev mcq: persisting to {}", db_path.display());
    }
    let store: &mut dyn Store = match disk.as_mut() {
        Some(db) => db,
        None => &mut mem,
    };
    // Show which file is being processed (avoids confusion with fixtures).
    println!(
        "dev mcq: pdf={pdf_name} phase={} start_page={start_page}",
        phase.as_str()
    );
    let chapter = dev_mcq::ensure_dev_chapter(store, &pdf_hash, &pdf_name, &unit)?;
    let mut items = store.list_mcq_items(chapter.id, phase.as_str(), dev_mcq::DEV_ATTEMPT_NO)?;
    if generation.is_some() && !items.is_empty() {
        // A named generation is a fresh experiment: it replaces the stored set
        // (LLM cache identity alone cannot do this — stored rows shadow it).
        let dropped =
            store.delete_mcq_items_for(chapter.id, phase.as_str(), dev_mcq::DEV_ATTEMPT_NO)?;
        println!(
            "dev mcq: discarded {dropped} stored question(s) for fresh generation '{}'",
            generation.unwrap_or_default()
        );
        items = Vec::new();
    }
    if items.is_empty() {
        let today = today_date().format("%Y-%m-%d").to_string();
        let hooks = llm::RunHooks {
            sleep: &std::thread::sleep,
            now_iso: today.as_str(),
        };
        let operation = phase.as_str().to_string();
        let request = llm::CachedRequest {
            operation: operation.as_str(),
            prompt: prompt.as_str(),
            source_hash: source_hash.as_str(),
            params_json: params.as_str(),
        };
        let unit_ref = &unit;
        let validate =
            |text: &str| mcq::validate_mcq_set(text, unit_ref).map(|_| text.trim().to_string());
        let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
        println!(
            "dev mcq: generated via LLM (cache: {}, transport sends: {})",
            if result.cache_hit { "hit" } else { "miss" },
            result.transport_calls
        );
        let validated = mcq::validate_mcq_set(&result.text, &unit)?;
        let new_rows = dev_mcq::to_new_items(chapter.id, phase, &validated)?;
        items = store.save_mcq_items(&new_rows)?;
    } else {
        println!(
            "dev mcq: resumed {} cached question(s) for {} (new --seed, or --generation <name> to replace them)",
            items.len(),
            phase.as_str()
        );
    }
    if print_only {
        print_mcq_set(&items, &unit)?;
        return Ok(());
    }
    run_mcq_session(
        store,
        &items,
        &unit,
        phase,
        chapter.id,
        dev_mcq::DEV_ATTEMPT_NO,
        seed,
    )
    .map(|_| ())
}

/// Fallback unit when the PDF has no usable outline: up to 50 pages from
/// `start_page`.
fn fallback_unit(
    source: &pdf::MuPdfSource,
    path: &std::path::Path,
    start_page: i64,
    page_count: i64,
) -> Result<engines::UnitText> {
    let end = start_page.saturating_add(49).min(page_count);
    let mut unit = source.text_for_range(start_page, end)?;
    unit.heading = ingest::default_title(path);
    println!("dev mcq: no outline — using pages {start_page}–{end} of {page_count} as one unit");
    Ok(unit)
}

/// Print a generated set without interacting (`--print-only`).
fn print_mcq_set(items: &[store::McqItem], unit: &engines::UnitText) -> Result<()> {
    println!("MCQ set: {} question(s)", items.len());
    for (position, item) in items.iter().enumerate() {
        let Some(number) = position.checked_add(1) else {
            continue;
        };
        let row = dev_mcq::validated_from_row(
            &dev_mcq::McqRowView {
                question: &item.question_text,
                options_json: &item.options_json,
                correct_index: item.correct_index,
                trap_index: item.trap_index,
                explanation: &item.explanation_text,
                topic: &item.topic,
                source_refs: &item.source_refs,
            },
            unit,
        )?;
        let shown = mcq::apply_shuffle(&row, dev_mcq::shuffle_seed_for(0, position))
            .ok_or_else(|| Error::Store("stored MCQ failed shuffle".to_string()))?;
        println!(
            "\n{}",
            question_header(number, items.len(), colors_enabled())
        );
        println!("{}", row.question);
        for (index, option) in shown.displayed_options.iter().enumerate() {
            println!("   {}) {option}", dev_mcq::option_label(index));
        }
        println!(
            "   {}) {}",
            dev_mcq::option_label(mcq::IDK_INDEX),
            dev_mcq::idk_label()
        );
    }
    Ok(())
}

/// Outcome of one interactive MCQ session: totals for the §15 checkpoint
/// footer plus whether every item now has a recorded answer.
struct SessionSummary {
    /// Items answered in this invocation (including previously resumed).
    answered: usize,
    /// Items answered correctly (best recorded response per item).
    correct: usize,
    /// Misconceptions logged during this invocation.
    misconceptions: usize,
    /// True when every item has a recorded answer (no early exit).
    completed: bool,
}

/// Divider in the `nvim` buffer (§7.2): the question above, the answer below.
const ANSWER_DIVIDER: &str = "--- write your answer below this line ---";

/// `cadence dev assignment`: isolated written-assignment stage against a
/// fixture PDF (§2.1, §7.2). Resolves the first unit, generates 3–5 written
/// questions plus one coding question through `complete_cached` (frozen
/// creation-time rubrics), persists them with `save_assignment_questions`,
/// then collects closed-book answers in `nvim` below a divider line.
/// Defaults to an in-memory store; `--persist --seed S` writes to
/// `.scratch/dev-assignment/` and resumes unanswered questions on rerun.
/// `--print-only` generates without opening the editor. Never touches
/// `~/.cadence/`.
#[expect(
    clippy::too_many_lines,
    reason = "dev-harness orchestration: generate, persist, and collect answers in one readable flow"
)]
fn run_dev_assignment(
    pdf_path: &str,
    start_page: i64,
    seed: Option<u64>,
    persist: bool,
    print_only: bool,
) -> Result<()> {
    use std::path::Path;
    let first = extract_first_unit(pdf_path, start_page)?;
    let unit = first.unit;
    // Store: in-memory default; `--persist` isolates to
    // `.scratch/dev-assignment/` keyed by seed (never production storage).
    let seed_tag = seed.unwrap_or(0);
    let mut mem = MemoryStore::new();
    let mut disk: Option<SqliteStore> = None;
    if persist {
        let dir = Path::new(".scratch/dev-assignment");
        let db_path = dir.join(format!("dev-assignment-{seed_tag}.db"));
        let lock_path = dir.join(format!("dev-assignment-{seed_tag}.lock"));
        disk = Some(SqliteStore::open(&db_path, &lock_path)?);
        println!("dev assignment: persisting to {}", db_path.display());
    }
    let store: &mut dyn Store = match disk.as_mut() {
        Some(db) => db,
        None => &mut mem,
    };
    println!(
        "dev assignment: pdf={} start_page={start_page}",
        first.pdf_name
    );
    let chapter = dev_mcq::ensure_dev_chapter(store, &first.pdf_hash, &first.pdf_name, &unit)?;
    let mut items = store.list_assignment_questions(chapter.id, dev_mcq::DEV_ATTEMPT_NO)?;
    if items.is_empty() {
        let open = open_misconceptions(store, chapter.id)?;
        let prompt = assignment::build_assignment_prompt(&unit, &open);
        let mut config = llm::LlmConfig::from_env()?;
        config.max_tokens = assignment::ASSIGNMENT_MAX_TOKENS;
        config.response_format_json = Some(llm::response_format_envelope(
            &assignment::assignment_response_schema(),
            "assignment_set",
        ));
        let provider = llm::HttpLlmProvider::new(config)?;
        let hash = assignment::misconceptions_hash(&open);
        let params = assignment::assignment_params_json(&hash);
        let source_hash = mcq::source_hash_for(&unit.text);
        let today = today_date().format("%Y-%m-%d").to_string();
        let hooks = llm::RunHooks {
            sleep: &std::thread::sleep,
            now_iso: today.as_str(),
        };
        let operation = "assignment".to_string();
        let request = llm::CachedRequest {
            operation: operation.as_str(),
            prompt: prompt.as_str(),
            source_hash: source_hash.as_str(),
            params_json: params.as_str(),
        };
        let unit_ref = &unit;
        let open_ref = &open;
        let validate = |text: &str| {
            assignment::validate_assignment_set(text, unit_ref, open_ref)
                .map(|_| text.trim().to_string())
        };
        let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
        println!(
            "dev assignment: generated via LLM (cache: {}, transport sends: {})",
            if result.cache_hit { "hit" } else { "miss" },
            result.transport_calls
        );
        let validated = assignment::validate_assignment_set(&result.text, &unit, &open)?;
        let new_rows =
            assignment::to_new_questions(chapter.id, dev_mcq::DEV_ATTEMPT_NO, &validated)?;
        items = store.save_assignment_questions(&new_rows)?;
    } else {
        println!(
            "dev assignment: resumed {} cached question(s) (answer files live in the editor until saved)",
            items.len()
        );
    }
    if print_only {
        print_assignment_set(&items);
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        return Err(Error::InvalidInput(
            "dev assignment needs an interactive terminal for nvim (or use --print-only)"
                .to_string(),
        ));
    }
    require_nvim()?;
    let completed = run_assignment_answers(store, &items, dev_mcq::DEV_ATTEMPT_NO)?;
    if completed {
        println!(
            "\nAssignment answers complete: {}/{}.",
            items.len(),
            items.len()
        );
    }
    Ok(())
}

/// Open misconceptions for assignment re-probing (§12): active rows mapped to
/// engine input (id + concept only; the engine never touches the store).
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn open_misconceptions(
    store: &dyn Store,
    chapter_id: i64,
) -> Result<Vec<assignment::OpenMisconception>> {
    let mut out = Vec::new();
    for row in store.list_misconceptions(chapter_id)? {
        if row.status == "ACTIVE" {
            out.push(assignment::OpenMisconception {
                id: row.id,
                concept: row.concept_description,
            });
        }
    }
    Ok(out)
}

/// Print a generated assignment without interacting (`--print-only`).
/// Model solutions stay hidden: they are grader-internal until grading.
/// Render the assignment set listing: per-question parts plus re-probe
/// targets. Pure text (the `print_*` twin writes it).
fn format_assignment_set(items: &[store::AssignmentQuestion]) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let _ = writeln!(out, "Assignment set: {} question(s)", items.len());
    for item in items {
        let position = item.position.saturating_add(1);
        let parts: Vec<String> = serde_json::from_str(&item.parts_json).unwrap_or_default();
        let _ = writeln!(out, "\n{position}. [{}]", item.kind);
        for part in &parts {
            let _ = writeln!(out, "   {part}");
        }
        let targets: Vec<i64> =
            serde_json::from_str(&item.target_misconception_ids).unwrap_or_default();
        if !targets.is_empty() {
            let _ = writeln!(out, "   (re-probes misconceptions: {targets:?})");
        }
    }
    out
}

fn print_assignment_set(items: &[store::AssignmentQuestion]) {
    print!("{}", format_assignment_set(items));
}

/// Split a rubric file into question text + frozen rubric. Accepts either a
/// bare stored rubric (`criteria`, `max_score`, `model_solution` — the exact
/// `rubric_json` shape) or a full question export (`parts` + `rubric`) so a
/// stored assignment row can be graded by dumping its columns to JSON.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the file is not either shape.
fn split_rubric_file(text: &str) -> Result<(String, grading::Rubric)> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::InvalidInput(format!("rubric file is not valid JSON: {e}")))?;
    if let Some(inner) = value.get("rubric") {
        let rubric = grading::parse_rubric_json(&inner.to_string())?;
        let parts: Vec<String> = value
            .get("parts")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(ToString::to_string))
                    .collect()
            })
            .unwrap_or_default();
        return Ok((parts.join("\n"), rubric));
    }
    Ok((String::new(), grading::parse_rubric_json(text)?))
}

/// `cadence dev grade`: isolated rubric-grading stage (§2.1, §7.3, §10).
/// Reads the student answer (`--answers` markdown) and the frozen rubric
/// (`--rubric` JSON: a stored `rubric_json` or a `parts`+`rubric` question
/// export — never a fresh rubric), grades via `complete_cached` (validated
/// before caching), and prints the §10 verdict with per-criterion scores.
/// Cache lives in `.scratch/dev-grade/`; never touches `~/.cadence/`.
fn run_dev_grade(answers_path: &str, rubric_path: &str) -> Result<()> {
    let answer = std::fs::read_to_string(answers_path)?;
    let rubric_text = std::fs::read_to_string(rubric_path)?;
    let (question, rubric) = split_rubric_file(&rubric_text)?;
    let prompt = grading::build_grading_prompt(&question, &rubric, &answer);
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = grading::GRADING_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &grading::grade_response_schema(&rubric),
        "grade",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = grading::grade_params_json_for(&rubric);
    let source_hash = grading::grade_source_hash(&question, &rubric_text, &answer);
    let db_path = std::path::PathBuf::from(".scratch/dev-grade/cadence.db");
    let lock_path = std::path::PathBuf::from(".scratch/dev-grade/cadence.lock");
    let mut store = SqliteStore::open(&db_path, &lock_path)?;
    println!("dev grade: answers={answers_path} rubric={rubric_path}");
    let today = today_date().format("%Y-%m-%d").to_string();
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today.as_str(),
    };
    let operation = "grade".to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let rubric_ref = &rubric;
    let validate =
        |text: &str| grading::validate_grade(text, rubric_ref).map(|_| text.trim().to_string());
    let result = llm::complete_cached(&provider, &mut store, &request, &validate, &hooks)?;
    println!(
        "dev grade: graded via LLM (cache: {}, transport sends: {})",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let grade = grading::validate_grade(&result.text, &rubric)?;
    println!(
        "\nVerdict: {} — score {}/{}",
        grade.classification.as_str(),
        grade.score,
        rubric.max_score
    );
    for criterion in &grade.criteria_results {
        println!(
            "  {}: {}/{} — {}",
            criterion.name, criterion.score, criterion.max_score, criterion.comment
        );
    }
    println!("Feedback: {}", grade.feedback);
    if grade.classification == grading::GradeClass::QuestionDefective {
        println!(
            "(QUESTION_DEFECTIVE never penalizes the user: full credit, item flagged for replacement.)"
        );
    }
    Ok(())
}

/// `cadence dev notes`: isolated notes-synthesis stage against a fixture PDF
/// (§2.1, §11). Resolves the first unit, feeds optional `--misconceptions`
/// JSON plus the chapter text through `complete_cached` (validated before
/// caching), persists the document with `save_note`, and prints it. The
/// isolated run has no graded answers, so the mistake/demonstrated sections
/// say so briefly. In-memory store; never touches `~/.cadence/`.
fn run_dev_notes(pdf_path: &str, misconceptions_path: Option<&str>) -> Result<()> {
    let first = extract_first_unit(pdf_path, 1)?;
    let unit = first.unit;
    let mut store = MemoryStore::new();
    let store_ref: &mut dyn Store = &mut store;
    println!("dev notes: pdf={} start_page=1", first.pdf_name);
    let chapter = dev_mcq::ensure_dev_chapter(store_ref, &first.pdf_hash, &first.pdf_name, &unit)?;
    let misconceptions: Vec<notes::MisconceptionItem> = match misconceptions_path {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            notes::parse_misconceptions_file(&text)?
        }
        None => Vec::new(),
    };
    let grades: Vec<notes::GradeSummary> = Vec::new();
    let existing = store_ref.list_notes(chapter.id, dev_mcq::DEV_ATTEMPT_NO)?;
    let note = if let Some(first) = existing.into_iter().next() {
        println!(
            "dev notes: resumed stored notes ({} chars)",
            first.content_markdown.chars().count()
        );
        first
    } else {
        let prompt = notes::build_notes_prompt(&unit, &misconceptions, &grades);
        let mut config = llm::LlmConfig::from_env()?;
        config.max_tokens = notes::NOTES_MAX_TOKENS;
        config.response_format_json = Some(llm::response_format_envelope(
            &notes::notes_response_schema(),
            "notes",
        ));
        let provider = llm::HttpLlmProvider::new(config)?;
        let params = notes::notes_params_json();
        let source_hash = notes::notes_source_hash_for(&unit.text, &misconceptions, &grades);
        let today = today_date().format("%Y-%m-%d").to_string();
        let hooks = llm::RunHooks {
            sleep: &std::thread::sleep,
            now_iso: today.as_str(),
        };
        let operation = "notes".to_string();
        let request = llm::CachedRequest {
            operation: operation.as_str(),
            prompt: prompt.as_str(),
            source_hash: source_hash.as_str(),
            params_json: params.as_str(),
        };
        let validate = |text: &str| notes::validate_notes(text).map(|_| text.trim().to_string());
        let result = llm::complete_cached(&provider, store_ref, &request, &validate, &hooks)?;
        println!(
            "dev notes: generated via LLM (cache: {}, transport sends: {})",
            if result.cache_hit { "hit" } else { "miss" },
            result.transport_calls
        );
        let validated = notes::validate_notes(&result.text)?;
        let row = notes::to_new_note(chapter.id, dev_mcq::DEV_ATTEMPT_NO, &validated, &today);
        store_ref.save_note(&row)?
    };
    println!("\n{}", note.content_markdown);
    Ok(())
}

/// One disputed answer with everything the §9 auditor needs (step 1):
/// chapter provenance, frozen rubric, student answer, and original grade.
struct DisputeCase {
    /// Owning chapter id (the `assignment_id` in `cadence dispute`).
    chapter_id: i64,
    /// Stored assignment question (re-probe targets for the §9 purge).
    question: store::AssignmentQuestion,
    /// Joined question parts.
    question_text: String,
    /// Stored `rubric_json` (cache identity uses the exact stored text).
    rubric_json: String,
    /// Parsed frozen rubric.
    rubric: grading::Rubric,
    /// Latest recorded answer.
    answer: String,
    /// Latest grade of record.
    grade: store::Grade,
    /// One-line grade summary embedded in the audit prompt.
    grade_summary: String,
    /// Chapter source excerpt when the unit file loads; otherwise `None` and
    /// the auditor judges against the frozen rubric alone.
    source_excerpt: Option<String>,
}

/// Load the disputed case (§9 step 1): question, latest answer, and latest
/// grade for question `question_no` (1-based) in the chapter's current
/// attempt. Every absence fails loudly — disputing an unanswered or ungraded
/// question is a usage error, never a silent default.
///
/// # Errors
///
/// Returns [`Error::NotFound`] for unknown chapters, [`Error::InvalidInput`]
/// for out-of-range questions and missing answers/grades.
fn load_dispute_case(
    store: &dyn Store,
    chapter: &domain::Chapter,
    question_no: usize,
) -> Result<DisputeCase> {
    let position = question_no
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidInput("question numbers start at 1".to_string()))?;
    let questions = store.list_assignment_questions(chapter.id, chapter.attempt_no)?;
    if questions.is_empty() {
        return Err(Error::InvalidInput(format!(
            "chapter {} has no assignment questions on attempt {}",
            chapter.id, chapter.attempt_no
        )));
    }
    let total = questions.len();
    let selected = questions.get(position).ok_or_else(|| {
        Error::InvalidInput(format!("question {question_no} out of range (1-{total})"))
    })?;
    let responses = store.list_assignment_responses(selected.id)?;
    let answer = responses.last().ok_or_else(|| {
        Error::InvalidInput(format!(
            "question {question_no} has no recorded answer yet — answer it first"
        ))
    })?;
    let grades = store.list_grades_for_question(selected.id)?;
    let grade = grades.last().cloned().ok_or_else(|| {
        Error::InvalidInput(format!(
            "question {question_no} has no recorded grade yet — grade it first"
        ))
    })?;
    let rubric = grading::parse_rubric_json(&selected.rubric_json)?;
    let parts: Vec<String> = serde_json::from_str(&selected.parts_json).unwrap_or_default();
    let source_excerpt = ingest::load_unit_text(std::path::Path::new(&chapter.file_path))
        .ok()
        .map(|unit| {
            unit.text
                .chars()
                .take(dispute::SOURCE_EXCERPT_CHARS)
                .collect::<String>()
        });
    Ok(DisputeCase {
        chapter_id: chapter.id,
        question: selected.clone(),
        question_text: parts.join("\n"),
        rubric_json: selected.rubric_json.clone(),
        rubric,
        answer: answer.answer_text.clone(),
        grade_summary: format!(
            "classification={} score={}/{} feedback={}",
            grade.classification, grade.score, grade.max_score, grade.feedback
        ),
        grade,
        source_excerpt,
    })
}

/// Resolve the dispute text: `--text` wins, then piped stdin, then an
/// interactive one-line prompt on a terminal (so the picker flow never dead
/// ends). Blank text is always a usage error.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the text is blank,
/// [`Error::Io`] when stdin cannot be read.
fn read_dispute_text(explicit: Option<&str>) -> Result<String> {
    if let Some(text) = explicit {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "dispute text must not be blank".to_string(),
            ));
        }
        return Ok(text.trim().to_string());
    }
    if std::io::stdin().is_terminal() {
        print!("Dispute text (one line) > ");
        // A flush failure never blocks the read; stdin still holds the line.
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line)?;
        if line.trim().is_empty() {
            return Err(Error::InvalidInput(
                "dispute text must not be blank".to_string(),
            ));
        }
        return Ok(line.trim().to_string());
    }
    let mut piped = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut piped)?;
    if piped.trim().is_empty() {
        return Err(Error::InvalidInput(
            "dispute text must not be blank".to_string(),
        ));
    }
    Ok(piped.trim().to_string())
}

/// One graded assignment question for the dispute picker: 1-based position,
/// kind label, and latest grade of record.
struct DisputableQuestion {
    /// 1-based position within the chapter's assignment set.
    position: usize,
    /// Stored kind label (`written` / `coding`).
    kind: String,
    /// Latest grade of record.
    score: i64,
    /// Rubric total.
    max_score: i64,
    /// Latest §10 verdict label.
    classification: String,
}

/// One chapter with graded assignment questions on its current attempt.
struct DisputableChapter {
    /// Owning chapter.
    chapter: domain::Chapter,
    /// Graded questions in position order.
    questions: Vec<DisputableQuestion>,
}

/// Chapters holding at least one graded assignment question on the current
/// attempt, in id order — the `cadence dispute` picker source. Chapters
/// without grades are not disputable and never listed.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn disputable_chapters(store: &dyn Store) -> Result<Vec<DisputableChapter>> {
    let mut out = Vec::new();
    for book in store.list_books()? {
        for chapter in store.list_chapters(book.id)? {
            let mut questions = Vec::new();
            for (index, question) in store
                .list_assignment_questions(chapter.id, chapter.attempt_no)?
                .iter()
                .enumerate()
            {
                let grades = store.list_grades_for_question(question.id)?;
                let Some(grade) = grades.last() else {
                    continue;
                };
                questions.push(DisputableQuestion {
                    position: index.saturating_add(1),
                    kind: question.kind.clone(),
                    score: grade.score,
                    max_score: grade.max_score,
                    classification: grade.classification.clone(),
                });
            }
            if !questions.is_empty() {
                out.push(DisputableChapter { chapter, questions });
            }
        }
    }
    Ok(out)
}

/// Parse a `1`-based picker choice against `count` options.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on blank, non-numeric, or out-of-range
/// input (never panics on hostile stdin).
fn parse_pick(input: &str, count: usize) -> Result<usize> {
    let trimmed = input.trim();
    let choice: usize = trimmed
        .parse()
        .map_err(|_| Error::InvalidInput(format!("pick a number 1-{count}, got '{trimmed}'")))?;
    if choice < 1 || choice > count {
        return Err(Error::InvalidInput(format!(
            "pick a number 1-{count}, got '{trimmed}'"
        )));
    }
    Ok(choice)
}

/// Read one picker choice from interactive stdin.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when stdin is not a terminal (picking
/// needs a human) or the choice does not parse.
fn prompt_pick(label: &str, count: usize) -> Result<usize> {
    if std::io::stdin().is_terminal() {
        print!("{label} [1-{count}] > ");
        // A flush failure never blocks the read; stdin still holds the line.
        let _ = std::io::Write::flush(&mut std::io::stdout());
    } else {
        return Err(Error::InvalidInput(
            "no selection given — pass explicit ids or run on a terminal".to_string(),
        ));
    }
    let mut line = String::new();
    std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line)?;
    parse_pick(&line, count)
}

/// Resolve the dispute target: explicit ids pass through untouched; a missing
/// chapter or question opens the interactive picker over graded assignments.
///
/// # Errors
///
/// Returns [`Error::NotFound`] for unknown chapters (via the picker index),
/// [`Error::InvalidInput`] when nothing is graded yet or stdin is not
/// interactive.
fn resolve_dispute_target(
    store: &dyn Store,
    assignment_id: Option<i64>,
    question_no: Option<usize>,
) -> Result<(i64, usize)> {
    let candidates = disputable_chapters(store)?;
    if candidates.is_empty() {
        return Err(Error::InvalidInput(
            "no graded assignment questions yet — grade an assignment first".to_string(),
        ));
    }
    let chapter_index = if let Some(id) = assignment_id {
        candidates
            .iter()
            .position(|c| c.chapter.id == id)
            .ok_or_else(|| {
                Error::NotFound(format!("assignment (chapter) {id} has no graded questions"))
            })?
    } else {
        println!("Graded assignments:");
        for (index, candidate) in candidates.iter().enumerate() {
            let total: i64 = candidate.questions.iter().map(|q| q.max_score).sum();
            let earned: i64 = candidate.questions.iter().map(|q| q.score).sum();
            println!(
                "  {}. ch{} '{}' — {earned}/{total} across {} question(s)",
                index.saturating_add(1),
                candidate.chapter.id,
                candidate.chapter.title,
                candidate.questions.len(),
            );
        }
        prompt_pick("Dispute which chapter?", candidates.len())?.saturating_sub(1)
    };
    let selected = candidates
        .get(chapter_index)
        .ok_or_else(|| Error::InvalidInput("choice out of range".to_string()))?;
    let position = if let Some(number) = question_no {
        if !selected.questions.iter().any(|q| q.position == number) {
            return Err(Error::InvalidInput(format!(
                "question {number} is not graded in chapter {} (1-{})",
                selected.chapter.id,
                selected.questions.len()
            )));
        }
        number
    } else {
        println!("Graded questions in '{}':", selected.chapter.title);
        for (index, question) in selected.questions.iter().enumerate() {
            println!(
                "  {}. Q{} {} — {}/{} {}",
                index.saturating_add(1),
                question.position,
                question.kind,
                question.score,
                question.max_score,
                question.classification,
            );
        }
        let choice = prompt_pick("Dispute which question?", selected.questions.len())?;
        selected
            .questions
            .get(choice.saturating_sub(1))
            .map(|q| q.position)
            .ok_or_else(|| Error::InvalidInput("choice out of range".to_string()))?
    };
    Ok((selected.chapter.id, position))
}

/// Purge misconceptions invalidated by a successful dispute (§9 step 3): the
/// `ASSIGNMENT` row the bad grade logged plus the re-probed targets it
/// nudged down are marked `DISPUTED` with confidence preserved (see
/// [`dispute::select_dispute_purge_ids`] for the exact set). `UPHELD` purges
/// nothing. Corrupt `target_misconception_ids` are reported and skipped on
/// the target side — the logged-row side still purges. Returns rows updated.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn apply_dispute_purge(
    store: &mut dyn Store,
    question: &store::AssignmentQuestion,
    action: dispute::DisputeAction,
    today: &str,
) -> Result<usize> {
    if !dispute::should_purge_misconceptions(action) {
        return Ok(0);
    }
    if serde_json::from_str::<Vec<i64>>(&question.target_misconception_ids).is_err() {
        println!(
            "Question {} has corrupt re-probe targets — purging only its logged assignment row.",
            question.position.saturating_add(1)
        );
    }
    let rows = store.list_misconceptions(question.chapter_id)?;
    let ids = dispute::select_dispute_purge_ids(
        &question.target_misconception_ids,
        question.position,
        &rows,
    );
    let mut purged = 0_usize;
    for row in &rows {
        if !ids.contains(&row.id) {
            continue;
        }
        store.update_misconception(
            row.id,
            row.confidence,
            "DISPUTED",
            today,
            row.resolved_at.as_deref(),
        )?;
        purged = purged.saturating_add(1);
        println!(
            "Misconception purged (disputed): {}",
            row.concept_description
        );
    }
    Ok(purged)
}

/// `cadence dispute <assignment_id> --question <N>` (§9): fresh independent
/// audit of one graded answer, grade correction in SQLite with the original
/// preserved, an inspectable `disputes` trail row, and the §9 step-3 purge —
/// on `REVISED`/`QUESTION_DEFECTIVE` the `ASSIGNMENT` row the bad grade
/// logged and the re-probed targets it nudged down are marked `DISPUTED`.
///
/// `assignment_id` is the chapter whose assignment set holds the question;
/// `N` is the 1-based position within that set.
///
/// # Errors
///
/// Propagates lookup failures from [`load_dispute_case`], LLM failures from
/// `complete_cached`, and persistence failures from the store.
/// Validated dispute audit plus the transport artifacts needed to persist it.
struct DisputeAuditOutcome {
    /// Parsed audit decision.
    audit: dispute::DisputeResult,
    /// Raw auditor response JSON (stored on the dispute row).
    response_text: String,
    /// Auditor model id (stored on the dispute row).
    model: String,
    /// Audit date (`YYYY-MM-DD`).
    today: String,
}

/// Run the dispute audit through the cached LLM transport: prompt → durable
/// job → validated decision. Prints the transport summary.
///
/// # Errors
///
/// Propagates LLM, validation, and [`Error::Store`] failures.
fn audit_dispute(
    store: &mut SqliteStore,
    case: &DisputeCase,
    question_no: usize,
    dispute_text: &str,
) -> Result<DisputeAuditOutcome> {
    let prompt = dispute::build_dispute_prompt(
        case.question_text.as_str(),
        &case.rubric,
        case.answer.as_str(),
        case.grade_summary.as_str(),
        dispute_text,
        case.source_excerpt.as_deref(),
    );
    let mut config = llm::LlmConfig::from_env()?;
    config.max_tokens = dispute::DISPUTE_MAX_TOKENS;
    config.response_format_json = Some(llm::response_format_envelope(
        &dispute::dispute_response_schema(case.rubric.max_score),
        "dispute",
    ));
    let provider = llm::HttpLlmProvider::new(config)?;
    let params = dispute::dispute_params_json_for(case.rubric.max_score);
    let source_hash = dispute::dispute_source_hash(
        case.question_text.as_str(),
        case.rubric_json.as_str(),
        case.answer.as_str(),
        case.grade_summary.as_str(),
        dispute_text,
    );
    let today = today_date().format("%Y-%m-%d").to_string();
    let hooks = llm::RunHooks {
        sleep: &std::thread::sleep,
        now_iso: today.as_str(),
    };
    let operation = "dispute".to_string();
    let request = llm::CachedRequest {
        operation: operation.as_str(),
        prompt: prompt.as_str(),
        source_hash: source_hash.as_str(),
        params_json: params.as_str(),
    };
    let max_score = case.rubric.max_score;
    let original_score = case.grade.score;
    let validate = |text: &str| {
        dispute::validate_dispute(text, max_score, original_score).map(|_| text.trim().to_string())
    };
    println!(
        "dispute: auditing assignment {} question {question_no} (grade {}: {} {}/{})",
        case.chapter_id,
        case.grade.id,
        case.grade.classification,
        case.grade.score,
        case.grade.max_score,
    );
    let result = llm::complete_cached(&provider, store, &request, &validate, &hooks)?;
    println!(
        "dispute: audited via LLM (cache: {}, transport sends: {})",
        if result.cache_hit { "hit" } else { "miss" },
        result.transport_calls
    );
    let audit = dispute::validate_dispute(&result.text, max_score, original_score)?;
    Ok(DisputeAuditOutcome {
        audit,
        response_text: result.text,
        model: llm::RawTransport::model_id(&provider).to_string(),
        today,
    })
}

/// Render the dispute verdict footer: purge notice, score line, explanation,
/// and whether the original grade stands. Pure text (the `print_*` twin
/// writes it).
fn format_dispute_verdict(
    audit: &dispute::DisputeResult,
    dispute_id: i64,
    max_score: i64,
    original_score: i64,
    purged: usize,
) -> String {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    if purged > 0 {
        let _ = writeln!(
            out,
            "Misconceptions purged: {purged} row(s) marked DISPUTED (bad-grade evidence withdrawn)."
        );
    }
    let _ = writeln!(
        out,
        "\nVerdict: {} — final {}/{} (was {}/{})",
        audit.action.as_str(),
        audit.final_score,
        max_score,
        original_score,
        max_score,
    );
    let _ = writeln!(out, "Explanation: {}", audit.explanation);
    if audit.dispute_valid {
        let _ = writeln!(
            out,
            "Grade corrected in SQLite (dispute row {dispute_id}; original {original_score} preserved)."
        );
    } else {
        let _ = writeln!(
            out,
            "Original grade stands (dispute row {dispute_id} recorded)."
        );
    }
    if audit.action == dispute::DisputeAction::QuestionDefective {
        let _ = writeln!(
            out,
            "(QUESTION_DEFECTIVE never penalizes the user: full credit, item flagged for replacement.)"
        );
    }
    out
}

/// Print the dispute verdict footer: purge notice, score line, explanation,
/// and whether the original grade stands.
fn print_dispute_verdict(
    audit: &dispute::DisputeResult,
    dispute_id: i64,
    max_score: i64,
    original_score: i64,
    purged: usize,
) {
    print!(
        "{}",
        format_dispute_verdict(audit, dispute_id, max_score, original_score, purged)
    );
}

fn run_dispute(
    store: &mut SqliteStore,
    assignment_id: i64,
    question_no: usize,
    text: Option<&str>,
) -> Result<()> {
    let chapter = store
        .get_chapter(assignment_id)
        .map_err(|_| Error::NotFound(format!("assignment (chapter) {assignment_id}")))?;
    let case = load_dispute_case(store, &chapter, question_no)?;
    let dispute_text = read_dispute_text(text)?;
    let outcome = audit_dispute(store, &case, question_no, dispute_text.as_str())?;
    let max_score = case.rubric.max_score;
    let original_score = case.grade.score;
    let trail = store.record_dispute(&store::NewDispute {
        grade_id: case.grade.id,
        text: dispute_text,
        decision: outcome.audit.action.as_str().to_string(),
        final_score: outcome.audit.final_score,
        adjudication_json: outcome.response_text,
        adjudicator_model: outcome.model,
        timestamp: outcome.today.clone(),
    })?;
    let purged = apply_dispute_purge(store, &case.question, outcome.audit.action, &outcome.today)?;
    print_dispute_verdict(&outcome.audit, trail.id, max_score, original_score, purged);
    Ok(())
}

/// Collect closed-book answers in `nvim` (§7.2): one editor buffer per
/// unanswered question, answer below the divider, `:wq` saves and advances.
/// Blank answers prompt once (`intentional?`); `:q!` without writing leaves
/// the question pending. Returns whether every question is answered.
fn run_assignment_answers(
    store: &mut dyn Store,
    items: &[store::AssignmentQuestion],
    attempt_no: i64,
) -> Result<bool> {
    let today = today_date().format("%Y-%m-%d").to_string();
    let total = items.len();
    for item in items {
        if !store.list_assignment_responses(item.id)?.is_empty() {
            continue;
        }
        let position = item.position.saturating_add(1);
        let parts: Vec<String> = serde_json::from_str(&item.parts_json).unwrap_or_default();
        let tmp = std::env::temp_dir().join(format!("cadence-assignment-{}.md", item.id));
        // Keep an existing draft (crash recovery); otherwise write the
        // template. The file lives until the answer is persisted (§16).
        if !tmp.exists() {
            let mut template = format!("# Question {position}/{total} [{}]\n\n", item.kind);
            for part in &parts {
                template.push_str(part);
                template.push('\n');
            }
            template.push('\n');
            template.push_str(ANSWER_DIVIDER);
            template.push('\n');
            std::fs::write(&tmp, template)?;
        }
        let answered = loop {
            println!(
                "\nQuestion {position}/{total} [{}] — opening nvim (:wq saves, :q! skips).",
                item.kind
            );
            let status = Command::new("nvim").arg(&tmp).status()?;
            if !status.success() {
                println!("Editor exited without saving — question left pending.");
                break false;
            }
            let buffer = std::fs::read_to_string(&tmp)?;
            let Some(answer) = buffer
                .split_once(ANSWER_DIVIDER)
                .map(|(_, after)| after.trim().to_string())
            else {
                println!("Divider deleted — reopening so the answer lands below it.");
                continue;
            };
            if answer.is_empty() {
                match ask_terminal("You left this blank — intentional? [y/n] > ")? {
                    Some(true) => {}
                    Some(false) => continue,
                    None => {
                        println!("\nSession saved — rerun to resume.");
                        break false;
                    }
                }
            }
            store.record_assignment_response(item.id, &answer, &today, attempt_no)?;
            let _ = std::fs::remove_file(&tmp);
            println!("Answer saved ({} chars).", answer.chars().count());
            break true;
        };
        if !answered {
            let mut done = 0_usize;
            for other in items {
                if !store.list_assignment_responses(other.id)?.is_empty() {
                    done = done.saturating_add(1);
                }
            }
            println!("Session saved — rerun to resume (answered {done}/{total}).");
            return Ok(false);
        }
    }
    Ok(true)
}

/// Running score totals for one MCQ session.
#[derive(Default)]
struct SessionTotals {
    /// Correct answers so far.
    correct: usize,
    /// Answered questions so far.
    answered: usize,
    /// Misconceptions logged so far.
    misconceptions: usize,
}

/// Fold a previously answered item into the running totals. Returns whether
/// the item was already answered (the caller skips it).
///
/// # Errors
///
/// Propagates [`Error::Store`] from response listing.
fn tally_prior_answer(
    store: &dyn Store,
    item: &store::McqItem,
    totals: &mut SessionTotals,
) -> Result<bool> {
    if !dev_mcq::is_answered(store, item.id)? {
        return Ok(false);
    }
    let prior = store.list_mcq_responses(item.id)?;
    if prior.iter().any(|r| r.is_correct) {
        totals.correct = totals.correct.saturating_add(1);
    }
    totals.answered = totals.answered.saturating_add(1);
    Ok(true)
}

/// Build the early-exit summary: print progress and return the incomplete
/// session (rerun resumes).
fn exit_session_early(items_len: usize, totals: &SessionTotals) -> SessionSummary {
    println!(
        "\nSession saved — rerun to resume (answered {}/{items_len}).",
        totals.answered
    );
    print_mcq_score(totals.answered, totals.misconceptions);
    SessionSummary {
        answered: totals.answered,
        correct: totals.correct,
        misconceptions: totals.misconceptions,
        completed: false,
    }
}

/// Interactive A–E loop with immediate feedback, trap labels, misconception
/// logging (retest only), and resume (answered items are skipped). `Ctrl-C` /
/// `Ctrl-D` (EOF) exits cleanly with persistence; rerun resumes.
fn run_mcq_session(
    store: &mut dyn Store,
    items: &[store::McqItem],
    unit: &engines::UnitText,
    phase: mcq::McqPhase,
    chapter_id: i64,
    attempt_no: i64,
    seed: Option<u64>,
) -> Result<SessionSummary> {
    let base_seed = seed.unwrap_or_else(mcq::random_seed);
    let today = today_date().format("%Y-%m-%d").to_string();
    let mut totals = SessionTotals::default();
    for (position, item) in items.iter().enumerate() {
        let Some(number) = position.checked_add(1) else {
            continue;
        };
        if tally_prior_answer(store, item, &mut totals)? {
            continue;
        }
        let row = dev_mcq::validated_from_row(
            &dev_mcq::McqRowView {
                question: &item.question_text,
                options_json: &item.options_json,
                correct_index: item.correct_index,
                trap_index: item.trap_index,
                explanation: &item.explanation_text,
                topic: &item.topic,
                source_refs: &item.source_refs,
            },
            unit,
        )?;
        let shown = mcq::apply_shuffle(&row, dev_mcq::shuffle_seed_for(base_seed, position))
            .ok_or_else(|| Error::Store("stored MCQ failed shuffle".to_string()))?;
        let Some(selected) = ask_answer(number, items.len(), &row, &shown)? else {
            return Ok(exit_session_early(items.len(), &totals));
        };
        let outcome = record_answer(
            store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase,
                row: &row,
                shown: &shown,
                selected,
                attempt_no,
                today: &today,
            },
        )?;
        totals.answered = totals.answered.saturating_add(1);
        if outcome.is_correct {
            totals.correct = totals.correct.saturating_add(1);
        }
        if outcome.misconception_logged {
            totals.misconceptions = totals.misconceptions.saturating_add(1);
        }
    }
    println!(
        "\nSession complete: {}/{} correct.",
        totals.correct,
        items.len()
    );
    print_mcq_score(totals.answered, totals.misconceptions);
    Ok(SessionSummary {
        answered: totals.answered,
        correct: totals.correct,
        misconceptions: totals.misconceptions,
        completed: true,
    })
}

/// Display one question and read the answer. Returns `None` on EOF (clean
/// exit; state already persisted up to the previous question).
///
/// The topic stays hidden until after answering: it often names the concept
/// under test and would give away the solution.
fn ask_answer(
    number: usize,
    total: usize,
    row: &mcq::ValidatedMcq,
    shown: &mcq::DisplayedMcq,
) -> Result<Option<usize>> {
    use std::io::Write as _;
    println!("\n{}", question_header(number, total, colors_enabled()));
    println!("{}", row.question);
    for (index, option) in shown.displayed_options.iter().enumerate() {
        println!("   {}) {option}", dev_mcq::option_label(index));
    }
    println!(
        "   {}) {}",
        dev_mcq::option_label(mcq::IDK_INDEX),
        dev_mcq::idk_label()
    );
    loop {
        print!("Your answer [A/B/C/D/E] > ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        let bytes = std::io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None);
        }
        if let Some(index) = dev_mcq::parse_answer(&line) {
            return Ok(Some(index));
        }
        println!("Please answer A, B, C, D, or E.");
    }
}

/// Outcome of grading one answer (feedback already printed).
struct AnswerOutcome {
    /// Whether the selection was correct.
    is_correct: bool,
    /// Whether a misconception row was logged (retest wrong non-E only).
    misconception_logged: bool,
}

/// ANSI color codes for verdict feedback: green for correct, red for its
/// counterpart.
const GREEN_CODE: &str = "\x1b[32m";
const RED_CODE: &str = "\x1b[31m";
const RESET_CODE: &str = "\x1b[0m";
/// Bold for question headers (same `NO_COLOR` gating as verdict colors).
const BOLD_CODE: &str = "\x1b[1m";

/// Wrap `text` in an ANSI color code unless `enabled` is false (plain text
/// for tests and `NO_COLOR` environments).
fn paint(text: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("{code}{text}{RESET_CODE}")
    } else {
        text.to_string()
    }
}

/// Whether verdict colors apply: suppressed only under `NO_COLOR`.
fn colors_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none()
}

/// High-visibility question header: the counter gets its own ruled line so
/// `Question 3/8` scans even between long explanations. Pure for tests;
/// callers pass [`colors_enabled`].
#[must_use]
fn question_header(number: usize, total: usize, enabled: bool) -> String {
    paint(
        &format!("─── Question {number}/{total} ───"),
        BOLD_CODE,
        enabled,
    )
}

/// Boost confidence on open misconception rows matching a correct answer's
/// topic (§12: correct retest increases it). Only `ACTIVE`/`IMPROVING` rows
/// move — resolved rows never reopen here, and disputed rows belong to the
/// §9 purge. Returns rows updated.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn boost_matching_misconceptions(
    store: &mut dyn Store,
    chapter_id: i64,
    topic: &str,
    today: &str,
) -> Result<usize> {
    let mut boosted = 0_usize;
    for row in store.list_misconceptions(chapter_id)? {
        if row.concept_description != topic {
            continue;
        }
        if row.status != "ACTIVE" && row.status != "IMPROVING" {
            continue;
        }
        let step = misconceptions::apply_outcome(row.confidence, &row.status, true, false);
        let resolved_at = if step.just_resolved {
            Some(today)
        } else {
            row.resolved_at.as_deref()
        };
        store.update_misconception(row.id, step.confidence, step.status, today, resolved_at)?;
        boosted = boosted.saturating_add(1);
        if step.just_resolved {
            println!("Misconception resolved: {}", row.concept_description);
        } else {
            println!(
                "Misconception improving: {} (confidence {:.2} → {:.2})",
                row.concept_description, row.confidence, step.confidence
            );
        }
    }
    Ok(boosted)
}

/// Nudge confidence down on open misconception rows matching a wrong review
/// answer's topic (§12: wrong retest evidence decreases it). Only
/// `ACTIVE`/`IMPROVING` rows move — resolved rows never reopen here, and
/// disputed rows belong to the §9 purge. `resolved_at` is always preserved:
/// a downward step never resolves, so any `RESOLVED` outcome here would mean
/// a corrupt status/confidence pair and must not mint a resolution stamp.
/// Returns rows updated.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn nudge_matching_misconceptions(
    store: &mut dyn Store,
    chapter_id: i64,
    topic: &str,
    today: &str,
) -> Result<usize> {
    let mut nudged = 0_usize;
    for row in store.list_misconceptions(chapter_id)? {
        if row.concept_description != topic {
            continue;
        }
        if row.status != "ACTIVE" && row.status != "IMPROVING" {
            continue;
        }
        let step = misconceptions::apply_outcome(row.confidence, &row.status, false, false);
        store.update_misconception(
            row.id,
            step.confidence,
            step.status,
            today,
            row.resolved_at.as_deref(),
        )?;
        nudged = nudged.saturating_add(1);
        println!(
            "Misconception persists: {} (confidence {:.2} → {:.2})",
            row.concept_description, row.confidence, step.confidence
        );
    }
    Ok(nudged)
}

/// Record one answer, print immediate feedback, and log a misconception when
/// [`mcq::should_log_misconception`] applies.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
/// One graded answer and its display context (one parameter, not eight).
struct Answer<'a> {
    /// Answered row id.
    item_id: i64,
    /// Owning chapter id.
    chapter_id: i64,
    /// Assessment phase.
    phase: mcq::McqPhase,
    /// Validated question + explanation.
    row: &'a mcq::ValidatedMcq,
    /// Shuffled presentation.
    shown: &'a mcq::DisplayedMcq,
    /// Chosen display index (`IDK_INDEX` = abstain).
    selected: usize,
    /// Attempt number (§4.1).
    attempt_no: i64,
    /// Session date (`YYYY-MM-DD`).
    today: &'a str,
}

/// Record one answer, print immediate feedback, and log a misconception when
/// [`mcq::should_log_misconception`] applies.
///
/// # Errors
///
/// Propagates [`Error::Store`] from the backend.
fn record_answer(store: &mut dyn Store, answer: &Answer<'_>) -> Result<AnswerOutcome> {
    let is_correct = answer.selected == answer.shown.displayed_correct;
    let selected_trap = answer.selected == answer.shown.displayed_trap;
    store.record_mcq_response(
        answer.item_id,
        i64::try_from(answer.selected).unwrap_or(4),
        is_correct,
        selected_trap,
        answer.today,
        answer.attempt_no,
    )?;
    if is_correct {
        return record_correct(store, answer);
    }
    let selected_text = if answer.selected == mcq::IDK_INDEX {
        dev_mcq::idk_label().to_string()
    } else {
        answer
            .shown
            .displayed_options
            .get(answer.selected)
            .cloned()
            .unwrap_or_default()
    };
    let correct_text = answer
        .shown
        .displayed_options
        .get(answer.shown.displayed_correct)
        .cloned()
        .unwrap_or_default();
    println!(
        "{}",
        paint(
            &format!(
                "Incorrect — correct answer: {}) {correct_text}",
                dev_mcq::option_label(answer.shown.displayed_correct)
            ),
            RED_CODE,
            colors_enabled()
        )
    );
    println!("[{}] {}", answer.row.topic, answer.row.explanation);
    if selected_trap {
        println!("TRAP DETECTED: {} — {selected_text}", answer.row.topic);
    }
    let logged = log_wrong_answer(store, answer, &selected_text, &correct_text, selected_trap)?;
    Ok(AnswerOutcome {
        is_correct,
        misconception_logged: logged,
    })
}

/// Feedback + lifecycle for a correct answer.
fn record_correct(store: &mut dyn Store, answer: &Answer<'_>) -> Result<AnswerOutcome> {
    println!("{}", paint("Correct.", GREEN_CODE, colors_enabled()));
    println!("[{}] {}", answer.row.topic, answer.row.explanation);
    if matches!(answer.phase, mcq::McqPhase::Retest | mcq::McqPhase::Review) {
        boost_matching_misconceptions(store, answer.chapter_id, &answer.row.topic, answer.today)?;
    }
    Ok(AnswerOutcome {
        is_correct: true,
        misconception_logged: false,
    })
}

/// Persist one fresh misconception row for a wrong answer and report it.
fn log_fresh_misconception(
    store: &mut dyn Store,
    answer: &Answer<'_>,
    selected_text: &str,
    correct_text: &str,
    selected_trap: bool,
) -> Result<()> {
    let (concept, description, evidence) = dev_mcq::misconception_texts(
        &answer.row.topic,
        &answer.row.question,
        selected_text,
        correct_text,
        selected_trap,
    );
    store.create_misconception(
        answer.chapter_id,
        &concept,
        &description,
        &evidence,
        "RETEST",
        answer.today,
    )?;
    println!("Misconception logged: {concept}");
    Ok(())
}

/// Misconception bookkeeping for a wrong answer: review nudges a matching
/// open row (logging fresh when nothing matches); other phases log via
/// [`mcq::should_log_misconception`]. `"I don't know"` never logs.
fn log_wrong_answer(
    store: &mut dyn Store,
    answer: &Answer<'_>,
    selected_text: &str,
    correct_text: &str,
    selected_trap: bool,
) -> Result<bool> {
    if answer.phase == mcq::McqPhase::Review {
        // Review re-probes an existing row: a wrong answer nudges the
        // topic-matched open rows down (−0.1) instead of logging a duplicate.
        // `"I don't know"` abstains (no signal). When no open row matches —
        // it resolved between generation and answering — the wrong answer is
        // still evidence, so it logs a fresh row like a retest would.
        if answer.selected != mcq::IDK_INDEX {
            let moved = nudge_matching_misconceptions(
                store,
                answer.chapter_id,
                &answer.row.topic,
                answer.today,
            )?;
            if moved > 0 {
                return Ok(true);
            }
            log_fresh_misconception(store, answer, selected_text, correct_text, selected_trap)?;
            return Ok(true);
        }
        return Ok(false);
    }
    if mcq::should_log_misconception(answer.phase, false, answer.selected) {
        log_fresh_misconception(store, answer, selected_text, correct_text, selected_trap)?;
        return Ok(true);
    }
    Ok(false)
}

/// Render the score footer shared by clean-EOF exits and full completions.
/// Pure text (the `print_*` twin writes it).
fn format_mcq_score(answered: usize, misconceptions: usize) -> String {
    format!("Answered: {answered}. Misconception updates this session: {misconceptions}.")
}

/// Score footer shared by clean-EOF exits and full completions.
fn print_mcq_score(answered: usize, misconceptions: usize) {
    println!("{}", format_mcq_score(answered, misconceptions));
}
/// Parse one `tasks[i]` fixture entry into a [`Task`]: ids derive from the
/// entry position; `status`/`chapter_id`/`attempt_no` default when absent.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] on missing fields or bad enums/dates.
fn parse_fixture_task(item: &serde_json::Value, index: usize, seq: i64) -> Result<Task> {
    let kind = item
        .get("task_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::InvalidInput(format!("task {index}: missing 'task_type'")))?;
    let scheduled = item
        .get("scheduled_for")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::InvalidInput(format!("task {index}: missing 'scheduled_for'")))?;
    let status_text = item
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("PENDING");
    let chapter_id = item
        .get("chapter_id")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(1);
    Ok(Task {
        id: seq,
        book_id: 1,
        chapter_id,
        task_type: TaskType::parse(kind)?,
        scheduled_for: parse_date(scheduled)?,
        status: TaskStatus::parse(status_text)?,
        completed_at: None,
        sequence: seq,
        attempt_no: item
            .get("attempt_no")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1),
    })
}

/// Render the dev-schedule report: executable queue plus the 3-day window
/// projection for the fixture date. Pure text (the `print_*` twin writes
/// it). The queue computation stays shared with production via
/// [`today_queue`].
///
/// # Errors
///
/// Propagates [`Error::Store`] from window projection.
fn format_dev_schedule_report(tasks: &[Task], today: NaiveDate) -> Result<String> {
    use std::fmt::Write as _;
    // `write!` on a `String` never fails; results discarded throughout.
    let mut out = String::new();
    let queue = today_queue(tasks, today);
    let _ = writeln!(
        out,
        "Fixture: {} task(s), today = {today}. Executable queue: {}",
        tasks.len(),
        queue.len()
    );
    for task in &queue {
        let _ = writeln!(
            out,
            "  [{}] chapter {} scheduled {} ({})",
            task.task_type.as_str(),
            task.chapter_id,
            task.scheduled_for,
            classify_label(task, today)
        );
    }
    let _ = writeln!(
        out,
        "pull_available: {}",
        if pull_available(tasks, today) {
            "yes"
        } else {
            "no"
        }
    );
    let window = project_window(2, today)?;
    let _ = writeln!(out, "3-day window projection (2 chapters):");
    for (date, row) in &window {
        let _ = writeln!(
            out,
            "  {date} ch+{} d+{}: {}",
            row.chapter_offset, row.day_offset, row.activity
        );
    }
    Ok(out)
}

/// Print the dev-schedule report: executable queue plus the 3-day window
/// projection for the fixture date.
///
/// # Errors
///
/// Propagates [`Error::Store`] from window projection.
fn print_dev_schedule_report(tasks: &[Task], today: NaiveDate) -> Result<()> {
    print!("{}", format_dev_schedule_report(tasks, today)?);
    Ok(())
}

/// Fixture shape: `{"today": "YYYY-MM-DD", "tasks": [{...}]}` where each task
/// has `task_type`, `scheduled_for`, `status`, `sequence`, `chapter_id`.
fn run_dev_schedule(fixture_path: &str) -> Result<()> {
    let text = std::fs::read_to_string(fixture_path)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::InvalidInput(e.to_string()))?;
    let today_text = parsed
        .get("today")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::InvalidInput("fixture needs 'today': 'YYYY-MM-DD'".to_string()))?;
    let today = parse_date(today_text)?;
    let raw_tasks = parsed
        .get("tasks")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::InvalidInput("fixture needs 'tasks': [...]".to_string()))?;
    let mut tasks = Vec::with_capacity(raw_tasks.len());
    for (index, item) in raw_tasks.iter().enumerate() {
        let Ok(seq) = i64::try_from(index).map(|i| i.saturating_add(1)) else {
            continue;
        };
        tasks.push(parse_fixture_task(item, index, seq)?);
    }
    print_dev_schedule_report(&tasks, today)
}

#[expect(
    clippy::too_many_lines,
    reason = "top-level dispatch; arms delegate and length is match-arm breadth"
)]
fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;
    let Some(command) = cli.command else {
        // Default entrypoint: today's scheduled loop.
        require_nvim()?;
        let mut store = open_production_store(&config)?;
        let today = today_date();
        let today_str = today.format("%Y-%m-%d").to_string();
        return run_daily_loop(&mut store, today, &today_str);
    };
    match command {
        Commands::Ingest(args) => {
            let Some(start_page) = args.start_page else {
                return Err(Error::InvalidInput(
                    "--start-page is mandatory (one-based physical page)".to_string(),
                ));
            };
            run_ingest(
                &args.pdf,
                start_page,
                args.max_unit_pages,
                args.title.as_deref(),
                args.manual_boundaries.as_deref(),
                args.chapter_level,
                &config,
            )
        }
        Commands::Today(args) => {
            let today = args
                .date
                .as_deref()
                .map_or_else(|| Ok(today_date()), parse_date)?;
            let mut store = open_production_store(&config)?;
            let today_str = today.format("%Y-%m-%d").to_string();
            let _ = ensure_all_books(&mut store, today, &today_str)?;
            let tasks = store.list_tasks()?;
            print_queue(&store, &tasks, today);
            Ok(())
        }
        Commands::Schedule(args) => {
            if args.days < 1 {
                return Err(Error::InvalidInput("--days must be >= 1".to_string()));
            }
            let mut store = open_production_store(&config)?;
            let today = today_date();
            let today_str = today.format("%Y-%m-%d").to_string();
            let _ = ensure_all_books(&mut store, today, &today_str)?;
            let tasks = store.list_tasks()?;
            print_queue(&store, &tasks, today);
            print_upcoming(&store, &tasks, today, args.days);
            Ok(())
        }
        Commands::Pull => {
            require_nvim()?;
            let mut store = open_production_store(&config)?;
            let today = today_date();
            let today_str = today.format("%Y-%m-%d").to_string();
            let _ = ensure_all_books(&mut store, today, &today_str)?;
            match pull_next(&mut store, today, &today_str) {
                Ok(Some((pulled, previous))) => {
                    let chapter = store.get_chapter(pulled.chapter_id)?;
                    println!(
                        "Pulled forward: Chapter '{}' — {} (was {previous}, now due {today}).",
                        chapter.title,
                        pulled.task_type.as_str()
                    );
                    println!("Pulled work is mandatory — pull again to cascade the pipeline.");
                    Ok(())
                }
                Ok(None) => {
                    println!("Mandatory work complete — no future pretest to pull.");
                    Ok(())
                }
                Err(Error::InvalidTransition(_)) => {
                    println!(
                        "Mandatory work remains — clear today's queue before pulling future work."
                    );
                    let tasks = store.list_tasks()?;
                    print_queue(&store, &tasks, today);
                    Ok(())
                }
                Err(other) => Err(other),
            }
        }
        Commands::Metrics => {
            let store = open_production_store(&config)?;
            run_metrics(&store, today_date())
        }
        Commands::Skip(args) => {
            let mut store = open_production_store(&config)?;
            let today_str = today_date().format("%Y-%m-%d").to_string();
            run_skip(&mut store, args.id, &today_str)
        }
        Commands::Unskip(args) => {
            let mut store = open_production_store(&config)?;
            let today = today_date();
            let today_str = today.format("%Y-%m-%d").to_string();
            run_unskip(&mut store, args.id, today, &today_str)
        }
        Commands::Misconceptions => {
            let store = open_production_store(&config)?;
            run_misconceptions(&store)
        }
        Commands::Notes(args) => {
            let store = open_production_store(&config)?;
            run_notes(&store, args.id)
        }
        Commands::Progress => {
            let store = open_production_store(&config)?;
            run_progress(&store)
        }
        Commands::Review => {
            let mut store = open_production_store(&config)?;
            let today_str = today_date().format("%Y-%m-%d").to_string();
            run_review(&mut store, &today_str)
        }
        Commands::Dispute(args) => {
            let mut store = open_production_store(&config)?;
            let (chapter_id, question_no) =
                resolve_dispute_target(&store, args.assignment_id, args.question)?;
            run_dispute(&mut store, chapter_id, question_no, args.text.as_deref())
        }
        Commands::Doctor => {
            let mut problems: u32 = 0;
            match Config::load() {
                Ok(cfg) => println!("config: OK ({})", cfg.db_path().display()),
                Err(e) => {
                    problems = problems.saturating_add(1);
                    println!("config: FAIL ({e})");
                }
            }
            match require_nvim() {
                Ok(()) => println!("nvim: OK"),
                Err(e) => {
                    problems = problems.saturating_add(1);
                    println!("nvim: FAIL ({e})");
                }
            }
            match open_production_store(&config) {
                Ok(_) => println!("db: OK (lock acquired, schema v{})", store::SCHEMA_VERSION),
                Err(e) => {
                    problems = problems.saturating_add(1);
                    println!("db: FAIL ({e})");
                }
            }
            println!("pdf: OK (mupdf 0.8.0, base14-fonts)");
            match llm::LlmConfig::from_env() {
                Err(_) => println!(
                    "llm: SKIP (no AI_GATEWAY_API_KEY, GOOGLE_API_KEY, or CADENCE_API_KEY/CADENCE_LLM_ENDPOINT configured)"
                ),
                Ok(cfg) => match llm::HttpLlmProvider::new(cfg.clone()).and_then(|p| p.check_api())
                {
                    Ok(()) => println!("llm: OK ({} reachable at {})", cfg.provider, cfg.endpoint),
                    Err(e) => {
                        problems = problems.saturating_add(1);
                        println!("llm: FAIL ({e})");
                    }
                },
            }
            if problems == 0 {
                println!("doctor: all checks passed");
                Ok(())
            } else {
                Err(Error::Store(format!("doctor found {problems} problem(s)")))
            }
        }
        Commands::Dev(dev) => match dev.stage {
            DevStage::Schedule(args) => run_dev_schedule(&args.fixture),
            DevStage::Ingest(args) => run_dev_ingest(
                &args.pdf,
                args.start_page,
                args.max_unit_pages,
                args.manual_boundaries.as_deref(),
                args.chapter_level,
                args.max_units,
            ),
            DevStage::Mcq(args) => run_dev_mcq(
                &args.pdf,
                &args.phase,
                args.start_page,
                args.seed,
                args.persist,
                args.print_only,
                args.generation.as_deref(),
            ),
            DevStage::Assignment(args) => run_dev_assignment(
                &args.pdf,
                args.start_page,
                args.seed,
                args.persist,
                args.print_only,
            ),
            DevStage::Grade(args) => run_dev_grade(&args.answers, &args.rubric),
            DevStage::Notes(args) => run_dev_notes(&args.pdf, args.misconceptions.as_deref()),
            DevStage::Llm(args) => run_dev_llm(
                &args.operation,
                args.prompt.as_deref(),
                args.model.as_deref(),
            ),
        },
    }
}

fn main() {
    if let Err(e) = run() {
        // Error report goes to stderr with a nonzero exit; stdout stays
        // script-parseable. `Stderr` writes can fail on a closed handle, so
        // the result is discarded.
        let _ = writeln!(std::io::stderr(), "cadence: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_question_text_joins_parts() {
        assert_eq!(
            assignment_question_text(r#"["a) Explain &x.", "b) Give a counterexample."]"#),
            "a) Explain &x.\nb) Give a counterexample."
        );
        // Corrupt rows degrade to rubric-only grading, never panic.
        assert_eq!(assignment_question_text("{broken"), String::new());
        assert_eq!(assignment_question_text("[]"), String::new());
    }

    #[test]
    fn date_helpers_bound() {
        assert_eq!(
            parse_date("2026-01-10").unwrap(),
            NaiveDate::from_ymd_opt(2026, 1, 10).unwrap()
        );
        assert!(parse_date("not-a-date").is_err());
        assert!(parse_date("2026-13-40").is_err());
        // Today is a real calendar date, not the epoch default.
        assert!(today_date() > NaiveDate::from_ymd_opt(2020, 1, 1).unwrap());
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        assert_eq!(horizon_end(today, 0), today);
        assert_eq!(horizon_end(today, -3), today);
        assert_eq!(
            horizon_end(today, 3),
            NaiveDate::from_ymd_opt(2026, 1, 13).unwrap()
        );
        assert_eq!(
            horizon_end(today, 1),
            NaiveDate::from_ymd_opt(2026, 1, 11).unwrap()
        );
    }

    #[test]
    fn display_helpers_format() {
        // snip: short text untouched, long text ellipsized on first line.
        assert_eq!(snip("hello", 10), "hello");
        assert_eq!(snip("hello world", 5), "hello...");
        assert_eq!(snip("exact", 5), "exact");
        assert_eq!(snip("line one\nline two", 20), "line one");
        assert_eq!(snip("", 5), "");
        // classify_label mirrors the scheduler buckets.
        let mk = |kind| domain::Task {
            id: 1,
            book_id: 1,
            chapter_id: 1,
            task_type: kind,
            scheduled_for: NaiveDate::from_ymd_opt(2026, 1, 9).unwrap(),
            status: domain::TaskStatus::Pending,
            completed_at: None,
            sequence: 1,
            attempt_no: 1,
        };
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        assert_eq!(
            classify_label(&mk(domain::TaskType::Pretest), today),
            "OVERDUE"
        );
        assert_eq!(
            classify_label(
                &domain::Task {
                    scheduled_for: today,
                    ..mk(domain::TaskType::Read)
                },
                today
            ),
            "DUE TODAY"
        );
    }

    #[test]
    fn criteria_results_json_round_trips() {
        let results = vec![grading::CriterionResult {
            name: "correctness".to_string(),
            score: 4,
            max_score: 5,
            comment: "Address named correctly.".to_string(),
        }];
        let parsed: serde_json::Value =
            serde_json::from_str(&criteria_results_json(&results)).unwrap();
        assert_eq!(parsed[0]["name"], serde_json::json!("correctness"));
        assert_eq!(parsed[0]["score"], serde_json::json!(4));
        assert_eq!(parsed[0]["max_score"], serde_json::json!(5));
        assert!(criteria_results_json(&[]).contains("[]"));
    }

    #[test]
    fn grade_to_new_preserves_audit_trail() {
        let grade = grading::Grade {
            classification: grading::GradeClass::PartiallyCorrect,
            score: 6,
            criteria_results: vec![grading::CriterionResult {
                name: "correctness".to_string(),
                score: 3,
                max_score: 5,
                comment: "Half right.".to_string(),
            }],
            feedback: "Good start on aliasing.".to_string(),
        };
        let row = grade_to_new(7, &grade, 8, "2026-09-25");
        assert_eq!(row.question_id, 7);
        assert_eq!(row.score, 6);
        assert_eq!(row.max_score, 8);
        assert_eq!(row.classification, "PARTIALLY_CORRECT");
        assert_eq!(row.feedback, "Good start on aliasing.");
        assert_eq!(row.grader_version, llm::PROMPT_VERSION);
        assert_eq!(row.created_at, "2026-09-25");
        assert!(row.criteria_results_json.contains("correctness"));
    }

    #[test]
    fn misconception_items_map_all_statuses() {
        let rows = vec![
            store::Misconception {
                id: 1,
                chapter_id: 2,
                concept_description: "aliasing".to_string(),
                description: " fuller ".to_string(),
                evidence: "missed it".to_string(),
                source_task: "RETEST".to_string(),
                status: "ACTIVE".to_string(),
                confidence: 0.5,
                created_at: "2026-09-25".to_string(),
                updated_at: "2026-09-25".to_string(),
                resolved_at: None,
            },
            store::Misconception {
                id: 2,
                chapter_id: 2,
                concept_description: "scales".to_string(),
                description: " fuller ".to_string(),
                evidence: "fixed on assignment".to_string(),
                source_task: "ASSIGNMENT".to_string(),
                status: "RESOLVED".to_string(),
                confidence: 0.9,
                created_at: "2026-09-25".to_string(),
                updated_at: "2026-09-25".to_string(),
                resolved_at: Some("2026-09-25".to_string()),
            },
        ];
        let items = misconception_items_for(&rows);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].concept, "aliasing");
        assert_eq!(items[0].evidence, "missed it");
        assert_eq!(items[0].status, "ACTIVE");
        assert_eq!(items[1].status, "RESOLVED");
        assert_eq!(misconception_items_for(&[]).len(), 0);
    }

    #[test]
    fn assignment_misconception_texts_label_and_truncate() {
        let (concept, description, evidence) = assignment_misconception_texts(
            0,
            "a) Explain what &x yields and why it matters for aliasing.",
            "INCORRECT",
            2,
            8,
            "Confused address with value.",
            "I think &x gives the value stored at x...",
        );
        assert!(concept.starts_with("Assignment Q1: "), "{concept}");
        assert!(concept.contains("&x"));
        assert!(description.contains("INCORRECT 2/8"), "{description}");
        assert!(description.contains("Confused address"), "{description}");
        assert!(evidence.starts_with("Answer excerpt: "), "{evidence}");
        // Long inputs truncate instead of bloating the row.
        let long_question = "q".repeat(200);
        let (long_concept, _, _) = assignment_misconception_texts(
            3,
            &long_question,
            "PARTIALLY_CORRECT",
            4,
            8,
            "Half right.",
            "a",
        );
        assert!(
            long_concept.starts_with("Assignment Q4: "),
            "{long_concept}"
        );
        assert!(long_concept.ends_with("..."), "{long_concept}");
        assert!(long_concept.chars().count() <= "Assignment Q4: ".len() + 103);
    }

    fn review_answer_fixtures() -> (
        MemoryStore,
        store::McqItem,
        mcq::ValidatedMcq,
        mcq::DisplayedMcq,
        i64,
    ) {
        let mut store = MemoryStore::new();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "T".to_string(),
                    filepath: "/t.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 1,
                },
                "2026-09-27",
            )
            .unwrap();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 1,
                end_page: 20,
                file_path: "u1.json".to_string(),
                status: domain::ChapterStatus::Completed,
            })
            .unwrap();
        let row = mcq::ValidatedMcq {
            question: "What does &x yield?".to_string(),
            options: vec![
                "The address of x".to_string(),
                "The value of x".to_string(),
                "A null pointer".to_string(),
                "A dangling reference".to_string(),
            ],
            correct_index: 0,
            trap_index: 1,
            explanation: "The & operator takes the address of its operand, plainly stated."
                .to_string(),
            topic: "addresses".to_string(),
            source_refs: mcq::SourceRefs {
                pages: vec![12],
                sections: vec!["Addresses".to_string()],
            },
        };
        let items = store
            .save_mcq_items(
                &dev_mcq::to_new_items_for(
                    chapter.id,
                    mcq::McqPhase::Review,
                    std::slice::from_ref(&row),
                    1,
                )
                .unwrap(),
            )
            .unwrap();
        let shown = mcq::DisplayedMcq {
            question: row.question.clone(),
            displayed_options: row.options.clone(),
            displayed_correct: 0,
            displayed_trap: 1,
            explanation: row.explanation.clone(),
            topic: row.topic.clone(),
        };
        let item = items.into_iter().next().unwrap();
        (store, item, row, shown, chapter.id)
    }

    #[test]
    fn review_correct_boosts_topic_row() {
        let (mut store, item, row, shown, chapter_id) = review_answer_fixtures();
        store
            .create_misconception(
                chapter_id,
                "addresses",
                "took &x for the value",
                "picked value",
                "RETEST",
                "2026-09-27",
            )
            .unwrap();
        let outcome = record_answer(
            &mut store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase: mcq::McqPhase::Review,
                row: &row,
                shown: &shown,
                selected: 0,
                attempt_no: 1,
                today: "2026-09-27",
            },
        )
        .unwrap();
        assert!(outcome.is_correct);
        assert!(!outcome.misconception_logged);
        let rows = store.list_misconceptions(chapter_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].confidence - 0.6).abs() < 1e-9);
        assert_eq!(rows[0].status, "IMPROVING");
    }

    #[test]
    fn review_wrong_nudges_instead_of_logging() {
        let (mut store, item, row, shown, chapter_id) = review_answer_fixtures();
        store
            .create_misconception(
                chapter_id,
                "addresses",
                "took &x for the value",
                "picked value",
                "RETEST",
                "2026-09-27",
            )
            .unwrap();
        let outcome = record_answer(
            &mut store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase: mcq::McqPhase::Review,
                row: &row,
                shown: &shown,
                selected: 2,
                attempt_no: 1,
                today: "2026-09-27",
            },
        )
        .unwrap();
        assert!(!outcome.is_correct);
        assert!(outcome.misconception_logged);
        // Nudged in place — no duplicate row.
        let rows = store.list_misconceptions(chapter_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].confidence - 0.4).abs() < 1e-9);
        assert_eq!(rows[0].status, "ACTIVE");
    }

    #[test]
    fn review_idk_abstains_without_lifecycle_effect() {
        let (mut store, item, row, shown, chapter_id) = review_answer_fixtures();
        store
            .create_misconception(
                chapter_id,
                "addresses",
                "took &x for the value",
                "picked value",
                "RETEST",
                "2026-09-27",
            )
            .unwrap();
        let outcome = record_answer(
            &mut store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase: mcq::McqPhase::Review,
                row: &row,
                shown: &shown,
                selected: mcq::IDK_INDEX,
                attempt_no: 1,
                today: "2026-09-27",
            },
        )
        .unwrap();
        assert!(!outcome.is_correct);
        assert!(!outcome.misconception_logged);
        let rows = store.list_misconceptions(chapter_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn review_wrong_trap_marks_trap_evidence() {
        let (mut store, item, row, shown, chapter_id) = review_answer_fixtures();
        // Selecting the designated trap records trap evidence (not generic
        // wrong-answer evidence) on the fresh row and the response.
        let outcome = record_answer(
            &mut store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase: mcq::McqPhase::Review,
                row: &row,
                shown: &shown,
                selected: 1,
                attempt_no: 1,
                today: "2026-09-27",
            },
        )
        .unwrap();
        assert!(!outcome.is_correct);
        assert!(outcome.misconception_logged);
        let rows = store.list_misconceptions(chapter_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].evidence.contains("trap selected"));
        let responses = store.list_mcq_responses(item.id).unwrap();
        assert_eq!(responses.len(), 1);
        assert!(responses[0].selected_trap);
    }

    #[test]
    fn review_wrong_without_open_row_logs_fresh() {
        let (mut store, item, row, shown, chapter_id) = review_answer_fixtures();
        // No open row (nothing logged yet): the wrong answer is still
        // evidence, so it falls back to a fresh row like a retest would.
        let outcome = record_answer(
            &mut store,
            &Answer {
                item_id: item.id,
                chapter_id,
                phase: mcq::McqPhase::Review,
                row: &row,
                shown: &shown,
                selected: 2,
                attempt_no: 1,
                today: "2026-09-27",
            },
        )
        .unwrap();
        assert!(!outcome.is_correct);
        assert!(outcome.misconception_logged);
        let rows = store.list_misconceptions(chapter_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "ACTIVE");
        assert!((rows[0].confidence - 0.5).abs() < 1e-9);
        assert!(rows[0].evidence.contains("non-trap wrong answer"));
        let responses = store.list_mcq_responses(item.id).unwrap();
        assert_eq!(responses.len(), 1);
        assert!(!responses[0].selected_trap);
    }

    #[test]
    fn notes_listing_shows_only_noted_chapters() {
        let (mut store, chapter) = sqlite_chapter();
        let empty = format_notes_listing(&store).unwrap();
        assert!(empty.contains("Chapters with notes:"));
        assert!(!empty.contains("Pointers"));
        store
            .save_note(&store::NewNote {
                chapter_id: chapter.id,
                content_markdown: "## A\nText.".to_string(),
                generated_at: "2026-01-10".to_string(),
                attempt_no: chapter.attempt_no,
            })
            .unwrap();
        let text = format_notes_listing(&store).unwrap();
        assert!(text.contains("Book 1 — Modern C:"));
        assert!(text.contains("'Pointers' (pages 10–20, attempt 1)"));
        // Unknown chapters fail loudly, not silently empty.
        assert!(stored_notes_markdown(&store, 999).is_err());
    }

    #[test]
    fn daily_loop_exits_clean_on_empty_queue() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let today_str = "2026-01-10".to_string();
        run_daily_loop(&mut store, today, &today_str).unwrap();
    }

    #[test]
    fn ensure_all_books_counts_scheduled() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        assert_eq!(
            ensure_all_books(&mut store, today, "2026-01-10").unwrap(),
            0
        );
        let (mut seeded, _) = sqlite_chapter();
        assert!(ensure_all_books(&mut seeded, today, "2026-01-10").unwrap() > 0);
    }

    #[test]
    fn paint_wraps_only_when_enabled() {
        assert_eq!(
            paint("Correct.", GREEN_CODE, true),
            "\x1b[32mCorrect.\x1b[0m"
        );
        assert_eq!(paint("Correct.", GREEN_CODE, false), "Correct.");
        assert_eq!(
            paint("Incorrect.", RED_CODE, true),
            "\x1b[31mIncorrect.\x1b[0m"
        );
    }

    #[test]
    fn question_header_counts_on_a_ruled_line() {
        assert_eq!(question_header(1, 8, false), "─── Question 1/8 ───");
        assert_eq!(question_header(8, 8, false), "─── Question 8/8 ───");
        assert_eq!(
            question_header(2, 8, true),
            "\x1b[1m─── Question 2/8 ───\x1b[0m"
        );
    }

    #[test]
    fn gate_answers_route_three_ways() {
        assert_eq!(parse_gate_answer(""), Some(ChapterGate::Proceed));
        assert_eq!(parse_gate_answer("y"), Some(ChapterGate::Proceed));
        assert_eq!(parse_gate_answer(" YES "), Some(ChapterGate::Proceed));
        assert_eq!(parse_gate_answer("s"), Some(ChapterGate::Skip));
        assert_eq!(parse_gate_answer("Skip"), Some(ChapterGate::Skip));
        // `n` exits — it never skips.
        assert_eq!(parse_gate_answer("n"), Some(ChapterGate::Exit));
        assert_eq!(parse_gate_answer("no"), Some(ChapterGate::Exit));
        assert_eq!(parse_gate_answer("q"), Some(ChapterGate::Exit));
        assert_eq!(parse_gate_answer("quit"), Some(ChapterGate::Exit));
        assert_eq!(parse_gate_answer("exit"), Some(ChapterGate::Exit));
        assert_eq!(parse_gate_answer("yellow"), None);
        assert_eq!(parse_gate_answer("snooze"), None);
    }

    #[test]
    fn due_and_future_partitions_and_orders() {
        use chrono::NaiveDate;
        let today = NaiveDate::from_ymd_opt(2026, 9, 27).unwrap();
        let task = |id: i64, kind: TaskType, date: &str, status: TaskStatus| Task {
            id,
            book_id: 1,
            chapter_id: id,
            task_type: kind,
            scheduled_for: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            status,
            completed_at: None,
            sequence: id,
            attempt_no: 1,
        };
        let tasks = vec![
            task(1, TaskType::Pretest, "2026-09-27", TaskStatus::Pending),
            task(2, TaskType::Retest, "2026-09-28", TaskStatus::Pending),
            task(
                3,
                TaskType::AssignmentWrite,
                "2026-09-29",
                TaskStatus::Pending,
            ),
            task(4, TaskType::Retest, "2026-09-28", TaskStatus::Pending),
            task(5, TaskType::Read, "2026-09-27", TaskStatus::Done),
        ];
        let (due, future) = due_and_future(&tasks, today);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].task_type, TaskType::Pretest);
        // Future work orders by date, then creation sequence.
        assert_eq!(
            future.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![2, 4, 3]
        );
    }

    #[test]
    fn chapter_display_names_or_falls_back() {
        let mut mem = SqliteStore::open_in_memory().unwrap();
        assert_eq!(chapter_title_or_id(&mem, 7), "chapter 7");
        let book = mem
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "m.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 10,
                },
                "2026-01-01",
            )
            .unwrap();
        let chapter = mem
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::PretestReady,
            })
            .unwrap();
        assert_eq!(chapter_title_or_id(&mem, chapter.id), "'Pointers'");
        let task = domain::Task {
            id: 1,
            book_id: book.id,
            chapter_id: chapter.id,
            task_type: domain::TaskType::Read,
            scheduled_for: NaiveDate::from_ymd_opt(2026, 1, 10).unwrap(),
            status: domain::TaskStatus::Pending,
            completed_at: None,
            sequence: 1,
            attempt_no: 1,
        };
        assert_eq!(
            describe_task(&mem, &task).unwrap(),
            "READ 'Pointers' (due 2026-01-10)"
        );
        assert!(
            describe_task(
                &mem,
                &domain::Task {
                    chapter_id: 999,
                    ..task
                }
            )
            .is_err()
        );
    }

    fn sqlite_chapter() -> (SqliteStore, domain::Chapter) {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "m.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 10,
                },
                "2026-01-01",
            )
            .unwrap();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::PretestReady,
            })
            .unwrap();
        (store, chapter)
    }

    fn slated_task(
        id: i64,
        chapter_id: i64,
        kind: domain::TaskType,
        date: &str,
        status: domain::TaskStatus,
    ) -> domain::Task {
        domain::Task {
            id,
            book_id: 1,
            chapter_id,
            task_type: kind,
            scheduled_for: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            status,
            completed_at: None,
            sequence: id,
            attempt_no: 1,
        }
    }

    #[test]
    fn queue_listing_splits_overdue_and_numbers_today() {
        let (store, chapter) = sqlite_chapter();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let tasks = vec![
            slated_task(
                1,
                chapter.id,
                domain::TaskType::Retest,
                "2026-01-09",
                domain::TaskStatus::Pending,
            ),
            slated_task(
                2,
                chapter.id,
                domain::TaskType::Pretest,
                "2026-01-10",
                domain::TaskStatus::Pending,
            ),
        ];
        let text = format_queue(&store, &tasks, today);
        assert!(text.contains("Overdue (1):"));
        assert!(text.contains("Today's queue (2) for 2026-01-10:"));
        assert!(text.contains("1. [RETEST] 'Pointers' (scheduled 2026-01-09)"));
        let clear = format_queue(&store, &[], today);
        assert!(clear.contains("No tasks due for 2026-01-10. Queue is clear."));
    }

    #[test]
    fn assignment_set_lists_parts_and_targets() {
        let items = vec![
            store::AssignmentQuestion {
                id: 1,
                chapter_id: 1,
                position: 0,
                kind: "written".to_string(),
                parts_json: r#"["a) One.", "b) Two."]"#.to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[3]".to_string(),
                attempt_no: 1,
            },
            store::AssignmentQuestion {
                id: 2,
                chapter_id: 1,
                position: 1,
                kind: "coding".to_string(),
                parts_json: r#"["Write it."]"#.to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no: 1,
            },
        ];
        let text = format_assignment_set(&items);
        assert!(text.contains("Assignment set: 2 question(s)"));
        assert!(text.contains("1. [written]"));
        assert!(text.contains("   a) One."));
        assert!(text.contains("(re-probes misconceptions: [3])"));
        assert!(text.contains("2. [coding]"));
        assert_eq!(
            format_assignment_set(&[]),
            "Assignment set: 0 question(s)\n"
        );
    }

    #[test]
    fn day_complete_names_next_or_nothing() {
        let (mut store, chapter) = sqlite_chapter();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let text = format_day_complete(&store, today).unwrap();
        assert!(text.contains("nothing scheduled ahead"));
        store
            .create_task(&store::NewTask {
                book_id: 1,
                chapter_id: chapter.id,
                task_type: domain::TaskType::Pretest,
                scheduled_for: NaiveDate::from_ymd_opt(2026, 1, 11).unwrap(),
                sequence: 1,
                attempt_no: 1,
            })
            .unwrap();
        let text = format_day_complete(&store, today).unwrap();
        assert!(text.contains("Next scheduled: PRETEST 'Pointers' (due 2026-01-11)."));
    }

    #[test]
    fn score_footer_counts() {
        assert_eq!(
            format_mcq_score(5, 1),
            "Answered: 5. Misconception updates this session: 1."
        );
    }

    #[test]
    fn dispute_verdict_branches() {
        let base = dispute::DisputeResult {
            dispute_valid: true,
            final_score: 5,
            explanation: "Scales exclude 5.".to_string(),
            action: dispute::DisputeAction::Revised,
        };
        let text = format_dispute_verdict(&base, 9, 5, 2, 2);
        assert!(text.contains("Misconceptions purged: 2 row(s)"));
        assert!(text.contains("Verdict: REVISED — final 5/5 (was 2/5)"));
        assert!(text.contains("Grade corrected in SQLite (dispute row 9;"));
        let upheld = dispute::DisputeResult {
            dispute_valid: false,
            action: dispute::DisputeAction::Upheld,
            ..base.clone()
        };
        let text = format_dispute_verdict(&upheld, 9, 5, 2, 0);
        assert!(!text.contains("Misconceptions purged"));
        assert!(text.contains("Original grade stands (dispute row 9 recorded)."));
        let defective = dispute::DisputeResult {
            action: dispute::DisputeAction::QuestionDefective,
            ..base
        };
        assert!(format_dispute_verdict(&defective, 9, 5, 2, 0).contains("QUESTION_DEFECTIVE"));
    }

    #[test]
    fn skip_listing_flags_meta_and_usage() {
        let (store, _) = sqlite_chapter();
        let text = format_skip_listing(&store).unwrap();
        assert!(text.contains("Book 1 — Modern C:"));
        assert!(text.contains("[PRETEST_READY] 1 'Pointers' (pages 10–20, attempt 1)"));
        assert!(text.contains("Usage: cadence skip <id> | cadence unskip <id>"));
        let empty = SqliteStore::open_in_memory().unwrap();
        assert!(
            format_skip_listing(&empty)
                .unwrap()
                .contains("No books ingested yet.")
        );
    }

    #[test]
    fn progress_lists_live_and_skipped() {
        let (mut store, chapter) = sqlite_chapter();
        let text = format_run_progress(&store).unwrap();
        assert!(text.contains("Modern C — 1 unit(s), 0 skipped:"));
        assert!(text.contains("Ch1 'Pointers' [PRETEST_READY]"));
        store
            .set_chapter_status(chapter.id, domain::ChapterStatus::Skipped)
            .unwrap();
        let text = format_run_progress(&store).unwrap();
        assert!(text.contains("0 unit(s), 1 skipped:"));
        assert!(text.contains("Skipped: Ch1 'Pointers' (excluded from evidence)"));
    }

    fn metric_board(eta: Option<NaiveDate>, remaining: usize, overdue: Vec<Task>) -> MetricsReport {
        let board = metrics::Dashboard {
            units_completed: 1,
            units_remaining: remaining,
            pages_completed: 20,
            pages_remaining: 40,
            tasks_done: 2,
            tasks_total: 3,
            skipped: 1,
            pretest: Some(80.0),
            retest: None,
            assignment: Some(75.0),
            pretest_fraction: (4, 5),
            retest_fraction: (0, 0),
            assignment_fraction: (6, 8),
            active_misconceptions: 2,
            resolved_misconceptions: 1,
            resolution: Some(33.3),
            pace: metrics::Pace {
                units_per_week: 1.0,
                pages_per_week: 20.0,
                eta,
            },
            consistency: metrics::Consistency {
                days_active: 3,
                current_streak: 2,
                longest_streak: 2,
            },
            on_time: Some(66.7),
            overdue: overdue.len(),
        };
        MetricsReport {
            board,
            overdue,
            chapters: Vec::new(),
        }
    }

    #[test]
    fn metrics_format_covers_eta_arms_and_debt() {
        // Unknown ETA + debt.
        let debt_task = slated_task(
            1,
            7,
            domain::TaskType::Retest,
            "2026-01-05",
            domain::TaskStatus::Pending,
        );
        let text = format_metrics(&metric_board(None, 2, vec![debt_task]));
        assert!(text.contains("Completion: 1/3 units (20 pages done, 40 left), 2/3 tasks done."));
        assert!(text.contains("Skipped: 1 chapter(s)"));
        assert!(text.contains("ETA unknown (no completions in the last 7 days)."));
        assert!(text.contains("Learning debt: 1 overdue task(s):"));
        assert!(text.contains("[RETEST] 'unknown chapter' (chapter 7, scheduled 2026-01-05)"));
        // Dated ETA with units left.
        let text = format_metrics(&metric_board(
            Some(NaiveDate::from_ymd_opt(2026, 2, 1).unwrap()),
            2,
            Vec::new(),
        ));
        assert!(text.contains("ETA 2026-02-01 (2 units left)."));
        assert!(text.contains("Learning debt: none."));
        // Dated ETA with nothing left: all complete.
        let text = format_metrics(&metric_board(
            Some(NaiveDate::from_ymd_opt(2026, 2, 1).unwrap()),
            0,
            Vec::new(),
        ));
        assert!(text.contains("all units complete."));
    }

    #[test]
    fn metrics_collect_counts_live_work() {
        let (mut store, chapter) = sqlite_chapter();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let done = store
            .create_task(&store::NewTask {
                book_id: 1,
                chapter_id: chapter.id,
                task_type: domain::TaskType::Pretest,
                scheduled_for: today,
                sequence: 1,
                attempt_no: 1,
            })
            .unwrap();
        store.complete_task(done.id, "2026-01-10").unwrap();
        store
            .create_task(&store::NewTask {
                book_id: 1,
                chapter_id: chapter.id,
                task_type: domain::TaskType::Read,
                scheduled_for: NaiveDate::from_ymd_opt(2026, 1, 9).unwrap(),
                sequence: 2,
                attempt_no: 1,
            })
            .unwrap();
        let report = collect_metrics(&store, today).unwrap();
        assert_eq!(report.board.tasks_done, 1);
        assert_eq!(report.board.tasks_total, 2);
        assert_eq!(report.board.skipped, 0);
        assert_eq!(report.overdue.len(), 1);
        assert_eq!(report.overdue[0].task_type, domain::TaskType::Read);
        assert_eq!(report.chapters.len(), 1);
    }

    /// Fixture PDF beside the checkout (or `CADENCE_FIXTURE_PDF`); `None`
    /// when absent so fixture-gated tests skip like the pdf suite.
    fn fixture_pdf() -> Option<std::path::PathBuf> {
        if let Some(path) = std::env::var_os("CADENCE_FIXTURE_PDF") {
            return Some(std::path::PathBuf::from(path));
        }
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.push("Modern C.pdf");
        p.is_file().then_some(p)
    }

    #[test]
    fn extract_first_unit_validates_and_resolves() {
        let Some(pdf) = fixture_pdf() else {
            return;
        };
        let pdf = pdf.to_string_lossy().into_owned();
        assert!(extract_first_unit(&pdf, 0).is_err());
        let err = extract_first_unit(&pdf, 999).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("exceeds document"));
        // Past the last outline entry the planner (not the range check)
        // reports the empty plan.
        // The last page resolves (fallback covers the past-outline tail).
        assert!(extract_first_unit(&pdf, 408).is_ok());
        assert!(extract_first_unit("/nonexistent.pdf", 1).is_err());
        let first = extract_first_unit(&pdf, 1).unwrap();
        assert!(first.unit.text.contains("physical page"));
        assert_eq!(first.pdf_name, "Modern C.pdf");
    }

    #[test]
    fn ingest_registers_book_from_fixture() {
        let Some(pdf) = fixture_pdf() else {
            return;
        };
        let pdf = pdf.to_string_lossy().into_owned();
        let dir: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            ".scratch",
            "ingest-fixture-test",
        ]
        .iter()
        .collect();
        let _ = std::fs::remove_dir_all(&dir);
        let config = Config {
            data_dir: dir.clone(),
            catch_up_first: true,
            reserve_new_per_day: 1,
        };
        // Bad caps and pages fail before touching the store.
        assert!(run_ingest(&pdf, 1, 0, None, None, None, &config).is_err());
        assert!(
            run_ingest(&pdf, 999, 50, None, None, None, &config)
                .unwrap_err()
                .to_string()
                .contains("exceeds document")
        );
        // A late start keeps the run fast while exercising the full path.
        run_ingest(&pdf, 387, 50, None, None, None, &config).unwrap();
        let store = SqliteStore::open(&config.db_path(), &config.lock_path()).unwrap();
        assert_eq!(store.list_books().unwrap().len(), 1);
        let tasks = store.list_tasks().unwrap();
        assert!(
            tasks
                .iter()
                .any(|t| t.status == domain::TaskStatus::Pending)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ingest_manual_overrides_unplannable_auto() {
        let Some(pdf) = fixture_pdf() else {
            return;
        };
        let pdf = pdf.to_string_lossy().into_owned();
        let dir: std::path::PathBuf =
            [env!("CARGO_MANIFEST_DIR"), ".scratch", "ingest-manual-test"]
                .iter()
                .collect();
        let _ = std::fs::remove_dir_all(&dir);
        let config = Config {
            data_dir: dir.clone(),
            catch_up_first: true,
            reserve_new_per_day: 1,
        };
        // A one-page cap cannot tile automatically: without manual ranges
        // the run fails asking for them (no terminal here); with ranges it
        // succeeds.
        assert!(run_ingest(&pdf, 387, 1, None, None, None, &config).is_err());
        run_ingest(&pdf, 387, 1, None, Some("387-408"), None, &config).unwrap();
        let store = SqliteStore::open(&config.db_path(), &config.lock_path()).unwrap();
        assert_eq!(store.list_books().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn mem_chapter(store: &mut MemoryStore) -> domain::Chapter {
        let book = store
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "m.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 10,
                },
                "2026-01-01",
            )
            .unwrap();
        store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::PretestReady,
            })
            .unwrap()
    }

    fn mem_misconception(
        store: &mut MemoryStore,
        chapter_id: i64,
        concept: &str,
    ) -> store::Misconception {
        store
            .create_misconception(chapter_id, concept, "d", "e", "RETEST", "2026-01-10")
            .unwrap()
    }

    #[test]
    fn misconception_tally_counts_statuses() {
        let mut store = MemoryStore::new();
        let chapter = mem_chapter(&mut store);
        assert_eq!(misconception_tally(&store, chapter.id).unwrap(), (0, 0));
        let active = mem_misconception(&mut store, chapter.id, "a");
        let improving = mem_misconception(&mut store, chapter.id, "b");
        store
            .update_misconception(improving.id, 0.5, "IMPROVING", "2026-01-10", None)
            .unwrap();
        let resolved = mem_misconception(&mut store, chapter.id, "c");
        store
            .update_misconception(
                resolved.id,
                1.0,
                "RESOLVED",
                "2026-01-10",
                Some("2026-01-10"),
            )
            .unwrap();
        assert_eq!(misconception_tally(&store, chapter.id).unwrap(), (2, 1));
        assert_eq!(
            misconception_tally(&store, chapter.id + 999).unwrap(),
            (0, 0)
        );
        let _ = active;
    }

    #[test]
    fn open_misconceptions_maps_active_only() {
        let mut store = MemoryStore::new();
        let chapter = mem_chapter(&mut store);
        let active = mem_misconception(&mut store, chapter.id, "addr");
        let resolved = mem_misconception(&mut store, chapter.id, "null");
        store
            .update_misconception(
                resolved.id,
                1.0,
                "RESOLVED",
                "2026-01-10",
                Some("2026-01-10"),
            )
            .unwrap();
        let open = open_misconceptions(&store, chapter.id).unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, active.id);
        assert_eq!(open[0].concept, "addr");
    }

    #[test]
    fn nudge_targeted_skips_gone_and_inactive() {
        let (mut store, chapter) = sqlite_chapter();
        // Empty targets: no-op.
        nudge_targeted_misconceptions(&mut store, chapter.id, &[], true, "2026-01-10").unwrap();
        let gone_before = store.list_misconceptions(chapter.id).unwrap().len();
        // Unknown ids are reported and skipped, never fatal.
        nudge_targeted_misconceptions(&mut store, chapter.id, &[999], false, "2026-01-10").unwrap();
        // Non-active rows never move.
        let resolved = store
            .create_misconception(chapter.id, "r", "d", "e", "RETEST", "2026-01-10")
            .unwrap();
        store
            .update_misconception(
                resolved.id,
                1.0,
                "RESOLVED",
                "2026-01-10",
                Some("2026-01-10"),
            )
            .unwrap();
        nudge_targeted_misconceptions(&mut store, chapter.id, &[resolved.id], false, "2026-01-10")
            .unwrap();
        let after = store.list_misconceptions(chapter.id).unwrap();
        assert_eq!(after.len(), gone_before + 1);
        // ACTIVE rows move down on wrong answers.
        let active = store
            .create_misconception(chapter.id, "a", "d", "e", "RETEST", "2026-01-10")
            .unwrap();
        let before_conf = active.confidence;
        nudge_targeted_misconceptions(&mut store, chapter.id, &[active.id], false, "2026-01-10")
            .unwrap();
        let moved = store
            .list_misconceptions(chapter.id)
            .unwrap()
            .into_iter()
            .find(|r| r.id == active.id)
            .unwrap();
        assert!(moved.confidence < before_conf);
    }

    #[test]
    fn topic_nudges_count_matching_rows_only() {
        let mut store = MemoryStore::new();
        let chapter = mem_chapter(&mut store);
        mem_misconception(&mut store, chapter.id, "addr");
        let settled = mem_misconception(&mut store, chapter.id, "addr");
        store
            .update_misconception(
                settled.id,
                1.0,
                "RESOLVED",
                "2026-01-10",
                Some("2026-01-10"),
            )
            .unwrap();
        mem_misconception(&mut store, chapter.id, "other");
        assert_eq!(
            boost_matching_misconceptions(&mut store, chapter.id, "addr", "2026-01-10").unwrap(),
            1
        );
        assert_eq!(
            nudge_matching_misconceptions(&mut store, chapter.id, "addr", "2026-01-10").unwrap(),
            1
        );
        assert_eq!(
            nudge_matching_misconceptions(&mut store, chapter.id, "missing", "2026-01-10").unwrap(),
            0
        );
    }

    #[test]
    fn grade_summaries_skip_ungraded() {
        let (mut store, chapter) = sqlite_chapter();
        let questions = store
            .save_assignment_questions(&[store::NewAssignmentQuestion {
                chapter_id: chapter.id,
                position: 0,
                kind: "written".to_string(),
                parts_json: "[\"a) Q.\"]".to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no: 1,
            }])
            .unwrap();
        // Unasked: ungraded questions are left out, not failed.
        assert_eq!(
            grade_summaries_for(&store, chapter.id, 1).unwrap(),
            Vec::new()
        );
        store
            .save_grade(&store::NewGrade {
                question_id: questions[0].id,
                score: 4,
                max_score: 5,
                classification: "CORRECT_BUT_BRIEF".to_string(),
                criteria_results_json: "[]".to_string(),
                feedback: "Good.".to_string(),
                grader_version: "v3".to_string(),
                created_at: "2026-01-08".to_string(),
            })
            .unwrap();
        let summaries = grade_summaries_for(&store, chapter.id, 1).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].score, 4);
        assert_eq!(summaries[0].max_score, 5);
    }

    #[test]
    fn ensure_resumes_stored_mcq_rows() {
        let (mut store, chapter) = sqlite_chapter();
        let unit = engines::UnitText {
            text: "t".to_string(),
            page_start: 10,
            page_end: 20,
            heading: "Pointers".to_string(),
        };
        // Seeded MCQ rows resume without generating (generation needs the
        // network and is covered by the dev harness, not here).
        let seeded = store
            .save_mcq_items(&[store::NewMcqItem {
                chapter_id: chapter.id,
                phase: "pretest".to_string(),
                question_text: "Q?".to_string(),
                options_json: "[\"a\",\"b\",\"c\",\"d\"]".to_string(),
                correct_index: 0,
                trap_index: 1,
                explanation_text: "E.".to_string(),
                source_refs: "{}".to_string(),
                topic: "t".to_string(),
                attempt_no: chapter.attempt_no,
            }])
            .unwrap();
        let resumed = ensure_production_items(
            &mut store,
            &chapter,
            &unit,
            mcq::McqPhase::Pretest,
            "2026-01-10",
        )
        .unwrap();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].id, seeded[0].id);
    }

    #[test]
    fn ensure_resumes_stored_assignment_rows() {
        let (mut store, chapter) = sqlite_chapter();
        let unit = engines::UnitText {
            text: "t".to_string(),
            page_start: 10,
            page_end: 20,
            heading: "Pointers".to_string(),
        };
        // Same for assignment rows.
        let saved = store
            .save_assignment_questions(&[store::NewAssignmentQuestion {
                chapter_id: chapter.id,
                position: 0,
                kind: "written".to_string(),
                parts_json: "[\"a) Q.\"]".to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no: chapter.attempt_no,
            }])
            .unwrap();
        let resumed =
            ensure_production_assignment_questions(&mut store, &chapter, &unit, "2026-01-10")
                .unwrap();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].id, saved[0].id);
    }

    fn graded_question(store: &mut SqliteStore, chapter_id: i64) -> store::AssignmentQuestion {
        store
            .save_assignment_questions(&[store::NewAssignmentQuestion {
                chapter_id,
                position: 0,
                kind: "written".to_string(),
                parts_json: "[\"a) Explain &x.\"]".to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no: 1,
            }])
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    fn incorrect_grade() -> grading::Grade {
        grading::Grade {
            classification: grading::GradeClass::Incorrect,
            score: 1,
            criteria_results: Vec::new(),
            feedback: "Missed the address.".to_string(),
        }
    }

    #[test]
    fn grade_lifecycle_logs_misconception_on_incorrect() {
        let (mut store, chapter) = sqlite_chapter();
        let question = graded_question(&mut store, chapter.id);
        assert_eq!(store.list_misconceptions(chapter.id).unwrap().len(), 0);
        apply_grade_lifecycle(
            &mut store,
            &question,
            &incorrect_grade(),
            5,
            "The value of x",
            "2026-01-10",
        )
        .unwrap();
        // Incorrect with no targets still logs the answer's misconception.
        assert_eq!(store.list_misconceptions(chapter.id).unwrap().len(), 1);
    }

    #[test]
    fn guard_single_book_passes_empty_library() {
        let mut store = MemoryStore::new();
        assert!(guard_single_book(&mut store, &Config::default(), "New").unwrap());
    }

    #[test]
    fn tally_prior_answer_folds_resumed_items() {
        let mut store = MemoryStore::new();
        let chapter = mem_chapter(&mut store);
        let saved = store
            .save_mcq_items(&[store::NewMcqItem {
                chapter_id: chapter.id,
                phase: "pretest".to_string(),
                question_text: "Q?".to_string(),
                options_json: "[\"a\",\"b\",\"c\",\"d\"]".to_string(),
                correct_index: 0,
                trap_index: 1,
                explanation_text: "E.".to_string(),
                source_refs: "{}".to_string(),
                topic: "t".to_string(),
                attempt_no: 1,
            }])
            .unwrap();
        let mut totals = SessionTotals::default();
        assert!(!tally_prior_answer(&store, &saved[0], &mut totals).unwrap());
        assert_eq!(totals.answered, 0);
        store
            .record_mcq_response(saved[0].id, 0, true, false, "2026-01-10", 1)
            .unwrap();
        assert!(tally_prior_answer(&store, &saved[0], &mut totals).unwrap());
        assert_eq!(totals.answered, 1);
        assert_eq!(totals.correct, 1);
    }

    #[test]
    fn collect_review_views_maps_rows() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "m.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 10,
                },
                "2026-01-01",
            )
            .unwrap();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::PretestReady,
            })
            .unwrap();
        assert!(collect_review_views(&store).unwrap()[0].rows.is_empty());
        store
            .create_misconception(chapter.id, "addr", "d", "e", "RETEST", "2026-01-10")
            .unwrap();
        let views = collect_review_views(&store).unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].rows.len(), 1);
        assert_eq!(views[0].rows[0].concept, "addr");
    }

    #[test]
    fn misconceptions_listing_counts_statuses() {
        let (mut store, chapter) = sqlite_chapter();
        assert!(format_misconceptions(&store).unwrap().contains("0 active"));
        for (concept, status) in [
            ("a", "ACTIVE"),
            ("b", "IMPROVING"),
            ("c", "RESOLVED"),
            ("d", "DISPUTED"),
        ] {
            let row = store
                .create_misconception(chapter.id, concept, "d", "e", "RETEST", "2026-01-10")
                .unwrap();
            store
                .update_misconception(row.id, 0.5, status, "2026-01-10", None)
                .unwrap();
        }
        let text = format_misconceptions(&store).unwrap();
        assert!(text.contains("Modern C:"));
        assert!(text.contains("2 active, 1 resolved, 1 disputed (purged)."));
        assert!(text.contains("[ACTIVE] a"));
        assert!(text.contains("[DISPUTED] d"));
    }

    #[test]
    fn upcoming_listing_groups_and_counts_beyond() {
        let mut store = SqliteStore::open_in_memory().unwrap();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "m.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 10,
                },
                "2026-01-01",
            )
            .unwrap();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::PretestReady,
            })
            .unwrap();
        let today = NaiveDate::from_ymd_opt(2026, 1, 10).unwrap();
        let on = |id: i64, date: &str| domain::Task {
            id,
            book_id: book.id,
            chapter_id: chapter.id,
            task_type: domain::TaskType::Pretest,
            scheduled_for: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            status: domain::TaskStatus::Pending,
            completed_at: None,
            sequence: id,
            attempt_no: 1,
        };
        let tasks = vec![
            on(1, "2026-01-11"),
            on(2, "2026-01-11"),
            on(3, "2026-01-20"),
        ];
        let text = format_upcoming(&store, &tasks, today, 2);
        assert!(text.contains("Coming up (next 2 day(s)):"));
        // One date header for the shared date, both tasks under it.
        assert_eq!(text.matches("2026-01-11:").count(), 1);
        assert!(text.contains("[PRETEST] 'Pointers'"));
        assert!(text.contains("…and 1 more task(s) beyond 2026-01-12."));
        let empty = format_upcoming(&store, &[], today, 2);
        assert!(empty.contains("Nothing scheduled in the next 2 day(s)."));
    }

    #[test]
    fn upcoming_tasks_respects_window_edges() {
        use chrono::NaiveDate;
        let today = NaiveDate::from_ymd_opt(2026, 9, 27).unwrap();
        let task = |id: i64, kind: TaskType, date: &str, status: TaskStatus| Task {
            id,
            book_id: 1,
            chapter_id: id,
            task_type: kind,
            scheduled_for: NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap(),
            status,
            completed_at: None,
            sequence: id,
            attempt_no: 1,
        };
        let tasks = vec![
            task(1, TaskType::Pretest, "2026-09-27", TaskStatus::Pending),
            task(2, TaskType::Pretest, "2026-09-28", TaskStatus::Pending),
            task(3, TaskType::Retest, "2026-09-28", TaskStatus::Pending),
            task(4, TaskType::Retest, "2026-09-29", TaskStatus::Pending),
            task(5, TaskType::Notes, "2026-10-05", TaskStatus::Pending),
            task(6, TaskType::Read, "2026-09-28", TaskStatus::Done),
        ];
        // Due-today and done rows are excluded; same-date rows follow §6
        // rank (retest before pretest); the horizon is inclusive.
        let two_day = upcoming_tasks(&tasks, today, 2);
        assert_eq!(
            two_day.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![3, 2, 4]
        );
        // A one-day horizon covers tomorrow only, not today.
        let one_day = upcoming_tasks(&tasks, today, 1);
        assert_eq!(one_day.iter().map(|t| t.id).collect::<Vec<_>>(), vec![3, 2]);
        // A zero horizon covers nothing; a wide one reaches the far task.
        assert_eq!(upcoming_tasks(&tasks, today, 0).len(), 0);
        let wide = upcoming_tasks(&tasks, today, 8);
        assert_eq!(
            wide.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![3, 2, 4, 5]
        );
    }

    /// Picker fixture: ch1 holds Q1 graded + Q2 ungraded; ch2 holds only an
    /// ungraded question. Returns `(graded_chapter_id, ungraded_chapter_id)`.
    fn disputable_setup(store: &mut MemoryStore) -> (i64, i64) {
        let book = store
            .create_book(
                &store::NewBook {
                    title: "T".to_string(),
                    filepath: "/t.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 1,
                },
                "2026-09-25",
            )
            .unwrap();
        let mut chapter = |index: i64, title: &str| {
            store
                .create_chapter(&store::NewChapter {
                    book_id: book.id,
                    index_in_book: index,
                    level: 1,
                    title: title.to_string(),
                    start_page: 1,
                    end_page: 20,
                    file_path: "u.json".to_string(),
                    status: domain::ChapterStatus::AssignmentComplete,
                })
                .unwrap()
        };
        let first = chapter(0, "Ch 1");
        let second = chapter(1, "Ch 2");
        let mut ask = |chapter_id: i64, position: i64| {
            store
                .save_assignment_questions(&[store::NewAssignmentQuestion {
                    chapter_id,
                    position,
                    kind: "written".to_string(),
                    parts_json: "[\"a) ...\"]".to_string(),
                    rubric_json: "{}".to_string(),
                    target_misconception_ids: "[]".to_string(),
                    attempt_no: 1,
                }])
                .unwrap()
                .pop()
                .unwrap()
        };
        let graded = ask(first.id, 0);
        ask(first.id, 1);
        ask(second.id, 0);
        store
            .save_grade(&store::NewGrade {
                question_id: graded.id,
                score: 7,
                max_score: 10,
                classification: "PARTIALLY_CORRECT".to_string(),
                criteria_results_json: "[]".to_string(),
                feedback: "Mostly right.".to_string(),
                grader_version: "v3".to_string(),
                created_at: "2026-09-26".to_string(),
            })
            .unwrap();
        (first.id, second.id)
    }

    #[test]
    fn disputable_chapters_lists_graded_only() {
        let mut store = MemoryStore::new();
        assert!(disputable_chapters(&store).unwrap().is_empty());
        let (graded_id, _) = disputable_setup(&mut store);
        let candidates = disputable_chapters(&store).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].chapter.id, graded_id);
        // Only the graded question appears, at its 1-based position.
        assert_eq!(candidates[0].questions.len(), 1);
        let only = &candidates[0].questions[0];
        assert_eq!(only.position, 1);
        assert_eq!((only.score, only.max_score), (7, 10));
        assert_eq!(only.classification, "PARTIALLY_CORRECT");
    }

    #[test]
    fn parse_pick_accepts_bounds_rejects_garbage() {
        assert_eq!(parse_pick("1", 2).unwrap(), 1);
        assert_eq!(parse_pick("  2  ", 2).unwrap(), 2);
        assert!(parse_pick("0", 2).is_err());
        assert!(parse_pick("3", 2).is_err());
        assert!(parse_pick("x", 2).is_err());
        assert!(parse_pick("", 2).is_err());
    }

    #[test]
    fn resolve_dispute_target_passes_explicit_ids() {
        let mut store = MemoryStore::new();
        assert!(resolve_dispute_target(&store, Some(1), Some(1)).is_err());
        let (graded_id, ungraded_id) = disputable_setup(&mut store);
        assert_eq!(
            resolve_dispute_target(&store, Some(graded_id), Some(1)).unwrap(),
            (graded_id, 1)
        );
        // Q2 exists but has no grade — not disputable.
        assert!(resolve_dispute_target(&store, Some(graded_id), Some(2)).is_err());
        // Ch2 has questions but no grades at all.
        assert!(resolve_dispute_target(&store, Some(ungraded_id), Some(1)).is_err());
        assert!(resolve_dispute_target(&store, Some(999), Some(1)).is_err());
        // No ids under test stdin (non-terminal) fails instead of blocking.
        assert!(resolve_dispute_target(&store, None, None).is_err());
    }

    #[test]
    fn continue_prompt_needs_explicit_yes() {
        assert!(parse_continue_answer("y"));
        assert!(parse_continue_answer("YES"));
        assert!(parse_continue_answer("  yes  "));
        assert!(!parse_continue_answer(""));
        assert!(!parse_continue_answer("n"));
        assert!(!parse_continue_answer("q"));
        assert!(!parse_continue_answer("yeah"));
    }

    #[test]
    fn stored_notes_returns_latest_and_fails_loudly() {
        let mut store = MemoryStore::new();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "T".to_string(),
                    filepath: "/t.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 1,
                },
                "2026-09-25",
            )
            .unwrap();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: book.id,
                index_in_book: 0,
                level: 1,
                title: "Ch 1".to_string(),
                start_page: 1,
                end_page: 20,
                file_path: "u.json".to_string(),
                status: domain::ChapterStatus::Completed,
            })
            .unwrap();
        assert!(stored_notes_markdown(&store, 999).is_err());
        assert!(stored_notes_markdown(&store, chapter.id).is_err());
        let note = |body: &str| store::NewNote {
            chapter_id: chapter.id,
            content_markdown: body.to_string(),
            generated_at: "2026-09-26".to_string(),
            attempt_no: 1,
        };
        store.save_note(&note("first")).unwrap();
        store.save_note(&note("second")).unwrap();
        assert_eq!(stored_notes_markdown(&store, chapter.id).unwrap(), "second");
    }

    fn purge_question(targets_json: &str) -> store::AssignmentQuestion {
        store::AssignmentQuestion {
            id: 7,
            chapter_id: 1,
            position: 0,
            kind: "written".to_string(),
            parts_json: "[]".to_string(),
            rubric_json: "{}".to_string(),
            target_misconception_ids: targets_json.to_string(),
            attempt_no: 1,
        }
    }

    #[test]
    fn dispute_purge_revised_clears_targets_and_logged_row() {
        let mut store = MemoryStore::new();
        let target = store
            .create_misconception(
                1,
                "aliasing",
                "addr vs value",
                "picked trap",
                "RETEST",
                "2026-09-25",
            )
            .unwrap();
        let logged = store
            .create_misconception(
                1,
                "Assignment Q1: Explain &x",
                "Graded INCORRECT 2/8: confused address with value",
                "Answer excerpt: I think &x gives the value",
                "ASSIGNMENT",
                "2026-09-25",
            )
            .unwrap();
        let resolved = store
            .create_misconception(
                1,
                "scales",
                "only 1,2,4,8",
                "fixed earlier",
                "RETEST",
                "2026-09-24",
            )
            .unwrap();
        store
            .update_misconception(
                resolved.id,
                0.9,
                "RESOLVED",
                "2026-09-25",
                Some("2026-09-25"),
            )
            .unwrap();
        // Unknown target 999 is stale — reported and skipped, never fatal.
        let question = purge_question(&format!("[{}, 999]", target.id));
        let purged = apply_dispute_purge(
            &mut store,
            &question,
            dispute::DisputeAction::Revised,
            "2026-09-26",
        )
        .unwrap();
        assert_eq!(purged, 2);
        let rows = store.list_misconceptions(1).unwrap();
        let status_of = |id: i64| {
            rows.iter()
                .find(|row| row.id == id)
                .map(|row| row.status.clone())
                .unwrap()
        };
        assert_eq!(status_of(target.id), "DISPUTED");
        assert_eq!(status_of(logged.id), "DISPUTED");
        // Resolved on earlier evidence — the bad grade never touched it.
        assert_eq!(status_of(resolved.id), "RESOLVED");
        // Confidence is preserved through the purge, not zeroed.
        let purged_row = rows.iter().find(|row| row.id == target.id).unwrap();
        assert!((purged_row.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn dispute_purge_upheld_purges_nothing_corrupt_targets_still_purge_log() {
        let mut store = MemoryStore::new();
        let logged = store
            .create_misconception(
                1,
                "Assignment Q1: Explain &x",
                "Graded INCORRECT 2/8: confused address with value",
                "Answer excerpt: I think &x gives the value",
                "ASSIGNMENT",
                "2026-09-25",
            )
            .unwrap();
        let question = purge_question(&format!("[{}]", logged.id));
        let purged = apply_dispute_purge(
            &mut store,
            &question,
            dispute::DisputeAction::Upheld,
            "2026-09-26",
        )
        .unwrap();
        assert_eq!(purged, 0);
        let corrupt = purge_question("{broken");
        let purged = apply_dispute_purge(
            &mut store,
            &corrupt,
            dispute::DisputeAction::QuestionDefective,
            "2026-09-26",
        )
        .unwrap();
        assert_eq!(purged, 1);
        let rows = store.list_misconceptions(1).unwrap();
        assert_eq!(rows[0].status, "DISPUTED");
    }

    fn evidence_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "fixture setup dominates; splitting the fixture from its assertions hurts readability"
    )]
    fn chapter_evidence_aggregates_store_rows() {
        let mut store = MemoryStore::new();
        let chapter = store
            .create_chapter(&store::NewChapter {
                book_id: 1,
                index_in_book: 0,
                level: 1,
                title: "Pointers".to_string(),
                start_page: 10,
                end_page: 19,
                file_path: "unit_0.json".to_string(),
                status: domain::ChapterStatus::Completed,
            })
            .unwrap();
        let first = store
            .create_task(&store::NewTask {
                book_id: 1,
                chapter_id: chapter.id,
                task_type: TaskType::Retest,
                scheduled_for: evidence_date(2026, 9, 20),
                sequence: 1,
                attempt_no: 1,
            })
            .unwrap();
        let second = store
            .create_task(&store::NewTask {
                book_id: 1,
                chapter_id: chapter.id,
                task_type: TaskType::AssignmentWrite,
                scheduled_for: evidence_date(2026, 9, 20),
                sequence: 2,
                attempt_no: 1,
            })
            .unwrap();
        let done = store.complete_task(first.id, "2026-09-20").unwrap();
        assert_eq!(done.status, TaskStatus::Done);
        store.complete_task(second.id, "2026-09-21").unwrap();
        let items = store
            .save_mcq_items(&[
                store::NewMcqItem {
                    chapter_id: chapter.id,
                    phase: "pretest".to_string(),
                    question_text: "What does &x yield?".to_string(),
                    options_json: "[\"a\",\"b\",\"c\",\"d\"]".to_string(),
                    correct_index: 0,
                    trap_index: 1,
                    explanation_text: "Address-of.".to_string(),
                    source_refs: "{}".to_string(),
                    topic: "aliasing".to_string(),
                    attempt_no: 1,
                },
                store::NewMcqItem {
                    chapter_id: chapter.id,
                    phase: "pretest".to_string(),
                    question_text: "What is a dangling pointer?".to_string(),
                    options_json: "[\"a\",\"b\",\"c\",\"d\"]".to_string(),
                    correct_index: 0,
                    trap_index: 2,
                    explanation_text: "Freed memory.".to_string(),
                    source_refs: "{}".to_string(),
                    topic: "lifetimes".to_string(),
                    attempt_no: 1,
                },
            ])
            .unwrap();
        store
            .record_mcq_response(items[0].id, 0, true, false, "2026-09-20", 1)
            .unwrap();
        store
            .record_mcq_response(items[1].id, 1, false, true, "2026-09-20", 1)
            .unwrap();
        let questions = store
            .save_assignment_questions(&[store::NewAssignmentQuestion {
                chapter_id: chapter.id,
                position: 0,
                kind: "written".to_string(),
                parts_json: "[]".to_string(),
                rubric_json: "{}".to_string(),
                target_misconception_ids: "[]".to_string(),
                attempt_no: 1,
            }])
            .unwrap();
        let grade = store
            .save_grade(&store::NewGrade {
                question_id: questions[0].id,
                score: 6,
                max_score: 8,
                classification: "PARTIALLY_CORRECT".to_string(),
                criteria_results_json: "[]".to_string(),
                feedback: "Half right.".to_string(),
                grader_version: "v3".to_string(),
                created_at: "2026-09-21".to_string(),
            })
            .unwrap();
        store
            .record_dispute(&store::NewDispute {
                grade_id: grade.id,
                text: "Rubric misapplied.".to_string(),
                decision: "REVISED".to_string(),
                final_score: 8,
                adjudication_json: "{}".to_string(),
                adjudicator_model: "test".to_string(),
                timestamp: "2026-09-22".to_string(),
            })
            .unwrap();
        store
            .create_misconception(
                chapter.id,
                "aliasing",
                "addr vs value",
                "picked trap",
                "RETEST",
                "2026-09-20",
            )
            .unwrap();
        let fixed = store
            .create_misconception(
                chapter.id,
                "scales",
                "only 1,2,4,8",
                "fixed",
                "RETEST",
                "2026-09-19",
            )
            .unwrap();
        store
            .update_misconception(fixed.id, 0.9, "RESOLVED", "2026-09-20", Some("2026-09-20"))
            .unwrap();

        let ev = chapter_evidence(&store, &chapter).unwrap();
        assert_eq!(ev.pages, 10);
        assert_eq!(ev.completed_on, Some(evidence_date(2026, 9, 21)));
        assert_eq!((ev.pretest_correct, ev.pretest_answered), (1, 2));
        assert_eq!((ev.retest_correct, ev.retest_answered), (0, 0));
        // The disputed grade counts at its corrected award, not the original.
        assert_eq!((ev.assignment_earned, ev.assignment_possible), (8, 8));
        assert_eq!(
            (ev.active_misconceptions, ev.resolved_misconceptions),
            (1, 1)
        );
    }

    #[test]
    fn single_book_guard_summarizes_progress() {
        let mut store = MemoryStore::new();
        let book = store
            .create_book(
                &store::NewBook {
                    title: "Modern C".to_string(),
                    filepath: "modern.pdf".to_string(),
                    file_hash: "h".to_string(),
                    start_page: 18,
                },
                "2026-09-29",
            )
            .unwrap();
        for (index, (start, end, status)) in [
            (18, 37, domain::ChapterStatus::Completed),
            (38, 43, domain::ChapterStatus::Skipped),
            (44, 53, domain::ChapterStatus::PretestReady),
        ]
        .iter()
        .copied()
        .enumerate()
        {
            store
                .create_chapter(&store::NewChapter {
                    book_id: book.id,
                    index_in_book: i64::try_from(index).unwrap_or(0),
                    level: 1,
                    title: format!("Ch {index}"),
                    start_page: start,
                    end_page: end,
                    file_path: "u.json".to_string(),
                    status,
                })
                .unwrap();
        }
        let progress = book_progress(&store, &book).unwrap();
        assert_eq!(progress.done, 1);
        assert_eq!(progress.total, 3);
        assert_eq!(progress.skipped, 1);
        assert_eq!(progress.pages_done, 20);
        assert_eq!(progress.pages_total, 36);
        assert_eq!(
            format_book_progress(&progress),
            "  Modern C: 1/3 (33%) chapters, 1 skipped, 20/36 pages"
        );
        let empty = store
            .create_book(
                &store::NewBook {
                    title: "Empty".to_string(),
                    filepath: "e.pdf".to_string(),
                    file_hash: "e".to_string(),
                    start_page: 1,
                },
                "2026-09-29",
            )
            .unwrap();
        assert_eq!(
            format_book_progress(&book_progress(&store, &empty).unwrap()),
            "  Empty: 0/0 (n/a) chapters, 0/0 pages"
        );
    }

    #[test]
    fn metrics_commands_succeed_on_empty_store() {
        let store = SqliteStore::open_in_memory().unwrap();
        let today = evidence_date(2026, 9, 27);
        run_metrics(&store, today).unwrap();
        run_misconceptions(&store).unwrap();
        run_progress(&store).unwrap();
    }
}
