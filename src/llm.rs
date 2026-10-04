//! LLM layer (§16 + §2.1): provider abstraction, durable jobs, cache, retry.
//!
//! Engines never touch the network directly: [`RawTransport`] is the
//! injectable boundary. [`HttpLlmProvider`] is the real `reqwest` blocking
//! implementation (Vercel AI Gateway, direct Google, or any
//! OpenAI-compatible endpoint — all spoken as `/chat/completions`);
//! `MockLlm` (test-only, below) exists
//! only for deterministic unit tests of retry/backoff/validation logic.
//!
//! Every request flows through [`complete_cached`]: cache lookup (with
//! caller-supplied revalidation) → durable [`crate::store`] job row → bounded
//! transport retries with backoff → validation-gated cache write. Malformed
//! responses are never cached and are retried immediately without backoff.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
#[cfg(test)]
use std::collections::VecDeque;
use std::io::IsTerminal;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::store::{NewLlmJob, Store};

/// Bump when prompt templates change; part of the cache identity so stale
/// prompts never collide with fresh ones.
pub const PROMPT_VERSION: &str = "v3";
/// Provider id recorded in job rows and hashed into the cache identity.
pub const PROVIDER_ID: &str = "ai-gateway";
/// Provider id for direct Google calls (OpenAI-compatible endpoint).
pub const GEMINI_PROVIDER_ID: &str = "google";
/// Provider id for any other OpenAI-compatible endpoint (local servers,
/// self-hosted gateways, other vendors). No vendor-specific behavior: plain
/// `/chat/completions` + bearer auth (which may be empty for no-auth local
/// servers) + the same `response_format` envelope as the `google` path.
pub const CUSTOM_PROVIDER_ID: &str = "custom";
/// Free-tier default (smoke-tested 2026-09-20, cost 0). See
/// `.scratch/free_models.md` for the full free list.
pub const DEFAULT_MODEL: &str = "poolside/laguna-s-2.1-free";
/// Paid-credits-only upgrade target (403 for free-tier users).
pub const UPGRADE_MODEL: &str = "zai/glm-5.3-flash";
/// OpenAI-compatible base URL for the gateway.
pub const DEFAULT_ENDPOINT: &str = "https://ai-gateway.vercel.sh/v1";
/// OpenAI-compatible base URL for direct Google calls (`GOOGLE_API_KEY`).
pub const GEMINI_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/openai";
/// Default base URL for the generic path: a local OpenAI-compatible server
/// (llama.cpp, Ollama, vLLM, ...) with no auth. Override with
/// `CADENCE_LLM_ENDPOINT` for any other endpoint.
pub const CUSTOM_ENDPOINT: &str = "http://localhost:8080/v1";
/// Default model on the direct Google path (user-selected 2026-09-21;
/// verified live: 3.6-flash serves this key, 2.0-flash 404s as retired,
/// 3.7-flash 503s).
pub const GEMINI_DEFAULT_MODEL: &str = "gemini-3.6-flash";
/// Transport attempts: 1 initial + 4 retries with backoff (§16).
pub const MAX_TRANSPORT_ATTEMPTS: u32 = 5;
/// Per-request HTTP timeout on the hosted paths (fast APIs).
pub const DEFAULT_TIMEOUT_SECS: u64 = 90;
/// Per-request HTTP timeout on the generic path: local models decode slowly
/// (measured 2026-09-25: ~39 tok/s on a 25B Q6, ~5 min prompt eval for a
/// 25k-token unit + minutes of constrained decode) and a short timeout
/// turns into cancel-and-retry churn server-side. Override any path with
/// `CADENCE_LLM_TIMEOUT_S`.
pub const CUSTOM_TIMEOUT_SECS: u64 = 600;
/// Malformed-response fast retries: no backoff, validation error appended.
pub const MAX_MALFORMED_ATTEMPTS: u32 = 3;
/// Backoff base: `base * 2^attempt + jitter`, capped below.
pub const BACKOFF_BASE_MS: u64 = 1_000;
/// Backoff / `Retry-After` ceiling.
pub const BACKOFF_CAP_MS: u64 = 30_000;
/// Upper bound quoted when quoting a provider error body.
pub const ERROR_BODY_CHARS: usize = 500;
/// Braille frames for the whimsical LLM spinner (one full whimsy rotation).
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Frame cadence for the spinner thread.
const SPINNER_INTERVAL_MS: u64 = 80;

/// One spinner frame by index, wrapping around without ever panicking.
/// Pure so unit tests can pin the rotation.
#[must_use]
pub fn spinner_frame_at(index: usize) -> &'static str {
    let len = SPINNER_FRAMES.len();
    if len == 0 {
        return "";
    }
    let pos = index.checked_rem(len).unwrap_or_default();
    SPINNER_FRAMES.get(pos).copied().unwrap_or("")
}

/// Whether the spinner stays off for the given env snapshots:
/// `CADENCE_NO_SPINNER=1`/`true`/`yes` forces quiet, as does `TERM=dumb`.
/// Pure so unit tests can pin the gating without touching the process env.
#[must_use]
pub fn spinner_suppressed(no_spinner_raw: &str, term_raw: &str) -> bool {
    parse_disable_thinking(no_spinner_raw) || term_raw.trim().eq_ignore_ascii_case("dumb")
}

/// One rendered spinner line (`"⠋ consulting model (3s)…"`). Pure for tests.
#[must_use]
pub fn format_spinner_line(frame: &str, message: &str, elapsed_secs: u64) -> String {
    format!("{frame} {message} ({elapsed_secs}s)…")
}

/// Whimsical CLI spinner shown while an LLM HTTP call is in flight.
///
/// Created at the top of [`HttpLlmProvider`] network calls and stopped by
/// `Drop`, so every early `return` still clears the line. Writes to stderr
/// (stdout stays pipe-clean) and only when stderr is an interactive
/// terminal — piped/CI runs and `CADENCE_NO_SPINNER=1` / `TERM=dumb` stay
/// silent. Unit tests using `MockLlm` never touch this; `HttpLlmProvider`
/// is the only constructor caller.
pub struct LlmSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    message: Option<String>,
    started: Instant,
}

impl LlmSpinner {
    /// Start spinning with `message` (e.g. `"✨ consulting <model>"`), unless
    /// suppressed (see [`spinner_suppressed`]) or stderr is not a terminal —
    /// then returns an inactive guard whose `Drop` is a no-op.
    #[must_use]
    pub fn start(message: String) -> Self {
        let no_spinner = std::env::var("CADENCE_NO_SPINNER").unwrap_or_default();
        let term = std::env::var("TERM").unwrap_or_default();
        if spinner_suppressed(&no_spinner, &term) || !std::io::stderr().is_terminal() {
            return Self::inactive();
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let line_message = message.clone();
        let thread_start = Instant::now();
        let handle = std::thread::spawn(move || {
            let mut index: usize = 0;
            while !thread_stop.load(Ordering::Relaxed) {
                let frame = spinner_frame_at(index);
                let elapsed = thread_start.elapsed().as_secs();
                let line = format_spinner_line(frame, &line_message, elapsed);
                {
                    use std::io::Write as _;
                    let mut err = std::io::stderr();
                    let _ = write!(err, "\r{line}");
                    let _ = err.flush();
                }
                std::thread::sleep(Duration::from_millis(SPINNER_INTERVAL_MS));
                index = index.wrapping_add(1);
            }
        });
        Self {
            stop,
            handle: Some(handle),
            message: Some(message),
            started: Instant::now(),
        }
    }

    /// Inactive guard: `Drop` does nothing. Used for suppressed output and
    /// unit tests of the no-op path.
    #[must_use]
    pub fn inactive() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(true)),
            handle: None,
            message: None,
            started: Instant::now(),
        }
    }

    /// Whether a spinner thread is running behind this guard.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.handle.is_some()
    }
}

impl Drop for LlmSpinner {
    fn drop(&mut self) {
        if !self.is_active() {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let elapsed = self.started.elapsed().as_secs();
        let done = self.message.take().unwrap_or_default();
        {
            use std::io::Write as _;
            let mut err = std::io::stderr();
            let _ = writeln!(err, "\r\x1b[2K✓ {done} ({elapsed}s)");
            let _ = err.flush();
        }
    }
}

/// Cache identity: `SHA256(provider + model + task_type + prompt_version +
/// source_content_hash + parameters)` (§16). `operation` is the task type
/// (`smoke`, `pretest`, ...); `source_hash` is the hex SHA-256 of the source
/// material (empty for sourceless smoke prompts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestIdentity {
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Stage label.
    pub operation: String,
    /// [`PROMPT_VERSION`] at call time.
    pub prompt_version: String,
    /// Hex hash of the source material.
    pub source_hash: String,
    /// Canonical extra params JSON (e.g. `{"max_tokens":2000}`).
    pub params_json: String,
}

impl RequestIdentity {
    /// Hex cache hash over the identity fields (`\0`-separated).
    #[must_use]
    pub fn cache_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for part in [
            self.provider.as_str(),
            self.model.as_str(),
            self.operation.as_str(),
            self.prompt_version.as_str(),
            self.source_hash.as_str(),
            self.params_json.as_str(),
        ] {
            hasher.update(part.as_bytes());
            hasher.update([0_u8]);
        }
        hex::encode(hasher.finalize())
    }
}

/// Hex SHA-256 of a prompt (the `input_hash` recorded on job rows).
#[must_use]
pub fn input_hash(prompt: &str) -> String {
    hex::encode(Sha256::digest(prompt.as_bytes()))
}

/// Whether an HTTP status is worth retrying with backoff: 429 and 5xx only.
/// 4xx (auth/config) never retries (§16).
#[must_use]
pub const fn is_retryable_status(code: u16) -> bool {
    code == 429 || (code >= 500 && code <= 599)
}

/// Exponential backoff with jitter, capped: `min(base * 2^attempt + jitter,
/// cap)`. Pure and saturating — safe to unit-test.
#[must_use]
pub const fn backoff_delay_ms(base_ms: u64, attempt: u32, jitter_ms: u64) -> u64 {
    let grown = base_ms.saturating_mul(2_u64.saturating_pow(attempt));
    let total = grown.saturating_add(jitter_ms);
    if total > BACKOFF_CAP_MS {
        BACKOFF_CAP_MS
    } else {
        total
    }
}

/// Parse a `Retry-After` header value (whole seconds) into a capped delay.
/// Returns `None` on garbage so callers fall back to [`backoff_delay_ms`].
#[must_use]
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let secs: u64 = value.trim().parse().ok()?;
    Some(Duration::from_millis(
        secs.saturating_mul(1_000).min(BACKOFF_CAP_MS),
    ))
}

/// Small clock-derived jitter for production backoff (tests inject exact
/// values into [`backoff_delay_ms`] instead).
#[must_use]
pub fn clock_jitter_ms(ceiling_ms: u64) -> u64 {
    let bound = ceiling_ms.saturating_add(1);
    if bound <= 1 {
        return 0;
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or_default();
    nanos.checked_rem(bound).unwrap_or_default()
}

/// Raw provider outcome, surfaced so the orchestrator can apply §16 policy
/// (backoff vs. fast retry vs. fatal) instead of guessing from strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderOutcome {
    /// Raw response text (still needs caller validation).
    Ok(String),
    /// Retry with backoff; carries a parsed `Retry-After` delay if present.
    Retryable {
        /// Human-readable cause.
        message: String,
        /// Honored before the computed backoff when present.
        retry_after: Option<Duration>,
    },
    /// Never retry (4xx auth/config, bad request shape).
    Fatal(String),
}

/// Stop diagnostics from one chat-completions response: how generation
/// ended (`finish_reason`: `stop` | `length` | ...) plus the server's
/// backend reason when present (`eos_reason`: `TabbyAPI` reports values like
/// `end_filter` on clean stops; a loop-detector kill shows up here).
/// Recorded on job-row attempts so truncated/loop-cut responses stay
/// debuggable from the DB instead of needing packet forensics. No
/// cleanliness judgment here — just facts; policy never branches on these.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StopInfo {
    /// `choices[0].finish_reason`, when the envelope carries one.
    pub finish_reason: Option<String>,
    /// `choices[0].eos_reason`, when the server reports one.
    pub eos_reason: Option<String>,
}

impl StopInfo {
    /// One-line diagnostics note for the attempt `error` column
    /// (`"stop finish=length eos=loop"`; missing halves render as `-`).
    /// Shares that column with validation/transport notes — no schema bump.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "stop finish={} eos={}",
            self.finish_reason.as_deref().unwrap_or("-"),
            self.eos_reason.as_deref().unwrap_or("-"),
        )
    }
}

/// Injectable transport boundary. Engines and [`complete_cached`] program
/// against this; the network lives only in [`HttpLlmProvider`].
pub trait RawTransport {
    /// Send one prompt; never retries internally.
    fn send(&self, prompt: &str) -> ProviderOutcome;
    /// Provider id for job rows / cache identity.
    fn provider_id(&self) -> &str;
    /// Model id for job rows / cache identity.
    fn model_id(&self) -> &str;
    /// Stop diagnostics from the most recent [`Self::send`] call, if the
    /// transport tracks any. [`complete_cached`] records these on the
    /// attempt row so abnormal stops (token-budget `length` cuts,
    /// loop-detector kills) are visible without re-probing. Defaults to
    /// `None` (e.g. [`MockLlm`] carries no envelope).
    fn last_stop_info(&self) -> Option<StopInfo> {
        None
    }
}

/// Connection settings for [`HttpLlmProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmConfig {
    /// Provider id for job rows / cache identity
    /// (`ai-gateway` | `google` | `custom`).
    pub provider: String,
    /// Base URL (`.../v1`, `/chat/completions` is appended).
    pub endpoint: String,
    /// Model slug.
    pub model: String,
    /// Bearer token (may be empty on the generic path: no-auth local
    /// servers ignore it).
    pub api_key: String,
    /// Upper bound on completion tokens per call.
    pub max_tokens: u32,
    /// Per-request HTTP timeout in seconds.
    pub timeout_secs: u64,
    /// When true, ask templated reasoning models to skip the thinking trace
    /// (`chat_template_kwargs.enable_thinking = false`). Sent only on the
    /// generic path — a llama.cpp/template extension, not portable
    /// `OpenAI` semantics — when `CADENCE_DISABLE_THINKING=1`, and always
    /// for models covered by [`model_disables_thinking`] (Gemma: thinking
    /// corrupts long structured output). Saves completion budget (which
    /// reasoning would otherwise share with the answer) at unknown quality
    /// cost; default off (thinking stays on).
    pub disable_thinking: bool,
    /// Thinking budget for templated reasoning models
    /// (`chat_template_kwargs.thinking_effort`): one of
    /// `low`/`medium`/`high`/`xhigh`, default `medium` (from
    /// `CADENCE_THINKING_EFFORT`, unrecognized values fall back to medium).
    /// Sent only on the generic path alongside `enable_thinking: true`, and
    /// never to models covered by [`model_disables_thinking`] (Gemma gets
    /// a lone `enable_thinking: false`). Never sent when
    /// [`Self::disable_thinking`] opts out.
    pub thinking_effort: &'static str,
    /// Optional `response_format` body value (a [`response_format_envelope`]
    /// JSON string constraining the model to a schema). Attached on the
    /// `google` and `custom` provider paths — never on the gateway path,
    /// which fronts third-party models that may not accept the field. Live-
    /// verified on Gemini (2026-09-22) and on llama.cpp (2026-09-25); other
    /// `custom` servers are expected to accept the standard `OpenAI`
    /// `json_schema` envelope but this is not guaranteed for every server.
    pub response_format_json: Option<String>,
}

impl LlmConfig {
    /// Build from the environment. Key selection: `AI_GATEWAY_API_KEY` wins
    /// when set (gateway endpoint + Laguna default); otherwise
    /// `GOOGLE_API_KEY` selects the direct Google path (Gemini endpoint +
    /// flash default); otherwise the generic OpenAI-compatible path is
    /// selected when `CADENCE_API_KEY` / `OPENAI_API_KEY` or
    /// `CADENCE_LLM_ENDPOINT` is set (default endpoint
    /// `http://localhost:8080/v1`, no-auth allowed with an empty key —
    /// llama.cpp ignores bearer auth). `CADENCE_LLM_MODEL` /
    /// `CADENCE_LLM_ENDPOINT` override the selected path's defaults, except
    /// that the generic path has no model default: `CADENCE_LLM_MODEL` is
    /// required there since no single default fits every server. Each source
    /// is read from the process environment first, then from a `./.env` file
    /// in the working directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LlmFatal`] when no API key is configured.
    pub fn from_env() -> Result<Self> {
        let gateway_key = Self::key_from_env("AI_GATEWAY_API_KEY");
        let google_key = Self::key_from_env("GOOGLE_API_KEY");
        let custom_key = Self::key_from_env("CADENCE_API_KEY");
        let custom_key = if custom_key.trim().is_empty() {
            Self::key_from_env("OPENAI_API_KEY")
        } else {
            custom_key
        };
        let model = Self::value_from_env("CADENCE_LLM_MODEL");
        let endpoint = Self::value_from_env("CADENCE_LLM_ENDPOINT");
        let mut config = Self::resolve(&gateway_key, &google_key, &custom_key, &model, &endpoint)?;
        let timeout = Self::value_from_env("CADENCE_LLM_TIMEOUT_S");
        config.timeout_secs = parse_timeout_secs(&timeout, config.timeout_secs);
        config.disable_thinking = parse_disable_thinking(
            Self::value_from_env("CADENCE_DISABLE_THINKING").as_str(),
        );
        config.thinking_effort = parse_thinking_effort(
            Self::value_from_env("CADENCE_THINKING_EFFORT").as_str(),
        );
        Ok(config)
    }

    /// Read one API key from the environment, falling back to `./.env`.
    fn key_from_env(name: &str) -> String {
        Self::value_from_env(name)
    }

    /// Read any config value from the environment, falling back to `./.env`.
    fn value_from_env(name: &str) -> String {
        let live = std::env::var(name).unwrap_or_default();
        if !live.trim().is_empty() {
            return live;
        }
        dotenv_key(".env", name).unwrap_or_default()
    }

    /// Pure key/path resolution behind [`Self::from_env`] (unit-testable:
    /// empty strings mean "not configured"). Priority is gateway, then
    /// Google, then the generic path — selected when `custom_key` or
    /// `endpoint_override` is set (explicit local intent; the key may be
    /// empty for no-auth servers). The generic path requires
    /// `model_override`: there is no universal model default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LlmFatal`] when no path is configured, or when the
    /// generic path is selected without a model.
    pub fn resolve(
        gateway_key: &str,
        google_key: &str,
        custom_key: &str,
        model_override: &str,
        endpoint_override: &str,
    ) -> Result<Self> {
        if !gateway_key.trim().is_empty() {
            return Ok(Self {
                provider: PROVIDER_ID.to_string(),
                endpoint: defaulted(endpoint_override, DEFAULT_ENDPOINT),
                model: defaulted(model_override, DEFAULT_MODEL),
                api_key: gateway_key.trim().to_string(),
                max_tokens: 2_000,
                timeout_secs: DEFAULT_TIMEOUT_SECS,
                disable_thinking: false,
                thinking_effort: "medium",
                response_format_json: None,
            });
        }
        if !google_key.trim().is_empty() {
            return Ok(Self {
                provider: GEMINI_PROVIDER_ID.to_string(),
                endpoint: defaulted(endpoint_override, GEMINI_ENDPOINT),
                model: defaulted(model_override, GEMINI_DEFAULT_MODEL),
                api_key: google_key.trim().to_string(),
                max_tokens: 2_000,
                timeout_secs: DEFAULT_TIMEOUT_SECS,
                disable_thinking: false,
                thinking_effort: "medium",
                response_format_json: None,
            });
        }
        if !custom_key.trim().is_empty() || !endpoint_override.trim().is_empty() {
            if model_override.trim().is_empty() {
                return Err(Error::LlmFatal(
                    "custom LLM endpoint needs CADENCE_LLM_MODEL (no default fits every server)"
                        .to_string(),
                ));
            }
            return Ok(Self {
                provider: CUSTOM_PROVIDER_ID.to_string(),
                endpoint: defaulted(endpoint_override, CUSTOM_ENDPOINT),
                model: model_override.trim().to_string(),
                api_key: custom_key.trim().to_string(),
                max_tokens: 2_000,
                timeout_secs: CUSTOM_TIMEOUT_SECS,
                disable_thinking: false,
                thinking_effort: "medium",
                response_format_json: None,
            });
        }
        Err(Error::LlmFatal(
            "missing API key (export AI_GATEWAY_API_KEY, GOOGLE_API_KEY, or CADENCE_API_KEY/OPENAI_API_KEY, or point CADENCE_LLM_ENDPOINT at an OpenAI-compatible server — or add one to ./.env)"
                .to_string(),
        ))
    }

    /// Override the model (e.g. `--model` / upgrade path to [`UPGRADE_MODEL`]).
    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

/// Override-or-default for [`LlmConfig::resolve`]: blank overrides fall back.
fn defaulted(override_value: &str, default: &str) -> String {
    if override_value.trim().is_empty() {
        default.to_string()
    } else {
        override_value.trim().to_string()
    }
}

/// Parse `CADENCE_LLM_TIMEOUT_S`: positive integers win, anything else
/// (blank, garbage, zero) keeps the path default — a broken timeout must
/// never silently become "no timeout".
#[must_use]
pub fn parse_timeout_secs(raw: &str, default_secs: u64) -> u64 {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default_secs;
    }
    trimmed.parse::<u64>().map_or(default_secs, |secs| {
        if secs == 0 { default_secs } else { secs }
    })
}

/// Parse `CADENCE_DISABLE_THINKING`: `1`/`true`/`yes` (case-insensitive)
/// opt out of the thinking trace; everything else keeps it.
#[must_use]
pub fn parse_disable_thinking(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

/// Parse `CADENCE_THINKING_EFFORT`: `low`/`medium`/`high`/`xhigh`
/// (case-insensitive, surrounding whitespace ignored) select the
/// template thinking budget. Blank or unrecognized values fall back to
/// `medium` — a broken setting must never silently become the server's
/// maximum-effort default.
#[must_use]
pub fn parse_thinking_effort(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "low" => "low",
        "high" => "high",
        "xhigh" => "xhigh",
        _ => "medium",
    }
}

/// Look up `key` in dotenv-format text (`KEY=value`, `#` comments, optional
/// quotes). Returns `None` when absent or malformed.
#[must_use]
pub fn parse_dotenv_key(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line.split_once('=')?;
        if name.trim() != key {
            continue;
        }
        let value = value.trim().trim_matches('"').trim_matches('\'').trim();
        if value.is_empty() {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

/// Read one key from a dotenv file; `None` when the file is unreadable or the
/// key is absent. Never logs or prints the value.
#[must_use]
pub fn dotenv_key(path: &str, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    parse_dotenv_key(&text, key)
}

/// Wrap a JSON Schema in a `response_format` envelope for constrained
/// decoding (`strict` requires every object property listed in
/// `required` with `additionalProperties: false` — all stage builders below
/// comply). Live-verified on the Gemini compat endpoint (2026-09-22).
#[must_use]
pub fn response_format_envelope(schema_json: &str, name: &str) -> String {
    let schema: serde_json::Value =
        serde_json::from_str(schema_json).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": true},
    })
    .to_string()
}

/// Short content tag for a schema (8 hex chars): recorded in stage
/// `params_json` so constrained and unconstrained calls never share a cache
/// identity, without storing the whole schema in every params string.
#[must_use]
pub fn schema_tag(schema_json: &str) -> String {
    hex::encode(Sha256::digest(schema_json.as_bytes()))
        .chars()
        .take(8)
        .collect()
}

/// Whether thinking must stay OFF for a custom-path model, regardless of
/// the global thinking policy. Gemma templates are on/off only (no effort
/// budget), and thinking-on still corrupts their long structured
/// generations: live-verified 2026-09-28, Gemma 4 with the fixed Jinja
/// template degenerated into endlessly repeated explanation text
/// (`"and so on and so on ..."`) on the 8-question MCQ task 3/3 times with
/// a lone `enable_thinking: true`, while `enable_thinking: false`
/// produced a clean validated set in 1 send. (Short synthetic probes pass
/// with thinking on — the failure is specific to long constrained output.)
/// Matching is a case-insensitive substring on the model slug.
#[must_use]
pub fn model_disables_thinking(model: &str) -> bool {
    model.to_ascii_lowercase().contains("gemma")
}

/// Build the `/chat/completions` request body: model, messages, token cap,
/// plus the configured `response_format` — attached on every provider path
/// except the gateway (third-party models behind it may not accept the
/// field). Thinking controls ride in `chat_template_kwargs` on the generic
/// path only (a llama.cpp/template extension): thinking on plus the
/// configured effort by default — server defaults vary by model (Gemma
/// thinks nothing, Qwen maxes out), so the body is explicit — except a
/// lone `enable_thinking: false` under `CADENCE_DISABLE_THINKING=1` or for
/// models that must not think at all (see [`model_disables_thinking`]). No
/// `temperature` is sent: the server default applies (unlike Google's tuned structured-output defaults,
/// generic servers pick their own sampling; determinism here comes from the
/// JSON-only contract + validated retry, documented per stage).
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] when the configured `response_format` is not
/// valid JSON (practically unreachable: stage builders emit it via `json!`).
fn chat_body(config: &LlmConfig, prompt: &str) -> Result<serde_json::Value> {
    let mut body = serde_json::json!({
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
    });
    if config.provider != PROVIDER_ID {
        if let Some(format) = config.response_format_json.as_deref() {
            let value: serde_json::Value = serde_json::from_str(format)
                .map_err(|e| Error::LlmFatal(format!("bad response_format: {e}")))?;
            if let Some(object) = body.as_object_mut() {
                object.insert("response_format".to_string(), value);
            }
        }
        if config.provider == CUSTOM_PROVIDER_ID {
            if config.disable_thinking || model_disables_thinking(&config.model) {
                if let Some(object) = body.as_object_mut() {
                    object.insert(
                        "chat_template_kwargs".to_string(),
                        serde_json::json!({"enable_thinking": false}),
                    );
                }
            } else if let Some(object) = body.as_object_mut() {
                object.insert(
                    "chat_template_kwargs".to_string(),
                    serde_json::json!({
                        "enable_thinking": true,
                        "thinking_effort": config.thinking_effort,
                    }),
                );
            }
        }
    }
    Ok(body)
}

/// Real HTTP transport: `reqwest` blocking client against an
/// OpenAI-compatible `/chat/completions` endpoint. Fully synchronous.
#[derive(Debug)]
pub struct HttpLlmProvider {
    client: reqwest::blocking::Client,
    config: LlmConfig,
    /// Stop diagnostics from the most recent [`RawTransport::send`] (empty
    /// before the first send and after any send without a parsed envelope).
    /// Synchronous transport, so no interleaving: [`complete_cached`] reads
    /// this right after each `Ok` to annotate the attempt row.
    last_stop: RefCell<Option<StopInfo>>,
}

impl HttpLlmProvider {
    /// Build with the configured per-request timeout
    /// ([`DEFAULT_TIMEOUT_SECS`] on hosted paths, [`CUSTOM_TIMEOUT_SECS`]
    /// on the generic path, `CADENCE_LLM_TIMEOUT_S` overrides either).
    ///
    /// # Errors
    ///
    /// Returns [`Error::LlmFatal`] when the HTTP client cannot be built.
    pub fn new(config: LlmConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::LlmFatal(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            client,
            config,
            last_stop: RefCell::new(None),
        })
    }

    /// `GET {endpoint}/models`: cheap key/endpoint check for `doctor` (no
    /// generation spend).
    ///
    /// # Errors
    ///
    /// Returns [`Error::LlmFatal`] on auth failures and
    /// [`Error::LlmTransient`] on network/5xx failures.
    pub fn check_api(&self) -> Result<()> {
        let _spinner = LlmSpinner::start(format!("✨ probing {}", self.config.endpoint));
        let url = format!("{}/models", self.config.endpoint);
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .map_err(|e| {
                if e.is_timeout() || e.is_connect() {
                    Error::LlmTransient(format!("LLM API unreachable: {e}"))
                } else {
                    Error::LlmFatal(format!("LLM API check failed: {e}"))
                }
            })?;
        let status = response.status().as_u16();
        if status == 200 {
            Ok(())
        } else if is_retryable_status(status) {
            Err(Error::LlmTransient(format!(
                "LLM API check returned HTTP {status}"
            )))
        } else {
            Err(Error::LlmFatal(format!(
                "LLM API check returned HTTP {status}"
            )))
        }
    }

    /// Map a blocking failure to an outcome (timeout/connect → retryable).
    fn transport_error(error: &reqwest::Error) -> ProviderOutcome {
        if error.is_timeout() || error.is_connect() {
            ProviderOutcome::Retryable {
                message: format!("transport failure: {error}"),
                retry_after: None,
            }
        } else if let Some(status) = error.status() {
            let code = status.as_u16();
            let message = format!("HTTP {code}: {error}");
            if is_retryable_status(code) {
                ProviderOutcome::Retryable {
                    message,
                    retry_after: None,
                }
            } else {
                ProviderOutcome::Fatal(message)
            }
        } else {
            ProviderOutcome::Fatal(format!("request failed: {error}"))
        }
    }

    /// Extract stop diagnostics (`finish_reason` + backend `eos_reason`)
    /// from a chat-completions envelope. Pure and total: missing halves
    /// become `None` rather than failing (older servers omit `eos_reason`).
    fn stop_info_of(body: &serde_json::Value) -> StopInfo {
        let choice = body
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first());
        StopInfo {
            finish_reason: choice
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|reason| reason.as_str())
                .map(ToString::to_string),
            eos_reason: choice
                .and_then(|choice| choice.get("eos_reason"))
                .and_then(|reason| reason.as_str())
                .map(ToString::to_string),
        }
    }

    /// Extract the assistant text from a chat-completions body.
    fn assistant_text(body: &serde_json::Value) -> Option<String> {
        body.get("choices")?
            .as_array()?
            .first()?
            .get("message")?
            .get("content")?
            .as_str()
            .map(ToString::to_string)
    }

    /// Extract a provider error message from a non-2xx JSON body.
    fn provider_message(body: &serde_json::Value) -> Option<String> {
        body.get("error")?
            .get("message")?
            .as_str()
            .map(ToString::to_string)
    }

    /// Truncate a body for error messages without slicing UTF-8 boundaries.
    fn clipped(text: &str) -> String {
        text.chars().take(ERROR_BODY_CHARS).collect()
    }
}

impl RawTransport for HttpLlmProvider {
    fn send(&self, prompt: &str) -> ProviderOutcome {
        let _spinner = LlmSpinner::start(format!(
            "✨ consulting {} {}",
            self.config.provider, self.config.model
        ));
        let url = format!("{}/chat/completions", self.config.endpoint);
        let body = match chat_body(&self.config, prompt) {
            Ok(body) => body,
            Err(e) => return ProviderOutcome::Fatal(e.to_string()),
        };
        let response = match self.client.post(&url).bearer_auth(&self.config.api_key).json(&body).send() {
            Ok(response) => response,
            Err(e) => return Self::transport_error(&e),
        };
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);
        let text = match response.text() {
            Ok(text) => text,
            Err(e) => {
                return ProviderOutcome::Retryable {
                    message: format!("unreadable response body: {e}"),
                    retry_after: None,
                };
            }
        };
        if status == 200 {
            let parsed: serde_json::Value = match serde_json::from_str(&text) {
                Ok(parsed) => parsed,
                Err(e) => {
                    self.last_stop.replace(None);
                    return ProviderOutcome::Retryable {
                        message: format!("invalid JSON in 200 response: {e}"),
                        retry_after: None,
                    };
                }
            };
            self.last_stop.replace(Some(Self::stop_info_of(&parsed)));
            return Self::assistant_text(&parsed).map_or_else(
                || {
                    ProviderOutcome::Fatal(format!(
                        "response has no assistant content: {}",
                        Self::clipped(&text)
                    ))
                },
                ProviderOutcome::Ok,
            );
        }
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        let detail = Self::provider_message(&parsed).unwrap_or_else(|| Self::clipped(&text));
        let message = format!("HTTP {status}: {detail}");
        if is_retryable_status(status) {
            ProviderOutcome::Retryable {
                message,
                retry_after,
            }
        } else {
            ProviderOutcome::Fatal(message)
        }
    }

    fn provider_id(&self) -> &str {
        self.config.provider.as_str()
    }

    fn model_id(&self) -> &str {
        self.config.model.as_str()
    }

    fn last_stop_info(&self) -> Option<StopInfo> {
        self.last_stop.borrow().clone()
    }
}

/// Deterministic scripted transport for unit tests of pure logic
/// (retry/backoff, trap rules, validation). Never for content quality.
#[cfg(test)]
#[derive(Debug)]
pub struct MockLlm {
    outcomes: RefCell<VecDeque<ProviderOutcome>>,
    calls: Cell<usize>,
    provider: String,
    model: String,
}

#[cfg(test)]
impl MockLlm {
    /// Script the next `outcomes.len()` sends; exhausted scripts are fatal.
    #[must_use]
    pub fn new(outcomes: Vec<ProviderOutcome>) -> Self {
        Self {
            outcomes: RefCell::new(outcomes.into_iter().collect()),
            calls: Cell::new(0),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
        }
    }

    /// How many times `send` was called.
    #[must_use]
    pub const fn calls(&self) -> usize {
        self.calls.get()
    }
}

#[cfg(test)]
impl RawTransport for MockLlm {
    fn send(&self, _prompt: &str) -> ProviderOutcome {
        self.calls.set(self.calls.get().saturating_add(1));
        self.outcomes.borrow_mut().pop_front().unwrap_or_else(|| {
            ProviderOutcome::Fatal("mock transport exhausted".to_string())
        })
    }

    fn provider_id(&self) -> &str {
        self.provider.as_str()
    }

    fn model_id(&self) -> &str {
        self.model.as_str()
    }
}

/// Caller-supplied request bundle for [`complete_cached`].
#[derive(Debug, Clone, Copy)]
pub struct CachedRequest<'a> {
    /// Stage label (task type for the cache identity).
    pub operation: &'a str,
    /// Full prompt text.
    pub prompt: &'a str,
    /// Hex hash of the source material (empty for sourceless smoke prompts).
    pub source_hash: &'a str,
    /// Canonical extra params JSON.
    pub params_json: &'a str,
}

/// Injectable execution hooks for [`complete_cached`]: production passes
/// real sleep + wall-clock date; tests pass recorders + fixed dates.
#[derive(Clone, Copy)]
pub struct RunHooks<'a> {
    /// Sleep between transport retries (honors `Retry-After` first).
    pub sleep: &'a dyn Fn(Duration),
    /// ISO date string recorded on job/cache rows.
    pub now_iso: &'a str,
}

/// Outcome of [`complete_cached`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallResult {
    /// Validated response text.
    pub text: String,
    /// True when served from cache without touching the provider.
    pub cache_hit: bool,
    /// Transport sends performed (0 on cache hit).
    pub transport_calls: u32,
    /// Durable job row id (0 on cache hit — no job was needed).
    pub job_id: i64,
}

/// Validate a smoke response: non-empty after trimming. Phase 4+ stages pass
/// stricter validators (schema, trap presence, source membership); cache
/// reads re-run the same validator before accepting a hit.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] when the text is blank.
pub fn validate_smoke(text: &str) -> Result<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::LlmFatal("empty response from provider".to_string()));
    }
    Ok(trimmed.to_string())
}

/// Canonical request JSON stored on cache rows (prompt + params, for audit).
///
/// # Errors
///
/// Returns [`Error::Io`] when serialization fails (practically unreachable).
pub fn request_json(prompt: &str, params_json: &str) -> Result<String> {
    serde_json::to_string(&serde_json::json!({"prompt": prompt, "params": params_json}))
        .map_err(|e| Error::Io(e.to_string()))
}

/// Run one LLM call with cache, durable job, and §16 retry policy.
///
/// - Cache hit + validator passes → return immediately (no job row, no sleep).
/// - Cache hit + validator fails → treated as a miss (stale entry is left in
///   place; a fresh validated write overwrites it on success).
/// - Only validated responses are written to the cache; malformed responses
///   are retried immediately (≤ [`MAX_MALFORMED_ATTEMPTS`], validation error
///   appended, no backoff); transport failures retry with backoff (≤
///   [`MAX_TRANSPORT_ATTEMPTS`] total sends, honoring `Retry-After`).
/// - Every 200-envelope send contributes its [`StopInfo`] line to the
///   attempt row's diagnostics (and to the terminal error on exhaustion),
///   so token-budget cuts and loop-detector kills are visible in the DB.
/// - 4xx/auth failures never retry. Exhaustion fails with
///   `"LLM provider repeatedly failed (last error: ...). State saved — rerun
///   to resume."` after recording `FAILED` on the job row.
///
/// # Errors
///
/// Returns [`Error::LlmFatal`] on validation exhaustion, provider refusal,
/// and transport exhaustion, plus [`Error::Store`] on persistence failures.
#[allow(clippy::too_many_lines)]
pub fn complete_cached(
    transport: &dyn RawTransport,
    store: &mut dyn Store,
    request: &CachedRequest<'_>,
    validate: &dyn Fn(&str) -> Result<String>,
    hooks: &RunHooks<'_>,
) -> Result<CallResult> {
    let identity = RequestIdentity {
        provider: transport.provider_id().to_string(),
        model: transport.model_id().to_string(),
        operation: request.operation.to_string(),
        prompt_version: PROMPT_VERSION.to_string(),
        source_hash: request.source_hash.to_string(),
        params_json: request.params_json.to_string(),
    };
    let hash = identity.cache_hash();

    if let Some(entry) = store.get_llm_cache(&hash)? {
        if let Ok(text) = validate(&entry.response_json) {
            return Ok(CallResult {
                text,
                cache_hit: true,
                transport_calls: 0,
                job_id: 0,
            });
        }
    }

    let stored_json = request_json(request.prompt, request.params_json)?;
    let job = store.create_llm_job(&NewLlmJob {
        operation: identity.operation.clone(),
        provider: identity.provider.clone(),
        model: identity.model.clone(),
        input_hash: input_hash(request.prompt),
        prompt_version: identity.prompt_version.clone(),
    })?;

    let mut current_prompt = request.prompt.to_string();
    let mut transport_failures: u32 = 0;
    let mut malformed_attempts: u32 = 0;
    let mut sends: u32 = 0;

    loop {
        if sends >= MAX_TRANSPORT_ATTEMPTS {
            let last = "transport attempt budget exhausted".to_string();
            fail_job(store, job.id, sends, &last, None)?;
            return exhausted(&last);
        }
        sends = sends.saturating_add(1);
        store.record_llm_attempt(job.id, i64::from(sends), "RUNNING", None, None, None)?;

        match transport.send(&current_prompt) {
            ProviderOutcome::Ok(raw) => match validate(&raw) {
                Ok(text) => {
                    store.put_llm_cache(&crate::store::LlmCacheEntry {
                        cache_hash: hash,
                        operation: identity.operation,
                        provider: identity.provider,
                        model: identity.model,
                        prompt_version: identity.prompt_version,
                        request_json: stored_json,
                        response_json: text.clone(),
                        status: "OK".to_string(),
                        created_at: hooks.now_iso.to_string(),
                    })?;
                    let stop_note = transport.last_stop_info().map(|info| info.describe());
                    store.record_llm_attempt(
                        job.id,
                        i64::from(sends),
                        "OK",
                        Some(raw.as_str()),
                        Some(text.as_str()),
                        stop_note.as_deref(),
                    )?;
                    return Ok(CallResult {
                        text,
                        cache_hit: false,
                        transport_calls: sends,
                        job_id: job.id,
                    });
                }
                Err(validation_error) => {
                    malformed_attempts = malformed_attempts.saturating_add(1);
                    let detail = validation_error.to_string();
                    // Annotate the row (and any terminal error) with how
                    // generation stopped; the retry prompt itself stays
                    // clean — stop facts don't help the model correct itself.
                    let mut row_note = detail.clone();
                    if let Some(note) = transport.last_stop_info().map(|info| info.describe()) {
                        row_note.push_str("; ");
                        row_note.push_str(&note);
                    }
                    store.record_llm_attempt(
                        job.id,
                        i64::from(sends),
                        "RUNNING",
                        Some(raw.as_str()),
                        None,
                        Some(row_note.as_str()),
                    )?;
                    if malformed_attempts >= MAX_MALFORMED_ATTEMPTS {
                        fail_job(store, job.id, sends, &row_note, Some(raw.as_str()))?;
                        return exhausted(&row_note);
                    }
                    current_prompt = format!(
                        "{prompt}\n\nPrevious response failed validation: {detail}. Respond with corrected output.",
                        prompt = request.prompt,
                    );
                }
            },
            ProviderOutcome::Retryable {
                message,
                retry_after,
            } => {
                transport_failures = transport_failures.saturating_add(1);
                store.record_llm_attempt(
                    job.id,
                    i64::from(sends),
                    "RUNNING",
                    None,
                    None,
                    Some(message.as_str()),
                )?;
                if sends >= MAX_TRANSPORT_ATTEMPTS {
                    fail_job(store, job.id, sends, &message, None)?;
                    return exhausted(&message);
                }
                let jitter = clock_jitter_ms(250);
                let delay = retry_after.unwrap_or_else(|| {
                    Duration::from_millis(backoff_delay_ms(
                        BACKOFF_BASE_MS,
                        transport_failures.saturating_sub(1),
                        jitter,
                    ))
                });
                let capped = delay.min(Duration::from_millis(BACKOFF_CAP_MS));
                (hooks.sleep)(capped);
            }
            ProviderOutcome::Fatal(message) => {
                fail_job(store, job.id, sends, &message, None)?;
                return Err(Error::LlmFatal(message));
            }
        }
    }
}

/// Record a terminal `FAILED` attempt on a job row. Keeps the last raw
/// response when one exists so malformed-output failures stay debuggable
/// from the job row instead of vanishing.
///
/// # Errors
///
/// Propagates [`Error::NotFound`] / [`Error::Store`] from the backend.
fn fail_job(
    store: &mut dyn Store,
    job_id: i64,
    sends: u32,
    message: &str,
    raw: Option<&str>,
) -> Result<()> {
    store.record_llm_attempt(
        job_id,
        i64::from(sends),
        "FAILED",
        raw,
        None,
        Some(message),
    )
}

/// §16 hard-failure message: state is on the job row, rerun resumes.
fn exhausted(last_error: &str) -> Result<CallResult> {
    Err(Error::LlmFatal(format!(
        "LLM provider repeatedly failed (last error: {last_error}). State saved — rerun to resume."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;

    fn call(
        transport: &dyn RawTransport,
        store: &mut dyn Store,
        prompt: &str,
        validate: &dyn Fn(&str) -> Result<String>,
        sleeps: &RefCell<Vec<Duration>>,
    ) -> Result<CallResult> {
        let hooks = RunHooks {
            sleep: &|d: Duration| {
                sleeps.borrow_mut().push(d);
            },
            now_iso: "2026-09-20",
        };
        complete_cached(
            transport,
            store,
            &CachedRequest {
                operation: "smoke",
                prompt,
                source_hash: "",
                params_json: "{}",
            },
            validate,
            &hooks,
        )
    }

    #[test]
    fn identity_hash_is_stable_and_sensitive() {
        let base = RequestIdentity {
            provider: "ai-gateway".to_string(),
            model: "m".to_string(),
            operation: "smoke".to_string(),
            prompt_version: PROMPT_VERSION.to_string(),
            source_hash: "s".to_string(),
            params_json: "{}".to_string(),
        };
        assert_eq!(base.cache_hash(), base.cache_hash());
        assert_eq!(base.cache_hash().len(), 64);
        for tweak in [
            RequestIdentity {
                provider: "other".to_string(),
                ..base.clone()
            },
            RequestIdentity {
                model: "other".to_string(),
                ..base.clone()
            },
            RequestIdentity {
                operation: "pretest".to_string(),
                ..base.clone()
            },
            RequestIdentity {
                prompt_version: "v9".to_string(),
                ..base.clone()
            },
            RequestIdentity {
                source_hash: "other".to_string(),
                ..base.clone()
            },
            RequestIdentity {
                params_json: "{\"a\":1}".to_string(),
                ..base.clone()
            },
        ] {
            assert_ne!(tweak.cache_hash(), base.cache_hash());
        }
    }

    #[test]
    fn input_hash_matches_sha256() {
        assert_eq!(
            input_hash("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn retryable_status_matrix() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(503));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(404));
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay_ms(1_000, 0, 0), 1_000);
        assert_eq!(backoff_delay_ms(1_000, 1, 0), 2_000);
        assert_eq!(backoff_delay_ms(1_000, 2, 0), 4_000);
        assert_eq!(backoff_delay_ms(1_000, 3, 250), 8_250);
        assert_eq!(backoff_delay_ms(1_000, 10, 0), BACKOFF_CAP_MS);
        assert_eq!(backoff_delay_ms(1_000, u32::MAX, 0), BACKOFF_CAP_MS);
    }

    #[test]
    fn retry_after_parsing() {
        assert_eq!(parse_retry_after("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_retry_after("  5 "), Some(Duration::from_secs(5)));
        assert_eq!(parse_retry_after("garbage"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(
            parse_retry_after("999999"),
            Some(Duration::from_millis(BACKOFF_CAP_MS))
        );
    }

    #[test]
    fn dotenv_parsing() {
        let text = "# comment\nEMPTY=\nAI_GATEWAY_API_KEY=vck_abc123\nOTHER=1\n";
        assert_eq!(
            parse_dotenv_key(text, "AI_GATEWAY_API_KEY"),
            Some("vck_abc123".to_string())
        );
        assert_eq!(parse_dotenv_key(text, "MISSING"), None);
        assert_eq!(parse_dotenv_key(text, "EMPTY"), None);
        assert_eq!(
            parse_dotenv_key("K=\"quoted val\"\n", "K"),
            Some("quoted val".to_string())
        );
    }

    #[test]
    fn dotenv_parsing_custom_endpoint_without_keys() {
        // Regression: `.env` with commented-out keys plus an endpoint/model
        // pair must still yield a usable custom-path config (endpoint-only
        // selects the generic path; commented keys stay absent).
        let text = "# AI_GATEWAY_API_KEY=vck_secret\n\n# GOOGLE_API_KEY=xyz\n\nCADENCE_LLM_ENDPOINT=http://localhost:5000/v1\nCADENCE_LLM_MODEL=Swift-1.5-Qwen3.8-27B-exl3-3.50bpw\n";
        assert_eq!(parse_dotenv_key(text, "AI_GATEWAY_API_KEY"), None);
        assert_eq!(parse_dotenv_key(text, "GOOGLE_API_KEY"), None);
        let endpoint = parse_dotenv_key(text, "CADENCE_LLM_ENDPOINT").unwrap_or_default();
        let model = parse_dotenv_key(text, "CADENCE_LLM_MODEL").unwrap_or_default();
        assert_eq!(endpoint, "http://localhost:5000/v1");
        assert_eq!(model, "Swift-1.5-Qwen3.8-27B-exl3-3.50bpw");
        let config = LlmConfig::resolve("", "", "", &model, &endpoint).unwrap();
        assert_eq!(config.provider, CUSTOM_PROVIDER_ID);
        assert_eq!(config.endpoint, "http://localhost:5000/v1");
        assert_eq!(config.model, "Swift-1.5-Qwen3.8-27B-exl3-3.50bpw");
    }

    #[test]
    fn smoke_validator_rejects_blank() {
        assert!(validate_smoke("  \n ").is_err());
        assert_eq!(validate_smoke("  ok  ").unwrap(), "ok");
    }

    #[test]
    fn resolve_prefers_gateway_then_google_then_custom() {
        // Gateway key wins with gateway defaults.
        let gateway = LlmConfig::resolve("gw-key", "", "", "", "").unwrap();
        assert_eq!(gateway.provider, PROVIDER_ID);
        assert_eq!(gateway.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(gateway.model, DEFAULT_MODEL);
        assert_eq!(gateway.api_key, "gw-key");
        // Google key selects the Gemini path.
        let google = LlmConfig::resolve("", "g-key", "", "", "").unwrap();
        assert_eq!(google.provider, GEMINI_PROVIDER_ID);
        assert_eq!(google.endpoint, GEMINI_ENDPOINT);
        assert_eq!(google.model, GEMINI_DEFAULT_MODEL);
        assert_eq!(google.api_key, "g-key");
        // Explicit overrides win on either path.
        let over = LlmConfig::resolve("", "g-key", "", "custom-model", "https://x/v1").unwrap();
        assert_eq!(over.model, "custom-model");
        assert_eq!(over.endpoint, "https://x/v1");
        assert_eq!(over.provider, GEMINI_PROVIDER_ID);
        // Custom key selects the generic path (explicit model required).
        let custom = LlmConfig::resolve("", "", "c-key", "my-model", "").unwrap();
        assert_eq!(custom.provider, CUSTOM_PROVIDER_ID);
        assert_eq!(custom.endpoint, CUSTOM_ENDPOINT);
        assert_eq!(custom.model, "my-model");
        assert_eq!(custom.api_key, "c-key");
        // An endpoint override alone also selects it (no-auth local server).
        let local = LlmConfig::resolve("", "", "", "local", "http://localhost:8080/v1").unwrap();
        assert_eq!(local.provider, CUSTOM_PROVIDER_ID);
        assert_eq!(local.endpoint, "http://localhost:8080/v1");
        assert_eq!(local.api_key.len(), 0);
        // Gateway still wins when several keys are set.
        let both = LlmConfig::resolve("gw-key", "g-key", "c-key", "m", "https://x/v1").unwrap();
        assert_eq!(both.provider, PROVIDER_ID);
        // Custom path without a model fails loudly instead of guessing.
        let err = LlmConfig::resolve("", "", "c-key", "", "").unwrap_err();
        assert!(matches!(err, Error::LlmFatal(_)));
        assert!(err.to_string().contains("CADENCE_LLM_MODEL"));
        // Nothing configured: loud failure naming every option.
        let err = LlmConfig::resolve("", "", "", "", "").unwrap_err();
        assert!(matches!(err, Error::LlmFatal(_)));
        assert!(err.to_string().contains("GOOGLE_API_KEY"));
        assert!(err.to_string().contains("CADENCE_API_KEY"));
        // Constrained decoding is opt-in per stage, never a resolve default.
        assert!(gateway.response_format_json.is_none());
        assert!(google.response_format_json.is_none());
        assert!(custom.response_format_json.is_none());
        // Hosted paths get the short timeout, the generic path the long one
        // (slow local decode); thinking stays on unless opted out via env.
        assert_eq!(gateway.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert_eq!(google.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert_eq!(custom.timeout_secs, CUSTOM_TIMEOUT_SECS);
        assert!(!custom.disable_thinking);
        assert_eq!(custom.thinking_effort, "medium");
    }

    #[test]
    fn timeout_and_thinking_env_parsing() {
        assert_eq!(parse_timeout_secs("", 90), 90);
        assert_eq!(parse_timeout_secs("  ", 90), 90);
        assert_eq!(parse_timeout_secs("600", 90), 600);
        assert_eq!(parse_timeout_secs("0", 90), 90);
        assert_eq!(parse_timeout_secs("garbage", 90), 90);
        assert_eq!(parse_timeout_secs("-5", 90), 90);
        assert!(parse_disable_thinking("1"));
        assert!(parse_disable_thinking("TRUE"));
        assert!(parse_disable_thinking(" yes "));
        assert!(!parse_disable_thinking(""));
        assert!(!parse_disable_thinking("0"));
        assert!(!parse_disable_thinking("no"));
        assert_eq!(parse_thinking_effort("low"), "low");
        assert_eq!(parse_thinking_effort(" Medium "), "medium");
        assert_eq!(parse_thinking_effort("HIGH"), "high");
        assert_eq!(parse_thinking_effort("xhigh"), "xhigh");
        // Blank or unrecognized values fall back to medium, never to the
        // server's maximum-effort default.
        assert_eq!(parse_thinking_effort(""), "medium");
        assert_eq!(parse_thinking_effort("ultra"), "medium");
    }

    fn config_for(provider: &str) -> LlmConfig {
        LlmConfig {
            provider: provider.to_string(),
            endpoint: GEMINI_ENDPOINT.to_string(),
            model: GEMINI_DEFAULT_MODEL.to_string(),
            api_key: "k".to_string(),
            max_tokens: 100,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            disable_thinking: false,
            thinking_effort: "medium",
            response_format_json: None,
        }
    }

    #[test]
    fn response_format_envelope_is_strict() {
        let envelope = response_format_envelope("{\"type\":\"object\"}", "probe");
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        assert_eq!(parsed["type"], serde_json::json!("json_schema"));
        assert_eq!(parsed["json_schema"]["name"], serde_json::json!("probe"));
        assert_eq!(parsed["json_schema"]["strict"], serde_json::json!(true));
        assert_eq!(
            parsed["json_schema"]["schema"],
            serde_json::json!({"type": "object"})
        );
    }

    #[test]
    fn schema_tag_is_stable_short_and_sensitive() {
        let first = schema_tag("{\"type\":\"object\"}");
        assert_eq!(first.len(), 8);
        assert_eq!(first, schema_tag("{\"type\":\"object\"}"));
        assert_ne!(first, schema_tag("{\"type\":\"array\"}"));
    }

    #[test]
    fn chat_body_attaches_format_everywhere_but_gateway() {
        let format = response_format_envelope("{\"type\":\"object\"}", "probe");
        let mut google = config_for(GEMINI_PROVIDER_ID);
        google.response_format_json = Some(format.clone());
        let google_body = chat_body(&google, "hi").unwrap();
        assert_eq!(
            google_body["response_format"]["json_schema"]["name"],
            serde_json::json!("probe")
        );
        // Generic path carries the same constraint (llama.cpp accepts the
        // standard `json_schema` envelope; probe 2026-09-25).
        let mut custom = config_for(CUSTOM_PROVIDER_ID);
        custom.response_format_json = Some(format.clone());
        assert_eq!(
            chat_body(&custom, "hi").unwrap()["response_format"]["json_schema"]["name"],
            serde_json::json!("probe")
        );
        // Gateway path never carries the constraint, even when configured.
        let mut gateway = config_for(PROVIDER_ID);
        gateway.response_format_json = Some(format);
        assert!(chat_body(&gateway, "hi").unwrap().get("response_format").is_none());
        // Thinking controls are custom-path only and on at medium by
        // default (server defaults vary by model, so the body is explicit).
        assert_eq!(
            chat_body(&config_for(CUSTOM_PROVIDER_ID), "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": true, "thinking_effort": "medium"})
        );
        let mut high = config_for(CUSTOM_PROVIDER_ID);
        high.thinking_effort = "high";
        assert_eq!(
            chat_body(&high, "hi").unwrap()["chat_template_kwargs"]["thinking_effort"],
            serde_json::json!("high")
        );
        // Opt-out sends only the off switch, never an effort alongside it.
        let mut no_think = config_for(CUSTOM_PROVIDER_ID);
        no_think.disable_thinking = true;
        assert_eq!(
            chat_body(&no_think, "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false})
        );
        let mut gateway_think = config_for(PROVIDER_ID);
        gateway_think.disable_thinking = true;
        assert!(chat_body(&gateway_think, "hi").unwrap().get("chat_template_kwargs").is_none());
        // Hosted paths never carry template kwargs (non-portable extension).
        assert!(chat_body(&config_for(GEMINI_PROVIDER_ID), "hi").unwrap().get("chat_template_kwargs").is_none());
        let mut google_think = config_for(GEMINI_PROVIDER_ID);
        google_think.disable_thinking = true;
        assert!(chat_body(&google_think, "hi").unwrap().get("chat_template_kwargs").is_none());
        // Unset format means an unconstrained body on either path.
        assert!(chat_body(&config_for(GEMINI_PROVIDER_ID), "hi").unwrap().get("response_format").is_none());
        // Corrupt format fails loudly instead of silently unconstrained.
        let mut broken = config_for(GEMINI_PROVIDER_ID);
        broken.response_format_json = Some("{broken".to_string());
        assert!(chat_body(&broken, "hi").is_err());
    }

    #[test]
    fn gemma_thinking_stays_off() {
        assert!(model_disables_thinking("gemma-4-26B-A4B-it-exl3-3.10bpw"));
        assert!(model_disables_thinking("Gemma-3-27B-IT"));
        assert!(!model_disables_thinking(
            "Swift-1.5-Qwen3.8-27B-exl3-3.50bpw"
        ));
        assert!(!model_disables_thinking("Qwen3.8-27B-exl3-3.00bpw"));
        assert!(!model_disables_thinking("gemini-3.6-flash"));
        // Gemma on the custom path: thinking off even at default settings
        // (thinking-on corrupts its long structured output; 2026-09-28).
        let mut gemma = config_for(CUSTOM_PROVIDER_ID);
        gemma.model = "gemma-4-26B-A4B-it-exl3-3.10bpw".to_string();
        assert_eq!(
            chat_body(&gemma, "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false})
        );
        // Effort setting must not leak through for Gemma either.
        gemma.thinking_effort = "xhigh";
        assert_eq!(
            chat_body(&gemma, "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false})
        );
        // Qwen keeps the explicit budget.
        let mut qwen = config_for(CUSTOM_PROVIDER_ID);
        qwen.model = "Swift-1.5-Qwen3.8-27B-exl3-3.50bpw".to_string();
        assert_eq!(
            chat_body(&qwen, "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": true, "thinking_effort": "medium"})
        );
        // Opt-out still sends the off switch for non-Gemma models too.
        qwen.disable_thinking = true;
        assert_eq!(
            chat_body(&qwen, "hi").unwrap()["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false})
        );
    }

    #[test]
    fn stop_info_extraction_is_total() {
        let full = serde_json::json!({"choices": [{
            "finish_reason": "length",
            "eos_reason": "loop",
            "message": {"content": "x"},
        }]});
        assert_eq!(
            HttpLlmProvider::stop_info_of(&full),
            StopInfo {
                finish_reason: Some("length".to_string()),
                eos_reason: Some("loop".to_string()),
            }
        );
        // Older servers omit `eos_reason`; halves degrade to None, never panic.
        let no_eos = serde_json::json!({"choices": [{
            "finish_reason": "stop",
            "message": {"content": "x"},
        }]});
        assert_eq!(
            HttpLlmProvider::stop_info_of(&no_eos),
            StopInfo {
                finish_reason: Some("stop".to_string()),
                eos_reason: None,
            }
        );
        for broken in [
            serde_json::json!({}),
            serde_json::json!({"choices": []}),
            serde_json::json!({"choices": [{"finish_reason": 7}]}),
            serde_json::json!({"choices": [{"eos_reason": serde_json::Value::Null}]}),
        ] {
            assert_eq!(
                HttpLlmProvider::stop_info_of(&broken),
                StopInfo::default(),
                "{broken}",
            );
        }
    }

    #[test]
    fn stop_info_describe_renders_missing_halves() {
        assert_eq!(
            StopInfo {
                finish_reason: Some("length".to_string()),
                eos_reason: Some("loop".to_string()),
            }
            .describe(),
            "stop finish=length eos=loop"
        );
        assert_eq!(StopInfo::default().describe(), "stop finish=- eos=-");
        assert_eq!(
            StopInfo {
                finish_reason: Some("stop".to_string()),
                eos_reason: None,
            }
            .describe(),
            "stop finish=stop eos=-"
        );
    }

    /// Stub transport with canned stop diagnostics behind
    /// [`RawTransport::last_stop_info`].
    struct StopStub {
        stop: StopInfo,
        calls: Cell<usize>,
        provider: String,
        model: String,
    }

    impl RawTransport for StopStub {
        fn send(&self, _prompt: &str) -> ProviderOutcome {
            self.calls.set(self.calls.get().saturating_add(1));
            ProviderOutcome::Ok("junk".to_string())
        }

        fn provider_id(&self) -> &str {
            self.provider.as_str()
        }

        fn model_id(&self) -> &str {
            self.model.as_str()
        }

        fn last_stop_info(&self) -> Option<StopInfo> {
            Some(self.stop.clone())
        }
    }

    #[test]
    fn malformed_exhaustion_carries_stop_note() {
        let stub = StopStub {
            stop: StopInfo {
                finish_reason: Some("length".to_string()),
                eos_reason: Some("loop".to_string()),
            },
            calls: Cell::new(0),
            provider: "stub".to_string(),
            model: "stub-model".to_string(),
        };
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let reject = |_: &str| Err(Error::LlmFatal("bad".to_string()));
        let err = call(&stub, &mut store, "p-stop", &reject, &sleeps).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("bad"), "{text}");
        assert!(
            text.contains("stop finish=length eos=loop"),
            "{text}"
        );
        // Still exactly the fast malformed budget: annotation adds no sends.
        assert_eq!(stub.calls.get(), 3);
        assert!(sleeps.borrow().is_empty());
    }

    #[test]
    fn mock_transport_carries_no_stop_note() {
        // `MockLlm` uses the default `last_stop_info` (None): the success
        // path must record attempts exactly as before (no annotation).
        assert!(MockLlm::new(vec![]).last_stop_info().is_none());
    }

    #[test]
    fn cache_hit_skips_transport() {
        let transport = MockLlm::new(vec![ProviderOutcome::Ok("fresh".to_string())]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let ok = |s: &str| Ok(s.to_string());
        let first = call(&transport, &mut store, "p1", &ok, &sleeps).unwrap();
        assert!(!first.cache_hit);
        assert_eq!(first.transport_calls, 1);
        // Second call with an empty script: must serve from cache.
        let second = call(&transport, &mut store, "p1", &ok, &sleeps).unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.text, "fresh");
        assert_eq!(transport.calls(), 1);
        assert!(sleeps.borrow().is_empty());
    }

    #[test]
    fn invalid_cache_entry_is_a_miss() {
        let transport = MockLlm::new(vec![
            ProviderOutcome::Ok("good".to_string()),
            ProviderOutcome::Ok("good".to_string()),
            ProviderOutcome::Ok("good".to_string()),
        ]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        // Validator rejects "good" on the first pass only via call counting.
        let strict = |s: &str| {
            if s == "good" {
                Err(Error::LlmFatal("not good enough".to_string()))
            } else {
                Ok(s.to_string())
            }
        };
        // Seed the cache directly with a stale-shaped entry.
        let identity = RequestIdentity {
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            operation: "smoke".to_string(),
            prompt_version: PROMPT_VERSION.to_string(),
            source_hash: String::new(),
            params_json: "{}".to_string(),
        };
        store
            .put_llm_cache(&crate::store::LlmCacheEntry {
                cache_hash: identity.cache_hash(),
                operation: "smoke".to_string(),
                provider: "mock".to_string(),
                model: "mock-model".to_string(),
                prompt_version: PROMPT_VERSION.to_string(),
                request_json: "{}".to_string(),
                response_json: "good".to_string(),
                status: "OK".to_string(),
                created_at: "2026-09-20".to_string(),
            })
            .unwrap();
        // Stale entry fails validation → miss → transport → still invalid →
        // malformed retries exhaust without sleeping.
        let err = call(&transport, &mut store, "p9", &strict, &sleeps).unwrap_err();
        assert!(err.to_string().contains("repeatedly failed"));
        assert_eq!(transport.calls(), 3);
        assert!(sleeps.borrow().is_empty());
    }

    #[test]
    fn malformed_retries_fast_without_backoff() {
        let transport = MockLlm::new(vec![
            ProviderOutcome::Ok(String::new()),
            ProviderOutcome::Ok(String::new()),
            ProviderOutcome::Ok(String::new()),
        ]);
        // Only non-blank passes; first two sends fail validation.
        let validate = |s: &str| validate_smoke(s);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let err = call(&transport, &mut store, "p2", &validate, &sleeps).unwrap_err();
        assert!(err.to_string().contains("repeatedly failed"));
        assert_eq!(transport.calls(), 3);
        assert!(sleeps.borrow().is_empty());
        // Nothing malformed was cached.
        assert!(store.get_llm_cache("whatever").unwrap().is_none());
    }

    #[test]
    fn transport_retry_uses_backoff_then_succeeds() {
        let transport = MockLlm::new(vec![
            ProviderOutcome::Retryable {
                message: "busy".to_string(),
                retry_after: None,
            },
            ProviderOutcome::Ok("recovered".to_string()),
        ]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let ok = |s: &str| Ok(s.to_string());
        let result = call(&transport, &mut store, "p3", &ok, &sleeps).unwrap();
        assert_eq!(result.text, "recovered");
        assert_eq!(result.transport_calls, 2);
        assert_eq!(sleeps.borrow().len(), 1);
        let slept = sleeps.borrow()[0];
        assert!(
            slept >= Duration::from_millis(1_000)
                && slept <= Duration::from_millis(1_250),
            "first backoff should be base + small jitter, got {slept:?}"
        );
    }

    #[test]
    fn retry_after_header_beats_computed_backoff() {
        let transport = MockLlm::new(vec![
            ProviderOutcome::Retryable {
                message: "slow down".to_string(),
                retry_after: Some(Duration::from_secs(7)),
            },
            ProviderOutcome::Ok("late".to_string()),
        ]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let ok = |s: &str| Ok(s.to_string());
        let result = call(&transport, &mut store, "p4", &ok, &sleeps).unwrap();
        assert_eq!(result.text, "late");
        assert_eq!(sleeps.borrow().as_slice(), [Duration::from_secs(7)]);
    }

    #[test]
    fn fatal_never_retries() {
        let transport = MockLlm::new(vec![ProviderOutcome::Fatal("HTTP 401: bad key".to_string())]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let ok = |s: &str| Ok(s.to_string());
        let err = call(&transport, &mut store, "p5", &ok, &sleeps).unwrap_err();
        assert!(matches!(err, Error::LlmFatal(_)));
        assert!(err.to_string().contains("bad key"));
        assert_eq!(transport.calls(), 1);
        assert!(sleeps.borrow().is_empty());
    }

    #[test]
    fn transport_exhaustion_reports_last_error() {
        let transport = MockLlm::new(vec![
            ProviderOutcome::Retryable { message: "e0".to_string(), retry_after: None },
            ProviderOutcome::Retryable { message: "e1".to_string(), retry_after: None },
            ProviderOutcome::Retryable { message: "e2".to_string(), retry_after: None },
            ProviderOutcome::Retryable { message: "e3".to_string(), retry_after: None },
            ProviderOutcome::Retryable { message: "e4-final".to_string(), retry_after: None },
        ]);
        let mut store = MemoryStore::new();
        let sleeps = RefCell::new(Vec::new());
        let ok = |s: &str| Ok(s.to_string());
        let err = call(&transport, &mut store, "p6", &ok, &sleeps).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("repeatedly failed"), "{text}");
        assert!(text.contains("e4-final"), "{text}");
        assert!(text.contains("rerun to resume"), "{text}");
        assert_eq!(transport.calls(), 5);
        assert_eq!(sleeps.borrow().len(), 4);
    }

    #[test]
    fn clock_jitter_stays_in_bounds() {
        assert_eq!(clock_jitter_ms(0), 0);
        for _ in 0..50 {
            assert!(clock_jitter_ms(250) <= 250);
        }
    }

    #[test]
    fn spinner_frames_cycle_without_panic() {
        let first = spinner_frame_at(0);
        assert!(!first.is_empty(), "frames must render something");
        // Full rotation wraps back to the start.
        assert_eq!(spinner_frame_at(SPINNER_FRAMES.len()), first);
        // Huge indices wrap instead of panicking (no indexing/slicing).
        assert_eq!(
            spinner_frame_at(usize::MAX),
            spinner_frame_at(usize::MAX.checked_rem(SPINNER_FRAMES.len()).unwrap_or_default())
        );
        // Every frame in the rotation is non-empty.
        for index in 0..SPINNER_FRAMES.len().saturating_mul(2) {
            assert!(!spinner_frame_at(index).is_empty(), "frame {index}");
        }
    }

    #[test]
    fn spinner_suppression_parses_env() {
        assert!(spinner_suppressed("1", "xterm"));
        assert!(spinner_suppressed("true", "xterm"));
        assert!(spinner_suppressed("yes", "xterm"));
        assert!(spinner_suppressed("", "dumb"));
        assert!(spinner_suppressed("", "DUMB"));
        assert!(spinner_suppressed("", " dumb "));
        assert!(!spinner_suppressed("", "xterm"));
        assert!(!spinner_suppressed("0", "xterm"));
        assert!(!spinner_suppressed("", ""));
    }

    #[test]
    fn spinner_line_formats_elapsed() {
        assert_eq!(
            format_spinner_line("⠋", "✨ consulting model", 3),
            "⠋ ✨ consulting model (3s)…"
        );
        assert_eq!(
            format_spinner_line("⠏", "✨ probing endpoint", 0),
            "⠏ ✨ probing endpoint (0s)…"
        );
    }

    #[test]
    fn inactive_spinner_is_noop() {
        let guard = LlmSpinner::inactive();
        assert!(!guard.is_active());
        // Drop of the inactive guard must not write anything or panic.
    }
}
