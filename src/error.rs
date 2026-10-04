//! Application-wide error type.

use thiserror::Error;

/// All recoverable failures in `cadence`.
#[derive(Debug, Error)]
pub enum Error {
    /// SQLite / persistence failure.
    #[error("store error: {0}")]
    Store(String),
    /// Configuration failure.
    #[error("config error: {0}")]
    Config(String),
    /// Invalid state transition or scheduling invariant violation.
    #[error("invalid transition: {0}")]
    InvalidTransition(String),
    /// Requested entity was not found.
    #[error("not found: {0}")]
    NotFound(String),
    /// Database is already open in another process.
    #[error("already open: {0}")]
    AlreadyOpen(String),
    /// Task was already completed; state is unchanged.
    #[error("already completed: {0}")]
    AlreadyCompleted(String),
    /// Required external tool is missing.
    #[error("missing dependency: {0}")]
    MissingDependency(String),
    /// Invalid user input / CLI usage.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// PDF parsing / extraction failure.
    #[error("pdf error: {0}")]
    Pdf(String),
    /// A unit exceeds the page cap with no semantic split; the user must
    /// supply manual boundaries (`--manual-boundaries` or interactive stdin).
    #[error("manual boundaries required: {0}")]
    NeedsManual(String),
    /// Transient LLM failure (429, 5xx, timeout): safe to retry with backoff.
    #[error("temporary LLM failure: {0}")]
    LlmTransient(String),
    /// Non-retryable LLM failure (4xx auth/config, exhausted retries,
    /// malformed input): fail immediately.
    #[error("LLM error: {0}")]
    LlmFatal(String),
    /// I/O failure.
    #[error("io error: {0}")]
    Io(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Map rusqlite errors into [`Error::Store`].
impl From<rusqlite::Error> for Error {
    fn from(value: rusqlite::Error) -> Self {
        Self::Store(value.to_string())
    }
}

/// Map I/O errors into [`Error::Io`].
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
